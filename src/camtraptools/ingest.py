"""Build a Camtrap DP package from a directory of camera-trap images.

Camtrap DP specification: https://camtrap-dp.tdwg.org

Expected input layout, one subdirectory per deployment:

    images/
        camera_A/
            IMG_0001.JPG
            ...
        camera_B/
            ...
"""

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import ExifTags, Image


def read_timestamp(image_path: Path) -> datetime | None:
    """Return the capture time from the image's EXIF ``DateTimeOriginal`` tag.

    Returns ``None`` if the tag is missing or the file cannot be read; corrupt
    files are common on field SD cards and must not abort ingestion.
    """
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
    except OSError:
        return None

    value = exif.get_ifd(ExifTags.IFD.Exif).get(ExifTags.Base.DateTimeOriginal)
    if value is None:
        value = exif.get(ExifTags.Base.DateTime)
    if value is None:
        return None

    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def is_image(path: Path) -> bool:
    """Return True for image files, skipping hidden files such as the
    ``._IMG_0001.JPG`` copies macOS writes to SD cards."""
    return path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")


def find_images(deployment_dir: Path) -> list[Path]:
    """Return every image under ``deployment_dir``, including subfolders, sorted by path."""
    images = []
    for path in deployment_dir.rglob("*"):
        if path.is_file() and is_image(path):
            images.append(path)
    return sorted(images)


def media_row(image_path: Path, images_dir: Path, deployment_id: str) -> dict:
    """Return the Camtrap DP media record for a single image."""
    relative_path = image_path.relative_to(images_dir).as_posix()
    return {
        "mediaID": relative_path,
        "deploymentID": deployment_id,
        "timestamp": read_timestamp(image_path),
        "filePath": relative_path,
        "fileName": image_path.name,
        "fileMediatype": "image/jpeg",
    }


MEDIA_COLUMNS = [
    "mediaID",
    "deploymentID",
    "timestamp",
    "filePath",
    "fileName",
    "fileMediatype",
]


def deployment_folders(images_dir: Path, depth: int = 1) -> list[tuple[str, Path]]:
    """Return the deployment ID and folder of every deployment: the folders ``depth``
    levels below ``images_dir``, skipping hidden ones. With a depth of 2,
    ``north/camera_A`` becomes the deployment ``north_camera_A``."""
    folders = [images_dir]
    for _ in range(depth):
        folders = [
            child
            for folder in folders
            for child in sorted(folder.iterdir())
            if child.is_dir() and not child.name.startswith(".")
        ]
    return [("_".join(folder.relative_to(images_dir).parts), folder) for folder in folders]


def build_media(images_dir: Path, depth: int = 1) -> pd.DataFrame:
    """Return one row per image in the deployment folders of ``images_dir`` (see
    ``deployment_folders``)."""
    rows = []
    for deployment_id, deployment_dir in deployment_folders(images_dir, depth):
        for image_path in find_images(deployment_dir):
            rows.append(media_row(image_path, images_dir, deployment_id))
    return pd.DataFrame(rows, columns=MEDIA_COLUMNS)


OUTSIDE_DEPLOYMENTS = "JPEG outside deployment folders"


def skipped_files(images_dir: Path, depth: int = 1) -> Counter:
    """Return the files under ``images_dir`` that ``build_media`` does not read, counted
    by extension (such as ``.mp4`` videos), and JPEG images that are not inside a
    deployment folder. Hidden files and folders are not counted."""
    deployments = {folder for _, folder in deployment_folders(images_dir, depth)}
    skipped = Counter()
    for path in images_dir.rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(images_dir).parts):
            continue
        if not is_image(path):
            skipped[path.suffix.lower() or "(no extension)"] += 1
        elif not deployments.intersection(path.parents):
            skipped[OUTSIDE_DEPLOYMENTS] += 1
    return skipped


def build_deployments(media: pd.DataFrame) -> pd.DataFrame:
    """Return one row per deployment, spanning its first and last timestamped image."""
    timestamps = media.groupby("deploymentID")["timestamp"]
    deployments = pd.DataFrame(
        {
            "deploymentStart": timestamps.min(),
            "deploymentEnd": timestamps.max(),
        }
    )
    return deployments.reset_index()


TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


def run(
    images_dir: Path, out_dir: Path, field_sheet: Path | None = None, depth: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, pd.Timedelta]]:
    """Build the media and deployments tables, merge the optional field sheet,
    check it, and write the tables as CSV files to ``out_dir``. Deployments are the
    folders ``depth`` levels below ``images_dir``.

    If the field sheet gives a deployment's ``cameraTime`` and ``actualTime``, its
    image times are corrected first (see ``correct_clocks``).

    Returns the two tables, a list of field-sheet issues and the clock correction
    applied to each corrected deployment.
    """
    media = build_media(images_dir, depth)
    issues = []
    corrections = {}
    if field_sheet is not None:
        sheet = parse_recorded_periods(read_field_sheet(field_sheet))
        corrections, clock_issues = clock_offsets(sheet)
        media = correct_clocks(media, corrections)
        issues += check_field_sheet(sheet) + clock_issues
        issues += match_deployments(sheet, media)
        issues += check_recorded_periods(sheet, media)
        deployments = add_field_sheet(build_deployments(media), sheet.drop(columns=CLOCK_COLUMNS, errors="ignore"))
    else:
        deployments = build_deployments(media)
    out_dir.mkdir(parents=True, exist_ok=True)
    media.to_csv(out_dir / "media.csv", index=False, date_format=TIMESTAMP_FORMAT)
    deployments.to_csv(
        out_dir / "deployments.csv", index=False, date_format=TIMESTAMP_FORMAT
    )
    return media, deployments, issues, corrections


CLOCK_COLUMNS = ["cameraTime", "actualTime"]


def clock_offsets(sheet: pd.DataFrame) -> tuple[dict[str, pd.Timedelta], list[str]]:
    """Return how far each deployment's camera clock was off, from the field sheet's
    optional ``cameraTime`` (what the camera showed, e.g. on a photo of a phone clock)
    and ``actualTime`` (the real time at that moment) columns, and problems with them."""
    if not set(CLOCK_COLUMNS) & set(sheet.columns):
        return {}, []
    offsets, issues = {}, []
    for row in sheet.drop_duplicates("deploymentID").to_dict("records"):
        values = [row.get(column) for column in CLOCK_COLUMNS]
        given = [isinstance(value, str) and value.strip() or (not pd.isna(value) and value) for value in values]
        if not any(given):
            continue
        if not all(given):
            issues.append(f"{row['deploymentID']}: clock correction needs both cameraTime and actualTime")
            continue
        camera, actual = (pd.to_datetime(value, format="ISO8601", errors="coerce") for value in values)
        if pd.isna(camera) or pd.isna(actual):
            issues.append(f"{row['deploymentID']}: unreadable cameraTime or actualTime")
        elif actual != camera:
            offsets[row["deploymentID"]] = actual - camera
    return offsets, issues


def format_offset(offset: pd.Timedelta) -> str:
    """Return a clock correction such as ``+1:00:00`` or ``-3 days 02:15:00``."""
    sign = "-" if offset < pd.Timedelta(0) else "+"
    seconds = int(abs(offset).total_seconds())
    days, rest = divmod(seconds, 86400)
    clock = f"{rest // 3600}:{rest % 3600 // 60:02d}:{rest % 60:02d}"
    return f"{sign}{days} days {clock}" if days else f"{sign}{clock}"


def correct_clocks(media: pd.DataFrame, offsets: dict[str, pd.Timedelta]) -> pd.DataFrame:
    """Return the media with each corrected deployment's timestamps shifted by its
    offset, and the camera's original time kept in ``mediaComments``."""
    if not offsets:
        return media
    media = media.copy()
    if "mediaComments" not in media.columns:
        media["mediaComments"] = None
    for deployment_id, offset in offsets.items():
        rows = (media["deploymentID"] == deployment_id) & media["timestamp"].notna()
        original = media.loc[rows, "timestamp"]
        media.loc[rows, "timestamp"] = original + offset
        media.loc[rows, "mediaComments"] = [
            f"Camera clock corrected by {format_offset(offset)}; camera time {time.isoformat()}" for time in original
        ]
    return media


REQUIRED_FIELD_SHEET_COLUMNS = ["deploymentID", "latitude", "longitude"]


def read_field_sheet(path: Path) -> pd.DataFrame:
    """Read a field sheet (CSV or Excel) with one row per deployment."""
    if path.suffix.lower() == ".xlsx":
        sheet = pd.read_excel(path)
    else:
        sheet = pd.read_csv(path)
    sheet.columns = sheet.columns.str.strip()
    missing = [
        column for column in REQUIRED_FIELD_SHEET_COLUMNS if column not in sheet.columns
    ]
    if missing:
        raise ValueError(
            f"Field sheet is missing required columns: {', '.join(missing)}"
        )
    sheet["deploymentID"] = sheet["deploymentID"].astype(str).str.strip()
    sheet["latitude"] = pd.to_numeric(sheet["latitude"], errors="coerce")
    sheet["longitude"] = pd.to_numeric(sheet["longitude"], errors="coerce")
    return sheet


def check_field_sheet(sheet: pd.DataFrame) -> list[str]:
    """Return problems in the field sheet itself: duplicate IDs and invalid coordinates."""
    issues = []
    duplicates = sheet.loc[sheet["deploymentID"].duplicated(), "deploymentID"]
    for deployment_id in sorted(set(duplicates)):
        issues.append(f"{deployment_id}: listed more than once in the field sheet")
    for row in sheet.itertuples():
        if pd.isna(row.latitude) or pd.isna(row.longitude):
            issues.append(f"{row.deploymentID}: missing or unreadable coordinates")
        elif not (-90 <= row.latitude <= 90 and -180 <= row.longitude <= 180):
            issues.append(
                f"{row.deploymentID}: impossible coordinates ({row.latitude}, {row.longitude})"
            )
    return issues


def normalize_id(deployment_id: str) -> str:
    """Reduce an ID to lowercase letters and numbers without leading zeros,
    so that ``CAM_03``, ``cam-3`` and ``Cam3`` all become ``cam3``."""
    parts = re.findall(r"[a-z]+|[0-9]+", deployment_id.lower())
    normalized = []
    for part in parts:
        if part.isdigit():
            part = str(int(part))
        normalized.append(part)
    return "".join(normalized)


def match_deployments(sheet: pd.DataFrame, media: pd.DataFrame) -> list[str]:
    """Return mismatches between field-sheet deployment IDs and image folders."""
    issues = []
    sheet_ids = set(sheet["deploymentID"])
    folder_ids = set(media["deploymentID"])
    sheet_ids_by_normalized = {
        normalize_id(sheet_id): sheet_id for sheet_id in sheet_ids
    }
    for deployment_id in sorted(folder_ids - sheet_ids):
        match = sheet_ids_by_normalized.get(normalize_id(deployment_id))
        hint = f" (did you mean '{match}'?)" if match else ""
        issues.append(
            f"{deployment_id}: has images but is not in the field sheet{hint}"
        )
    for deployment_id in sorted(sheet_ids - folder_ids):
        issues.append(f"{deployment_id}: in the field sheet but has no images")
    return issues


def add_field_sheet(deployments: pd.DataFrame, sheet: pd.DataFrame) -> pd.DataFrame:
    """Add the field sheet's columns (coordinates, camera details, notes) to each deployment.

    Recorded ``deploymentStart`` and ``deploymentEnd`` values take priority over
    the dates of the first and last image.
    """
    sheet = sheet.drop_duplicates("deploymentID")
    merged = deployments.merge(
        sheet, on="deploymentID", how="left", suffixes=("", "_recorded")
    )
    for column in ["deploymentStart", "deploymentEnd"]:
        recorded = f"{column}_recorded"
        if recorded in merged.columns:
            merged[column] = merged[recorded].fillna(merged[column])
            merged = merged.drop(columns=recorded)
    return merged


def parse_recorded_periods(sheet: pd.DataFrame) -> pd.DataFrame:
    """Convert the field sheet's optional ``deploymentStart`` and ``deploymentEnd``
    columns (ISO 8601, e.g. ``2024-05-01`` or ``2024-05-01T09:15``) to dates.
    An end date without a time means the end of that day."""
    if "deploymentStart" in sheet.columns:
        sheet["deploymentStart"] = pd.to_datetime(
            sheet["deploymentStart"], format="ISO8601", errors="coerce"
        )
    if "deploymentEnd" in sheet.columns:
        end = pd.to_datetime(sheet["deploymentEnd"], format="ISO8601", errors="coerce")
        date_only = end == end.dt.normalize()
        end[date_only] = end[date_only] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        sheet["deploymentEnd"] = end
    return sheet


def check_recorded_periods(sheet: pd.DataFrame, media: pd.DataFrame) -> list[str]:
    """Return problems with recorded deployment periods: missing or unreadable dates,
    a start after its end, and images taken outside the recorded period."""
    issues = []
    if "deploymentStart" not in sheet.columns or "deploymentEnd" not in sheet.columns:
        return issues
    periods = sheet.drop_duplicates("deploymentID").set_index("deploymentID")
    for row in periods.itertuples():
        if pd.isna(row.deploymentStart):
            issues.append(
                f"{row.Index}: no readable recorded start; using the first image instead"
            )
        if pd.isna(row.deploymentEnd):
            issues.append(
                f"{row.Index}: no readable recorded end; using the last image instead"
            )
        if row.deploymentStart > row.deploymentEnd:
            issues.append(f"{row.Index}: recorded start is after the recorded end")
    images = media.join(
        periods[["deploymentStart", "deploymentEnd"]], on="deploymentID", how="inner"
    )
    before = (
        images[images["timestamp"] < images["deploymentStart"]]
        .groupby("deploymentID")
        .size()
    )
    after = (
        images[images["timestamp"] > images["deploymentEnd"]]
        .groupby("deploymentID")
        .size()
    )
    for deployment_id, count in before.items():
        issues.append(
            f"{deployment_id}: {count} image(s) taken before the recorded start"
        )
    for deployment_id, count in after.items():
        issues.append(f"{deployment_id}: {count} image(s) taken after the recorded end")
    return issues
