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


def build_media(images_dir: Path) -> pd.DataFrame:
    """Return one row per image; each top-level subfolder of ``images_dir`` is one deployment."""
    rows = []
    for deployment_dir in sorted(images_dir.iterdir()):
        if not deployment_dir.is_dir() or deployment_dir.name.startswith("."):
            continue
        for image_path in find_images(deployment_dir):
            rows.append(media_row(image_path, images_dir, deployment_dir.name))
    return pd.DataFrame(rows, columns=MEDIA_COLUMNS)


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
    images_dir: Path, out_dir: Path, field_sheet: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build the media and deployments tables, merge the optional field sheet,
    check it, and write the tables as CSV files to ``out_dir``.

    Returns the two tables and a list of field-sheet issues.
    """
    media = build_media(images_dir)
    deployments = build_deployments(media)
    issues = []
    if field_sheet is not None:
        sheet = parse_recorded_periods(read_field_sheet(field_sheet))
        issues += check_field_sheet(sheet)
        issues += match_deployments(sheet, media)
        issues += check_recorded_periods(sheet, media)
        deployments = add_field_sheet(deployments, sheet)
    out_dir.mkdir(parents=True, exist_ok=True)
    media.to_csv(out_dir / "media.csv", index=False, date_format=TIMESTAMP_FORMAT)
    deployments.to_csv(
        out_dir / "deployments.csv", index=False, date_format=TIMESTAMP_FORMAT
    )
    return media, deployments, issues


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
