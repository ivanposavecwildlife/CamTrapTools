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
    """Return one row per deployment with ``deploymentStart`` and ``deploymentEnd``.

    Bounds are derived from media timestamps; implausible timestamps (see
    :mod:`camtraptools.check`) should not define the deployment period.
    """
    raise NotImplementedError


def run(images_dir: Path, out_dir: Path) -> None:
    """Write ``media.csv`` and ``deployments.csv`` to ``out_dir``."""
    raise NotImplementedError("`ingest` is not implemented yet.")
