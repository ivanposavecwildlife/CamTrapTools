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


def read_timestamp(image_path: Path) -> datetime | None:
    """Return the capture time from the image's EXIF ``DateTimeOriginal`` tag.

    Returns ``None`` if the tag is missing or the file cannot be read; corrupt
    files are common on field SD cards and must not abort ingestion.
    """
    raise NotImplementedError


def build_media(images_dir: Path) -> pd.DataFrame:
    """Return one row per image with the Camtrap DP media fields.

    Required columns: ``mediaID``, ``deploymentID`` (subdirectory name),
    ``timestamp``, ``filePath``, ``fileName``, ``fileMediatype``.
    """
    raise NotImplementedError


def build_deployments(media: pd.DataFrame) -> pd.DataFrame:
    """Return one row per deployment with ``deploymentStart`` and ``deploymentEnd``.

    Bounds are derived from media timestamps; implausible timestamps (see
    :mod:`traptools.check`) should not define the deployment period.
    """
    raise NotImplementedError


def run(images_dir: Path, out_dir: Path) -> None:
    """Write ``media.csv`` and ``deployments.csv`` to ``out_dir``."""
    raise NotImplementedError("`ingest` is not implemented yet.")
