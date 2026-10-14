"""Module 1a: turn a folder of SD-card images into a Camtrap DP package.

Camtrap DP spec: https://camtrap-dp.tdwg.org

Expected input layout (one subfolder per camera deployment):

    images/
        camera_A/
            IMG_0001.JPG
            ...
        camera_B/
            ...
"""

from pathlib import Path

import pandas as pd


def read_timestamp(image_path: Path):
    """Return when the photo was taken, read from its EXIF metadata.

    TODO (week 5):
    - Open the image with Pillow and read the EXIF tag DateTimeOriginal.
    - Convert the "YYYY:MM:DD HH:MM:SS" string to a datetime.
    - Return None when the tag is missing. Don't crash: real SD cards contain
      corrupt files.
    """
    raise NotImplementedError


def build_media(images_dir: Path) -> pd.DataFrame:
    """One row per image, with the Camtrap DP media columns.

    TODO (weeks 5–6): at minimum mediaID, deploymentID (the subfolder name),
    timestamp, filePath, fileName and fileMediatype.
    """
    raise NotImplementedError


def build_deployments(media: pd.DataFrame) -> pd.DataFrame:
    """One row per camera, with deploymentStart and deploymentEnd.

    TODO (week 6): group media by deploymentID. Think about whether
    min and max timestamps are trustworthy if a camera's clock was wrong.
    """
    raise NotImplementedError


def run(images_dir: Path, out_dir: Path) -> None:
    """Build both tables and write them as CSV files in out_dir."""
    raise NotImplementedError("ingest isn't implemented yet. That's your week 5 task!")
