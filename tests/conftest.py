"""Shared pytest fixtures."""

from pathlib import Path

import pytest
from PIL import ExifTags, Image


@pytest.fixture
def make_image(tmp_path):
    """Factory that writes a small JPEG with optional EXIF timestamps.

    Usage: ``make_image("cam_A/IMG_0001.JPG", date_time_original="2024:05:17 06:42:10")``
    """

    def _make(
        relative_path: str,
        date_time_original: str | None = None,
        date_time: str | None = None,
    ) -> Path:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        exif = Image.Exif()
        if date_time is not None:
            exif[ExifTags.Base.DateTime] = date_time
        if date_time_original is not None:
            exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.DateTimeOriginal] = date_time_original
        Image.new("RGB", (64, 48), "green").save(path, exif=exif)
        return path

    return _make
