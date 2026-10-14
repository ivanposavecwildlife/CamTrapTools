from datetime import datetime

from camtraptools.ingest import read_timestamp


def test_reads_date_time_original(make_image):
    path = make_image("IMG_0001.JPG", date_time_original="2024:05:17 06:42:10")
    assert read_timestamp(path) == datetime(2024, 5, 17, 6, 42, 10)


def test_falls_back_to_date_time(make_image):
    path = make_image("IMG_0001.JPG", date_time="2023:11:02 22:15:00")
    assert read_timestamp(path) == datetime(2023, 11, 2, 22, 15, 0)


def test_prefers_date_time_original(make_image):
    path = make_image(
        "IMG_0001.JPG",
        date_time_original="2024:05:17 06:42:10",
        date_time="2024:06:01 12:00:00",
    )
    assert read_timestamp(path) == datetime(2024, 5, 17, 6, 42, 10)


def test_missing_exif_returns_none(make_image):
    path = make_image("IMG_0001.JPG")
    assert read_timestamp(path) is None


def test_unparseable_timestamp_returns_none(make_image):
    path = make_image("IMG_0001.JPG", date_time_original="0000:00:00 00:00:00")
    assert read_timestamp(path) is None


def test_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "IMG_0001.JPG"
    path.write_bytes(b"not an image")
    assert read_timestamp(path) is None
