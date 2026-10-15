import pandas as pd

from camtraptools.check import (
    check_clock_jumps,
    check_deployment,
    check_implausible_dates,
    check_missing_timestamps,
    run,
)
from camtraptools.ingest import run as ingest


def make_media(*timestamps):
    return pd.DataFrame(
        {
            "deploymentID": "camera_A",
            "timestamp": pd.to_datetime(list(timestamps)),
            "filePath": [
                f"camera_A/IMG_{number:04}.JPG"
                for number in range(1, len(timestamps) + 1)
            ],
            "fileName": [
                f"IMG_{number:04}.JPG" for number in range(1, len(timestamps) + 1)
            ],
        }
    )


def test_missing_timestamps_are_counted():
    media = make_media("2024-03-01 06:00", None, None)

    assert check_missing_timestamps(media) == [
        "2 image(s) without a readable timestamp"
    ]


def test_complete_timestamps_pass():
    assert check_missing_timestamps(make_media("2024-03-01 06:00")) == []


def test_implausible_dates_flag_reset_clock_and_future():
    media = make_media("2000-01-01 00:05", "2024-03-01 06:00", "2034-03-01 06:00")

    issues = check_implausible_dates(media, now=pd.Timestamp("2026-01-01"))

    assert issues == [
        "1 image(s) dated before 2010; the camera clock was probably reset",
        "1 image(s) dated in the future; the camera clock was set wrongly",
    ]


def test_plausible_dates_pass():
    assert (
        check_implausible_dates(
            make_media("2024-03-01 06:00"), now=pd.Timestamp("2026-01-01")
        )
        == []
    )


def test_clock_jumps_report_time_going_backwards():
    media = make_media(
        "2024-03-01 06:00", "2024-03-01 18:00", "2000-01-01 00:05", "2000-01-01 09:05"
    )

    assert check_clock_jumps(media) == [
        "clock went back 8826 days 17:55:00 at IMG_0003.JPG"
    ]


def test_clock_jumps_follow_file_order_not_row_order():
    media = make_media("2024-03-01 06:00", "2024-03-01 18:00").iloc[::-1]

    assert check_clock_jumps(media) == []


def test_burst_with_identical_times_is_not_a_jump():
    assert check_clock_jumps(make_media("2024-03-01 06:00", "2024-03-01 06:00")) == []


def test_check_deployment_combines_checks():
    media = make_media("2024-03-01 06:00", "2000-01-01 00:05", None)

    issues = check_deployment(media)

    assert len(issues) == 3


def test_run_checks_each_deployment_in_package(make_image, tmp_path):
    make_image("images/healthy/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    make_image("images/reset/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    make_image("images/reset/IMG_0002.JPG", date_time_original="2000:01:01 00:05:00")
    ingest(tmp_path / "images", tmp_path / "package")

    results = run(tmp_path / "package")

    assert results["healthy"] == []
    assert len(results["reset"]) == 2
