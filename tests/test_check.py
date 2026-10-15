import pandas as pd

from camtraptools.check import (
    check_clock_jumps,
    check_day_night,
    check_deployment,
    check_gaps,
    check_implausible_dates,
    check_missing_timestamps,
    check_trigger_rate,
    find_gaps,
    run,
    summarize,
    summarize_deployment,
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


START = pd.Timestamp("2024-03-01 00:00")
END = pd.Timestamp("2024-04-10 00:00")


def test_find_gaps_includes_start_middle_and_end():
    media = make_media("2024-03-25", "2024-03-26", "2024-03-27")

    gaps = find_gaps(media, START, END, max_gap_days=10)

    assert gaps == [
        (START, pd.Timestamp("2024-03-25"), "start"),
        (pd.Timestamp("2024-03-27"), END, "end"),
    ]


def test_find_gaps_without_recorded_period_finds_middle_gap():
    media = make_media("2024-03-01", "2024-03-11")

    gaps = find_gaps(
        media, pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-11"), max_gap_days=7
    )

    assert gaps == [(pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-11"), "middle")]


def test_check_gaps_explains_each_kind_of_gap():
    media = make_media("2024-03-25", "2024-03-26", "2024-04-09")

    issues = check_gaps(media, START, pd.Timestamp("2024-05-20"), max_gap_days=10)

    assert issues == [
        "no images in the first 24.0 days (2024-03-01 to 2024-03-25); was the camera armed, and is the start date right?",
        "no images for 14.0 days (2024-03-26 to 2024-04-09); check for a camera fault",
        "no images in the last 41.0 days (2024-04-09 to 2024-05-20); the camera may have stopped (battery or full card)",
    ]


def test_check_gaps_accepts_short_silences():
    media = make_media("2024-03-10", "2024-03-25")

    assert check_gaps(media, START, END) == []


def test_check_gaps_skips_implausible_period():
    media = make_media("2000-01-01", "2024-03-25")

    assert check_gaps(media, pd.Timestamp("2000-01-01"), END) == []


def test_check_trigger_rate_flags_busy_days():
    busy_day = [
        pd.Timestamp("2024-03-04 09:00") + pd.Timedelta(minutes=i) for i in range(120)
    ]
    media = make_media("2024-03-01 06:00", *busy_day)

    assert check_trigger_rate(media) == [
        (
            "1 day(s) with more than 100 images (most: 120 on 2024-03-04); "
            "often vegetation moving in front of the lens or sun glare"
        )
    ]


def test_check_trigger_rate_accepts_normal_days():
    assert check_trigger_rate(make_media("2024-03-01 06:00", "2024-03-01 07:00")) == []


def test_check_day_night_flags_missing_night_images():
    daytime = [
        pd.Timestamp("2024-03-01 09:00") + pd.Timedelta(days=i) for i in range(50)
    ]

    assert check_day_night(make_media(*daytime)) == [
        "none of 50 images were taken at night; check the infrared flash"
    ]


def test_check_day_night_flags_missing_day_images():
    nighttime = [
        pd.Timestamp("2024-03-01 23:00") + pd.Timedelta(days=i) for i in range(50)
    ]

    assert check_day_night(make_media(*nighttime)) == [
        "none of 50 images were taken during the day; check the camera settings"
    ]


def test_check_day_night_needs_enough_images():
    daytime = [
        pd.Timestamp("2024-03-01 09:00") + pd.Timedelta(days=i) for i in range(49)
    ]

    assert check_day_night(make_media(*daytime)) == []


def test_summarize_deployment_reports_key_numbers():
    media = make_media("2024-03-02 23:00", "2024-03-05 12:00")

    summary = summarize_deployment(media, START, pd.Timestamp("2024-03-11"))

    assert summary == {
        "days": 10.0,
        "images": 2,
        "imagesPerDay": 0.2,
        "nightShare": 0.5,
        "longestGapDays": 5.5,
    }


def test_run_uses_recorded_period_for_gaps(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:02 06:00:00")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "deploymentID,latitude,longitude,deploymentStart,deploymentEnd\n"
        "camera_A,52.7,23.85,2024-03-01,2024-04-10\n"
    )
    ingest(tmp_path / "images", tmp_path / "package", sheet)

    results = run(tmp_path / "package")

    assert results["camera_A"] == [
        (
            "no images in the last 39.7 days (2024-03-02 to 2024-04-10); "
            "the camera may have stopped (battery or full card)"
        )
    ]
    assert summarize(tmp_path / "package").loc["camera_A", "days"] == 41.0
