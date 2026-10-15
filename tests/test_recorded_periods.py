import pandas as pd

from camtraptools.ingest import (
    add_field_sheet,
    build_deployments,
    build_media,
    check_recorded_periods,
    parse_recorded_periods,
    run,
)


def make_sheet(**columns):
    return pd.DataFrame(
        {
            "deploymentID": ["camera_A"],
            "latitude": [52.7],
            "longitude": [23.85],
            **columns,
        }
    )


def test_parse_recorded_periods_treats_date_only_end_as_whole_day():
    sheet = parse_recorded_periods(
        make_sheet(deploymentStart=["2024-05-01"], deploymentEnd=["2024-05-31"])
    )

    assert sheet.loc[0, "deploymentStart"] == pd.Timestamp("2024-05-01 00:00:00")
    assert sheet.loc[0, "deploymentEnd"] == pd.Timestamp("2024-05-31 23:59:59")


def test_parse_recorded_periods_keeps_end_time_when_given():
    sheet = parse_recorded_periods(make_sheet(deploymentEnd=["2024-05-31T14:30:00"]))

    assert sheet.loc[0, "deploymentEnd"] == pd.Timestamp("2024-05-31 14:30:00")


def test_parse_recorded_periods_accepts_mixed_iso_formats():
    sheet = pd.DataFrame(
        {
            "deploymentID": ["camera_A", "camera_B"],
            "latitude": [52.7, 52.7],
            "longitude": [23.85, 23.85],
            "deploymentEnd": ["2024-05-31", "2024-06-30T14:30:00"],
        }
    )

    sheet = parse_recorded_periods(sheet)

    assert sheet.loc[0, "deploymentEnd"] == pd.Timestamp("2024-05-31 23:59:59")
    assert sheet.loc[1, "deploymentEnd"] == pd.Timestamp("2024-06-30 14:30:00")


def test_add_field_sheet_prefers_recorded_period(make_image, tmp_path):
    make_image("camera_A/IMG_0001.JPG", date_time_original="2024:05:03 06:00:00")
    make_image("camera_A/IMG_0002.JPG", date_time_original="2024:05:20 18:00:00")
    sheet = parse_recorded_periods(
        make_sheet(
            deploymentStart=["2024-05-01 09:15"], deploymentEnd=["2024-05-31 11:00"]
        )
    )

    deployments = add_field_sheet(build_deployments(build_media(tmp_path)), sheet)

    assert list(deployments.columns[:3]) == [
        "deploymentID",
        "deploymentStart",
        "deploymentEnd",
    ]
    assert not any(column.endswith("_recorded") for column in deployments.columns)
    assert deployments.loc[0, "deploymentStart"] == pd.Timestamp("2024-05-01 09:15")
    assert deployments.loc[0, "deploymentEnd"] == pd.Timestamp("2024-05-31 11:00")


def test_add_field_sheet_falls_back_to_image_dates(make_image, tmp_path):
    make_image("camera_A/IMG_0001.JPG", date_time_original="2024:05:03 06:00:00")
    sheet = parse_recorded_periods(
        make_sheet(deploymentStart=[None], deploymentEnd=[None])
    )

    deployments = add_field_sheet(build_deployments(build_media(tmp_path)), sheet)

    assert deployments.loc[0, "deploymentStart"] == pd.Timestamp("2024-05-03 06:00:00")


def test_check_recorded_periods_reports_images_outside_period(make_image, tmp_path):
    make_image("camera_A/IMG_0001.JPG", date_time_original="2024:04:28 06:00:00")
    make_image("camera_A/IMG_0002.JPG", date_time_original="2024:05:10 06:00:00")
    make_image("camera_A/IMG_0003.JPG", date_time_original="2024:06:02 06:00:00")
    make_image("camera_A/IMG_0004.JPG", date_time_original="2024:06:03 06:00:00")
    sheet = parse_recorded_periods(
        make_sheet(deploymentStart=["2024-05-01"], deploymentEnd=["2024-05-31"])
    )

    issues = check_recorded_periods(sheet, build_media(tmp_path))

    assert issues == [
        "camera_A: 1 image(s) taken before the recorded start",
        "camera_A: 2 image(s) taken after the recorded end",
    ]


def test_check_recorded_periods_reports_start_after_end(make_image, tmp_path):
    make_image("camera_A/IMG_0001.JPG")
    sheet = parse_recorded_periods(
        make_sheet(deploymentStart=["2024-06-01"], deploymentEnd=["2024-05-01"])
    )

    issues = check_recorded_periods(sheet, build_media(tmp_path))

    assert issues == ["camera_A: recorded start is after the recorded end"]


def test_check_recorded_periods_reports_unreadable_dates(make_image, tmp_path):
    make_image("camera_A/IMG_0001.JPG")
    sheet = parse_recorded_periods(
        make_sheet(deploymentStart=["27/11/2018"], deploymentEnd=[None])
    )

    issues = check_recorded_periods(sheet, build_media(tmp_path))

    assert issues == [
        "camera_A: no readable recorded start; using the first image instead",
        "camera_A: no readable recorded end; using the last image instead",
    ]


def test_check_recorded_periods_without_dates_reports_nothing(make_image, tmp_path):
    make_image("camera_A/IMG_0001.JPG")

    assert check_recorded_periods(make_sheet(), build_media(tmp_path)) == []


def test_run_writes_recorded_period(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:05:03 06:00:00")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "deploymentID,latitude,longitude,deploymentStart,deploymentEnd\n"
        "camera_A,52.7,23.85,2024-05-01T09:15:00,2024-05-31\n"
    )

    run(tmp_path / "images", tmp_path / "package", sheet)

    deployments = pd.read_csv(tmp_path / "package/deployments.csv")
    assert list(deployments.columns[:3]) == [
        "deploymentID",
        "deploymentStart",
        "deploymentEnd",
    ]
    assert deployments.loc[0, "deploymentStart"] == "2024-05-01T09:15:00"
    assert deployments.loc[0, "deploymentEnd"] == "2024-05-31T23:59:59"
