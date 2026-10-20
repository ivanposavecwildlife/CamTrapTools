import pandas as pd
import pytest

from camtraptools.ingest import (
    add_field_sheet,
    build_deployments,
    build_media,
    check_field_sheet,
    clock_offsets,
    format_offset,
    match_deployments,
    normalize_id,
    read_field_sheet,
    run,
)


def write_sheet(path, text):
    path.write_text(text)
    return path


def test_read_field_sheet_cleans_names_and_coordinates(tmp_path):
    path = write_sheet(
        tmp_path / "sheet.csv",
        " deploymentID ,latitude,longitude\n camera_A ,52.70,23.85\ncamera_B,52.8 N,23.9\n",
    )

    sheet = read_field_sheet(path)

    assert list(sheet["deploymentID"]) == ["camera_A", "camera_B"]
    assert sheet.loc[0, "latitude"] == 52.70
    assert pd.isna(sheet.loc[1, "latitude"])


def test_read_field_sheet_reads_excel(tmp_path):
    path = tmp_path / "sheet.xlsx"
    pd.DataFrame(
        {"deploymentID": ["camera_A"], "latitude": [52.7], "longitude": [23.85]}
    ).to_excel(path, index=False)

    sheet = read_field_sheet(path)

    assert sheet.loc[0, "deploymentID"] == "camera_A"


def test_read_field_sheet_requires_coordinates(tmp_path):
    path = write_sheet(tmp_path / "sheet.csv", "deploymentID,notes\ncamera_A,ok\n")

    with pytest.raises(ValueError, match="latitude, longitude"):
        read_field_sheet(path)


def test_check_field_sheet_reports_problems(tmp_path):
    path = write_sheet(
        tmp_path / "sheet.csv",
        "deploymentID,latitude,longitude\n"
        "camera_A,52.70,23.85\n"
        "camera_A,52.71,23.86\n"
        "camera_B,152.70,23.85\n"
        "camera_C,,23.85\n"
        "camera_D,52.70,23.85\n",
    )

    issues = check_field_sheet(read_field_sheet(path))

    assert issues == [
        "camera_A: listed more than once in the field sheet",
        "camera_B: impossible coordinates (152.7, 23.85)",
        "camera_C: missing or unreadable coordinates",
    ]


def test_match_deployments_suggests_close_names(make_image, tmp_path):
    make_image("images/camera_B/IMG_0001.JPG")
    sheet = pd.DataFrame(
        {
            "deploymentID": ["camera-B", "camera_C"],
            "latitude": [1, 1],
            "longitude": [1, 1],
        }
    )

    issues = match_deployments(sheet, build_media(tmp_path / "images"))

    assert issues == [
        "camera_B: has images but is not in the field sheet (did you mean 'camera-B'?)",
        "camera-B: in the field sheet but has no images",
        "camera_C: in the field sheet but has no images",
    ]


def test_normalize_id_ignores_case_separators_and_leading_zeros():
    assert (
        normalize_id("CAM_03")
        == normalize_id("cam-3")
        == normalize_id("Cam3")
        == "cam3"
    )
    assert normalize_id("site10") != normalize_id("site1")


def test_add_field_sheet_merges_by_deployment(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:05:17 06:42:10")
    make_image("images/camera_B/IMG_0001.JPG", date_time_original="2024:05:18 06:42:10")
    sheet = pd.DataFrame(
        {"deploymentID": ["camera_A"], "latitude": [52.7], "longitude": [23.85]}
    )

    deployments = add_field_sheet(
        build_deployments(build_media(tmp_path / "images")), sheet
    )

    assert list(deployments["deploymentID"]) == ["camera_A", "camera_B"]
    assert deployments.loc[0, "latitude"] == 52.7
    assert pd.isna(deployments.loc[1, "latitude"])


def test_run_with_field_sheet_writes_coordinates(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:05:17 06:42:10")
    sheet = write_sheet(
        tmp_path / "sheet.csv", "deploymentID,latitude,longitude\ncamera_A,52.7,23.85\n"
    )

    _, _, issues, _ = run(tmp_path / "images", tmp_path / "package", sheet)

    deployments = pd.read_csv(tmp_path / "package/deployments.csv")
    assert issues == []
    assert deployments.loc[0, "latitude"] == 52.7


def test_clock_offsets_from_camera_and_actual_time():
    sheet = pd.DataFrame(
        {
            "deploymentID": ["cam_reset", "cam_ok", "cam_half", "cam_bad", "cam_none"],
            "cameraTime": ["2010-01-01T00:05:00", "2024-03-01T08:00:00", "2010-01-01T00:00:00", "soon", None],
            "actualTime": ["2024-03-01T08:05:00", "2024-03-01T08:00:00", None, "2024-03-01T08:00:00", None],
        }
    )

    offsets, issues = clock_offsets(sheet)

    assert offsets == {"cam_reset": pd.Timestamp("2024-03-01T08:05") - pd.Timestamp("2010-01-01T00:05")}
    assert issues == [
        "cam_half: clock correction needs both cameraTime and actualTime",
        "cam_bad: unreadable cameraTime or actualTime",
    ]


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (pd.Timedelta(hours=1), "+1:00:00"),
        (pd.Timedelta(hours=-1), "-1:00:00"),
        (pd.Timedelta(days=-3, hours=-2, minutes=-15), "-3 days 2:15:00"),
    ],
)
def test_format_offset(offset, expected):
    assert format_offset(offset) == expected


def test_run_corrects_clocks_before_deriving_periods(make_image, tmp_path):
    make_image("images/cam_reset/IMG_0001.JPG", date_time_original="2010:01:01 00:05:00")
    make_image("images/cam_reset/IMG_0002.JPG", date_time_original="2010:01:02 00:05:00")
    make_image("images/cam_ok/IMG_0001.JPG", date_time_original="2024:03:01 09:00:00")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "deploymentID,latitude,longitude,cameraTime,actualTime\n"
        "cam_reset,52.7,23.8,2010-01-01T00:00:00,2024-03-01T08:00:00\n"
        "cam_ok,52.7,23.8,,\n"
    )

    media, deployments, issues, corrections = run(tmp_path / "images", tmp_path / "package", sheet)

    assert issues == []
    assert list(corrections) == ["cam_reset"]
    reset = media[media["deploymentID"] == "cam_reset"]
    assert reset["timestamp"].tolist() == [pd.Timestamp("2024-03-01T08:05"), pd.Timestamp("2024-03-02T08:05")]
    assert reset["mediaComments"].iloc[0] == "Camera clock corrected by +5173 days 8:00:00; camera time 2010-01-01T00:05:00"
    assert deployments.set_index("deploymentID").loc["cam_reset", "deploymentStart"] == pd.Timestamp("2024-03-01T08:05")
    assert "cameraTime" not in deployments.columns
    saved = pd.read_csv(tmp_path / "package" / "media.csv")
    assert saved.loc[saved["deploymentID"] == "cam_ok", "mediaComments"].isna().all()
