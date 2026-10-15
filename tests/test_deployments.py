import pandas as pd

from camtraptools.ingest import build_deployments, build_media, run


def test_build_deployments_spans_first_and_last_image(make_image, tmp_path):
    make_image("camera_A/IMG_0002.JPG", date_time_original="2024:05:20 18:30:00")
    make_image("camera_A/IMG_0001.JPG", date_time_original="2024:05:17 06:42:10")
    make_image("camera_B/IMG_0001.JPG", date_time_original="2024:06:01 12:00:00")

    deployments = build_deployments(build_media(tmp_path))

    assert list(deployments.columns) == [
        "deploymentID",
        "deploymentStart",
        "deploymentEnd",
    ]
    camera_a = deployments.set_index("deploymentID").loc["camera_A"]
    assert camera_a["deploymentStart"] == pd.Timestamp("2024-05-17 06:42:10")
    assert camera_a["deploymentEnd"] == pd.Timestamp("2024-05-20 18:30:00")


def test_build_deployments_ignores_images_without_timestamp(make_image, tmp_path):
    make_image("camera_A/IMG_0001.JPG", date_time_original="2024:05:17 06:42:10")
    make_image("camera_A/IMG_0002.JPG")

    deployments = build_deployments(build_media(tmp_path))

    assert deployments.loc[0, "deploymentStart"] == pd.Timestamp("2024-05-17 06:42:10")
    assert deployments.loc[0, "deploymentEnd"] == pd.Timestamp("2024-05-17 06:42:10")


def test_run_writes_csv_files(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:05:17 06:42:10")
    out_dir = tmp_path / "package"

    run(tmp_path / "images", out_dir)

    media = pd.read_csv(out_dir / "media.csv")
    deployments = pd.read_csv(out_dir / "deployments.csv")
    assert media.loc[0, "timestamp"] == "2024-05-17T06:42:10"
    assert deployments.loc[0, "deploymentStart"] == "2024-05-17T06:42:10"
