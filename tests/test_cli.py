import json
import sys

import pytest
from typer.testing import CliRunner

from camtraptools import __version__
from camtraptools.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_commands_listed():
    result = runner.invoke(app, ["--help"])
    assert "ingest" in result.output
    assert "check" in result.output


def test_ingest_command_reports_counts(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:05:17 06:42:10")

    result = runner.invoke(
        app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")]
    )

    assert result.exit_code == 0
    assert "Found 1 images from 1 deployments." in result.output
    assert (tmp_path / "package/deployments.csv").exists()


def test_ingest_command_rejects_missing_folder(tmp_path):
    result = runner.invoke(app, ["ingest", str(tmp_path / "missing")])

    assert result.exit_code != 0


def test_ingest_command_reports_field_sheet_issues(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("deploymentID,latitude,longitude\ncamera_A,152.7,23.85\n")

    result = runner.invoke(
        app,
        [
            "ingest",
            str(tmp_path / "images"),
            "--out",
            str(tmp_path / "out"),
            "--field-sheet",
            str(sheet),
        ],
    )

    assert result.exit_code == 0
    assert "camera_A: impossible coordinates" in result.output


def test_ingest_command_rejects_incomplete_field_sheet(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("deploymentID,notes\ncamera_A,ok\n")

    result = runner.invoke(
        app, ["ingest", str(tmp_path / "images"), "--field-sheet", str(sheet)]
    )

    assert result.exit_code == 1
    assert "missing required columns" in result.output


def test_check_command_reports_issues_per_deployment(make_image, tmp_path):
    make_image("images/healthy/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    make_image("images/reset/IMG_0001.JPG", date_time_original="2000:01:01 00:05:00")
    runner.invoke(
        app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")]
    )

    result = runner.invoke(app, ["check", str(tmp_path / "package")])

    assert result.exit_code == 0
    assert "healthy: 0.0 days, 1 images, longest gap 0.0 days\n  OK" in result.output
    assert "! 1 image(s) dated before 2010" in result.output


def test_check_command_needs_an_ingested_package(tmp_path):
    result = runner.invoke(app, ["check", str(tmp_path)])

    assert result.exit_code == 1
    assert "run `camtraptools ingest` first" in result.output


def test_check_command_accepts_max_gap_days(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    make_image("images/camera_A/IMG_0002.JPG", date_time_original="2024:03:11 06:00:00")
    runner.invoke(
        app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")]
    )

    default = runner.invoke(app, ["check", str(tmp_path / "package")])
    strict = runner.invoke(
        app, ["check", str(tmp_path / "package"), "--max-gap-days", "7"]
    )

    assert "  OK" in default.output
    assert "no images for 10.0 days" in strict.output


def test_report_command_writes_html(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(
        app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")]
    )

    result = runner.invoke(
        app,
        ["report", str(tmp_path / "package"), "--out", str(tmp_path / "report.html")],
    )

    assert result.exit_code == 0
    assert (tmp_path / "report.html").exists()


def test_report_command_needs_an_ingested_package(tmp_path):
    result = runner.invoke(app, ["report", str(tmp_path)])

    assert result.exit_code == 1
    assert "run `camtraptools ingest` first" in result.output


def write_speciesnet_results(package, conf, country=None):
    gemsbok = "uuid;mammalia;artiodactyla;bovidae;oryx;gazella;gemsbok"
    detection = {
        "category": "1",
        "label": "animal",
        "conf": conf,
        "bbox": [0, 0, 1, 1],
        "processed_classification": {"class": gemsbok, "score": 0.99, "source": "classifier"},
    }
    results = {
        "info": {"model": "kaggle:google/speciesnet/pyTorch/v4.0.3a/1", "country": country},
        "predictions": [{"filepath": "camera_A/IMG_0001.JPG", "detections": [detection], "model_version": "4.0.3a"}],
    }
    results_file = package / "speciesnet.json"
    results_file.write_text(json.dumps(results), encoding="utf-8")
    return results_file


def test_detect_command_summarizes_results(make_image, monkeypatch, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    monkeypatch.setattr(
        "camtraptools.detect.run",
        lambda package, images, model, country, admin1_region: write_speciesnet_results(package, 0.9, country),
    )

    result = runner.invoke(
        app, ["detect", str(tmp_path / "package"), "--images", str(tmp_path / "images"), "--country", "ZAF"]
    )

    assert result.exit_code == 0
    assert "gemsbok" in result.output
    assert "observations.csv (1 observations)" in result.output


def test_detect_command_reuses_existing_results(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.3, "ZAF")

    result = runner.invoke(app, ["detect", str(tmp_path / "package"), "--country", "ZAF", "--threshold", "0.5"])

    assert result.exit_code == 0
    assert "Using existing results" in result.output
    assert "observations.csv (1 observations)" in result.output


def test_detect_command_needs_images_to_run(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])

    result = runner.invoke(app, ["detect", str(tmp_path / "package")])

    assert result.exit_code == 1
    assert "--images is required" in result.output


def test_detect_command_explains_missing_extra(make_image, monkeypatch, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    monkeypatch.setitem(sys.modules, "speciesnet", None)

    result = runner.invoke(app, ["detect", str(tmp_path / "package"), "--images", str(tmp_path / "images")])

    assert result.exit_code == 1
    assert "uv sync --extra detect" in result.output


def test_events_command_groups_images(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    make_image("images/camera_A/IMG_0002.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])

    result = runner.invoke(app, ["events", str(tmp_path / "package")])

    assert result.exit_code == 0
    assert "Grouped 2 images into 1 events (30-minute gap)." in result.output
    assert "Oryx gazella" in result.output


def test_events_command_needs_observations(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])

    result = runner.invoke(app, ["events", str(tmp_path / "package")])

    assert result.exit_code == 1
    assert "run `camtraptools detect` first" in result.output


def test_import_command_reports_coverage(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    make_image("images/camera_A/IMG_0002.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    results = {
        "info": {"detector_metadata": {"megadetector_version": "v5a.0.0"}},
        "detection_categories": {"1": "animal"},
        "images": [
            {"file": "survey/camera_A/IMG_0001.JPG", "detections": [{"category": "1", "conf": 0.9, "bbox": [0, 0, 1, 1]}]},
            {"file": "survey/camera_X/IMG_0001.JPG", "detections": []},
        ],
    }
    (tmp_path / "md.json").write_text(json.dumps(results), encoding="utf-8")

    result = runner.invoke(app, ["import", str(tmp_path / "package"), str(tmp_path / "md.json")])

    assert result.exit_code == 0
    assert "Imported results for 1 of 2 images." in result.output
    assert "Ignored 1 images in md.json" in result.output
    assert "1 images in the package have no results and are marked unclassified." in result.output
    assert "observations.csv (2 observations)" in result.output


def test_import_command_rejects_unrelated_results(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    (tmp_path / "md.json").write_text(json.dumps({"images": [{"file": "other/IMG_9.JPG", "detections": []}]}), encoding="utf-8")

    result = runner.invoke(app, ["import", str(tmp_path / "package"), str(tmp_path / "md.json")])

    assert result.exit_code == 1
    assert "none of the images in md.json are in the package" in result.output


def test_export_command_writes_package(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("deploymentID,latitude,longitude\ncamera_A,52.7,23.85\n")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package"), "--field-sheet", str(sheet)])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])

    result = runner.invoke(
        app,
        [
            "export", str(tmp_path / "package"), "--out", str(tmp_path / "dp"), "--title", "Test",
            "--contributor", "Ivan Posavec", "--timezone", "Africa/Johannesburg",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Wrote a Camtrap DP 1.0.2 data package" in result.output
    assert (tmp_path / "dp/datapackage.json").exists()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--timezone", "Mars/Olympus"], "unknown time zone 'Mars/Olympus'"),
        (["--timezone", "UTC", "--sampling-design", "random"], "--sampling-design must be one of"),
    ],
)
def test_export_command_rejects_bad_options(make_image, tmp_path, options, message):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])

    result = runner.invoke(
        app, ["export", str(tmp_path / "package"), "--out", str(tmp_path / "dp"), "--title", "T", "--contributor", "I", *options]
    )

    assert result.exit_code == 1
    assert message in result.output


def test_export_command_needs_coordinates(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])

    result = runner.invoke(
        app,
        ["export", str(tmp_path / "package"), "--out", str(tmp_path / "dp"), "--title", "T", "--contributor", "I", "--timezone", "UTC"],
    )

    assert result.exit_code == 1
    assert "deployments without coordinates: camera_A" in result.output


def test_review_command_starts_server(make_image, monkeypatch, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])
    runner.invoke(app, ["events", str(tmp_path / "package")])
    started = {}
    monkeypatch.setattr("uvicorn.run", lambda app, host, port, log_level: started.update(host=host, port=port))

    result = runner.invoke(
        app, ["review", str(tmp_path / "package"), "--images", str(tmp_path / "images"), "--no-open-browser"]
    )

    assert result.exit_code == 0, result.output
    assert started == {"host": "127.0.0.1", "port": 8765}
    assert "http://127.0.0.1:8765/" in result.output


def test_review_command_needs_events(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])

    result = runner.invoke(app, ["review", str(tmp_path / "package"), "--images", str(tmp_path / "images")])

    assert result.exit_code == 1
    assert "run `camtraptools events` first" in result.output
