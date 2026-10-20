import json
import sys

import pandas as pd
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
    assert "Found 1 image from 1 deployment." in result.output
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
        lambda package, images, model, country, admin1_region, **options: write_speciesnet_results(package, 0.9, country),
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
    monkeypatch.setattr("camtraptools.review.free_port", lambda port: port)

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


def test_review_command_reports_a_busy_port(make_image, monkeypatch, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])
    runner.invoke(app, ["events", str(tmp_path / "package")])
    monkeypatch.setattr("uvicorn.run", lambda app, host, port, log_level: None)
    monkeypatch.setattr("camtraptools.review.free_port", lambda port: 50123)

    result = runner.invoke(
        app, ["review", str(tmp_path / "package"), "--images", str(tmp_path / "images"), "--no-open-browser"]
    )

    assert "Port 8765 is in use by another program; using 50123 instead." in result.output
    assert "http://127.0.0.1:50123/" in result.output


def test_review_command_rejects_unknown_order(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])

    result = runner.invoke(app, ["review", str(tmp_path / "package"), "--images", str(tmp_path / "images"), "--order", "random"])

    assert result.exit_code == 1
    assert "--order must be one of: uncertainty, time" in result.output


def test_review_stats_command_without_reviews(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])

    result = runner.invoke(app, ["review-stats", str(tmp_path / "package")])

    assert result.exit_code == 0
    assert "No reviews" in result.output


def test_review_stats_command_reports_corrections(make_image, tmp_path):
    from camtraptools import reviews

    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])
    runner.invoke(app, ["events", str(tmp_path / "package")])
    reviews.save_review(tmp_path / "package", "camera_A-0001", "animal", "Canis mesomelas", 1, "Ivan Posavec")

    result = runner.invoke(app, ["review-stats", str(tmp_path / "package")])

    assert result.exit_code == 0
    assert "Reviewed 1 of 1 events; agreed with the AI on 0 (0%)." in result.output
    assert "camera_A-0001: Oryx gazella → Canis mesomelas" in result.output


def test_evaluate_command_reports_metrics(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])
    runner.invoke(app, ["events", str(tmp_path / "package")])
    truth = tmp_path / "truth.csv"
    truth.write_text("filePath,scientificName\ncamera_A/IMG_0001.JPG,Canis mesomelas\n")

    result = runner.invoke(app, ["evaluate", str(tmp_path / "package"), "--truth", str(truth)])

    assert result.exit_code == 0, result.output
    assert "Compared 1 events with labels in truth.csv." in result.output
    assert "wrong: 1 (100%)" in result.output
    assert "camera_A-0001: Canis mesomelas → Oryx gazella (wrong)" in result.output


def test_evaluate_command_without_truth(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])
    runner.invoke(app, ["events", str(tmp_path / "package")])

    result = runner.invoke(app, ["evaluate", str(tmp_path / "package")])

    assert result.exit_code == 1
    assert "No events to evaluate: reviews has no labels" in result.output


def test_evaluate_command_writes_report(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])
    runner.invoke(app, ["events", str(tmp_path / "package")])
    truth = tmp_path / "truth.csv"
    truth.write_text("filePath,scientificName\ncamera_A/IMG_0001.JPG,Canis mesomelas\n")
    report = tmp_path / "report.html"

    result = runner.invoke(
        app,
        ["evaluate", str(tmp_path / "package"), "--truth", str(truth), "--report", str(report), "--images", str(tmp_path / "images")],
    )

    assert result.exit_code == 0, result.output
    assert "Confidence thresholds for 95% precision" in result.output
    assert "camera_A-0001 · true: Canis mesomelas · AI: Oryx gazella · wrong" in report.read_text(encoding="utf-8")


def test_evaluate_report_needs_images(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])

    result = runner.invoke(app, ["evaluate", str(tmp_path / "package"), "--report", str(tmp_path / "r.html")])

    assert result.exit_code == 1
    assert "--report needs --images" in result.output


def test_crops_command_builds_dataset(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])
    runner.invoke(app, ["events", str(tmp_path / "package")])
    truth = tmp_path / "truth.csv"
    truth.write_text("filePath,scientificName\ncamera_A/IMG_0001.JPG,Oryx gazella\n")

    result = runner.invoke(
        app,
        [
            "crops", str(tmp_path / "package"), "--images", str(tmp_path / "images"),
            "--out", str(tmp_path / "crops"), "--truth", str(truth),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Saved 1 crop labelled from labels in truth.csv." in result.output
    assert "train: camera_A (1 crop)" in result.output
    assert "Oryx gazella: no crops in val or test" in result.output
    assert (tmp_path / "crops" / "train" / "Oryx_gazella").is_dir()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--val", "0.6", "--test", "0.5"], "leave room for training"),
        ([], "is not empty"),
    ],
)
def test_crops_command_rejects_bad_options(make_image, tmp_path, extra, message):
    make_image("images/camera_A/IMG_0001.JPG")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])
    out = tmp_path / "crops"
    if not extra:
        out.mkdir()
        (out / "old.jpg").write_bytes(b"")

    result = runner.invoke(
        app, ["crops", str(tmp_path / "package"), "--images", str(tmp_path / "images"), "--out", str(out), *extra]
    )

    assert result.exit_code == 1
    assert message in result.output


def test_adapt_command_reports_accuracy(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from test_adapt import make_crops, tiny_speciesnet

    crops_dir = make_crops(tmp_path / "crops")
    pd.read_csv(crops_dir / "crops.csv").assign(
        label=lambda crops: crops["label"].where(crops["filePath"] != "test/Vulpes_chama/east0.jpg", "Canis mesomelas")
    ).to_csv(crops_dir / "crops.csv", index=False)
    monkeypatch.setattr("camtraptools.adapt.load_speciesnet", lambda model: tiny_speciesnet())

    result = runner.invoke(app, ["adapt", str(crops_dir), "--out", str(tmp_path / "model"), "--device", "cpu"])

    assert result.exit_code == 0, result.output
    assert "Trained a new final layer for 2 labels on cpu, keeping epoch" in result.output
    assert "Vulpes chama: 3 training crops, new to SpeciesNet" in result.output
    assert "⚠ Labels with no training crops, which the adapted layer cannot name (1 held-out crop):" in result.output
    assert "Canis mesomelas (1)" in result.output
    assert "Crops named correctly per label, test cameras:" in result.output
    assert "average over labels" in result.output
    assert "Threshold for `detect --adapted`:" in result.output


def test_adapt_command_needs_crops(tmp_path):
    result = runner.invoke(app, ["adapt", str(tmp_path), "--out", str(tmp_path / "model")])

    assert result.exit_code == 1
    assert "no crops.csv" in result.output


def test_detect_command_uses_adapted_layer(make_image, monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from test_adapt import make_crops, tiny_speciesnet

    from camtraptools import adapt

    adapt.run(make_crops(tmp_path / "crops"), tmp_path / "model", device="cpu", speciesnet=tiny_speciesnet())
    monkeypatch.setattr("camtraptools.adapt.load_speciesnet", lambda model: tiny_speciesnet())
    monkeypatch.setattr("camtraptools.adapt.best_device", lambda: "cpu")
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    write_speciesnet_results(tmp_path / "package", 0.9)

    result = runner.invoke(
        app,
        [
            "detect", str(tmp_path / "package"), "--images", str(tmp_path / "images"),
            "--adapted", str(tmp_path / "model"), "--adapted-threshold", "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Adapted layer (2 labels) at confidence 0.000 or higher: used for 1 of 1 animal detections" in result.output
    observation = pd.read_csv(tmp_path / "package" / "observations.csv").iloc[0]
    assert observation["classifiedBy"].startswith("SpeciesNet 4.0.3a with adapted layer (2 labels")


def test_detect_adapted_needs_images(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])

    result = runner.invoke(app, ["detect", str(tmp_path / "package"), "--adapted", str(tmp_path)])

    assert result.exit_code == 1
    assert "--adapted needs --images" in result.output


def test_ecology_command_reports_naive_occupancy(tmp_path):
    from test_ecology import make_package

    package = make_package(tmp_path)

    result = runner.invoke(
        app, ["ecology", str(package), "--out", str(tmp_path / "out"), "--taxon", "Vulpes chama", "--taxon", "Lynx lynx"]
    )

    assert result.exit_code == 0, result.output
    assert "2 sites, 3 occasions of 7 days from 2024-03-01, 17 camera days." in result.output
    assert "No events of: Lynx lynx" in result.output
    assert "rai: events per 100 camera-days" in result.output
    assert "Species richness: 1 species recorded in 17 camera-days" in result.output
    assert "Vulpes chama" in result.output


def test_ecology_command_lists_left_out_events(tmp_path):
    from test_ecology import make_package

    result = runner.invoke(app, ["ecology", str(make_package(tmp_path)), "--out", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert "Left out 2 events not identified to species: Bovidae (1), animal (unidentified) (1)" in result.output


def test_ecology_command_needs_observations(tmp_path):
    result = runner.invoke(app, ["ecology", str(tmp_path), "--out", str(tmp_path / "out")])

    assert result.exit_code == 1
    assert "no observations.csv" in result.output


def test_ecology_command_reports_occupancy_model(tmp_path):
    from test_ecology import make_package

    package = make_package(tmp_path)
    observations = pd.read_csv(package / "observations.csv")
    deployments = pd.DataFrame(
        {
            "deploymentID": [f"cam{number}" for number in range(8)],
            "deploymentStart": "2024-03-01T12:00:00",
            "deploymentEnd": "2024-03-28T12:00:00",
        }
    )
    deployments.to_csv(package / "deployments.csv", index=False)
    events = observations.iloc[[0] * 6].assign(
        deploymentID=["cam0", "cam0", "cam1", "cam2", "cam3", "cam3"],
        eventID=[f"e{number}" for number in range(6)],
        eventStart=["2024-03-02", "2024-03-20", "2024-03-09", "2024-03-16", "2024-03-03", "2024-03-25"],
    )
    events.to_csv(package / "observations.csv", index=False)

    result = runner.invoke(app, ["ecology", str(package), "--out", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert "Occupancy model (taxa recorded at 3 or more sites)" in result.output
    assert "95% interval" in result.output
    assert "missedAtOccupiedSite" in result.output
    assert "No taxon has 20 or more events, the fewest for an activity curve." in result.output

    result = runner.invoke(
        app, ["ecology", str(package), "--out", str(tmp_path / "out"), "--min-activity-events", "2"]
    )

    assert "Daily activity (taxa with 2 or more events), busiest time of day:" in result.output
    assert "Vulpes chama: " in result.output
    assert (tmp_path / "out" / "activity.svg").exists()

    deployments = pd.read_csv(package / "deployments.csv")
    deployments["habitat"] = ["forest", "grassland"] * 4
    deployments.to_csv(package / "deployments.csv", index=False)
    result = runner.invoke(
        app, ["ecology", str(package), "--out", str(tmp_path / "out"), "--occupancy-covariate", "habitat"]
    )

    assert result.exit_code == 0, result.output
    assert "Occupancy with site covariates" in result.output
    assert "⚠ only 4 sites, exploratory" in result.output


def test_compare_command_reports_differences(tmp_path):
    from test_compare import make_package, write_truth

    package = make_package(tmp_path)

    result = runner.invoke(
        app, ["compare", str(package), "--out", str(tmp_path / "out"), "--truth", str(write_truth(tmp_path))]
    )

    assert result.exit_code == 0, result.output
    assert "Sources: machine, truth; compared with truth." in result.output
    assert "Differences from the reference:" in result.output
    assert "Sites where each species was recorded, where sources differ:" in result.output
    assert "Canis mesomelas" in result.output


def test_compare_command_with_one_source(tmp_path):
    from test_compare import make_package

    result = runner.invoke(app, ["compare", str(make_package(tmp_path)), "--out", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert "Only one source" in result.output


def test_compare_command_checks_other_packages(tmp_path):
    from test_compare import make_package

    result = runner.invoke(
        app, ["compare", str(make_package(tmp_path)), "--out", str(tmp_path / "out"), "--other", "adapted"]
    )

    assert result.exit_code == 2
    assert "NAME=PACKAGE" in result.output


def test_ingest_command_reports_skipped_files_and_reads_nested_folders(make_image, tmp_path):
    make_image("images/north/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    (tmp_path / "images/north/camera_A/IMG_0002.MP4").write_bytes(b"video")
    make_image("images/north/IMG_0003.JPG")

    result = runner.invoke(
        app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package"), "--folder-depth", "2"]
    )

    assert result.exit_code == 0, result.output
    assert "Found 1 image from 1 deployment." in result.output
    assert "Skipped 1 file that is not a JPEG image: .mp4 (1)" in result.output
    assert "Skipped 1 JPEG image not inside a deployment folder" in result.output
    assert pd.read_csv(tmp_path / "package" / "deployments.csv")["deploymentID"].tolist() == ["north_camera_A"]


def test_ingest_command_reports_clock_corrections(make_image, tmp_path):
    make_image("images/cam_reset/IMG_0001.JPG", date_time_original="2010:01:01 00:05:00")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "deploymentID,latitude,longitude,cameraTime,actualTime\n"
        "cam_reset,52.7,23.8,2010-01-01T00:00:00,2024-03-01T08:00:00\n"
    )

    result = runner.invoke(
        app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package"), "--field-sheet", str(sheet)]
    )
    checked = runner.invoke(app, ["check", str(tmp_path / "package")])

    assert result.exit_code == 0, result.output
    assert "Corrected the clock of cam_reset by +5173 days 8:00:00." in result.output
    assert "before 2010" not in checked.output


def test_labels_command_imports_a_csv(make_image, monkeypatch, tmp_path):
    monkeypatch.setattr("camtraptools.labels.speciesnet_taxonomy", lambda: None)
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    make_image("images/camera_A/IMG_0002.JPG", date_time_original="2024:03:01 06:01:00")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    source = tmp_path / "sheet.csv"
    source.write_text("image,what\ncamera_A/IMG_0001.JPG,Vulpes chama\nother/IMG_0009.JPG,Vulpes chama\n")

    result = runner.invoke(
        app,
        [
            "labels", str(tmp_path / "package"), str(source), "--out", str(tmp_path / "truth.csv"),
            "--species-column", "what", "--path-column", "image",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Read 2 labels (csv); matched 1 of the package's 2 images." in result.output
    assert "1 label matched no image in the package" in result.output
    assert "Vulpes chama: 1" in result.output


def test_labels_command_explains_missing_species_column(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])
    source = tmp_path / "sheet.csv"
    source.write_text("image,what\ncamera_A/IMG_0001.JPG,fox\n")

    result = runner.invoke(app, ["labels", str(tmp_path / "package"), str(source)])

    assert result.exit_code == 1
    assert "--species-column" in result.output


def test_export_command_warns_about_people(tmp_path):
    from test_export import package_with_a_person

    package = package_with_a_person(tmp_path)
    arguments = ["export", str(package), "--title", "T", "--contributor", "Ivan", "--timezone", "Africa/Johannesburg"]

    warned = runner.invoke(app, [*arguments, "--out", str(tmp_path / "a")])
    excluded = runner.invoke(app, [*arguments, "--out", str(tmp_path / "b"), "--exclude-humans", "--coordinate-decimals", "2"])

    assert "⚠ 1 image show people; use --exclude-humans" in warned.output.replace("image shows", "image show")
    assert "Left out 1 image of people and their observations." in excluded.output
    assert "Rounded coordinates to 2 decimals" in excluded.output


def test_summary_command_writes_a_page(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    make_image("images/camera_A/IMG_0002.JPG", date_time_original="2024:03:02 21:00:00")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("deploymentID,latitude,longitude,deploymentStart,deploymentEnd\ncamera_A,52.7,23.8,2024-03-01,2024-03-10\n")
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package"), "--field-sheet", str(sheet)])
    write_speciesnet_results(tmp_path / "package", 0.9)
    runner.invoke(app, ["detect", str(tmp_path / "package")])
    runner.invoke(app, ["events", str(tmp_path / "package")])

    result = runner.invoke(
        app, ["summary", str(tmp_path / "package"), "--out", str(tmp_path / "s.html"), "--title", "Test <survey>"]
    )

    assert result.exit_code == 0, result.output
    page = (tmp_path / "s.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>")
    assert "<h1>Test &lt;survey&gt;</h1>" in page
    assert "Oryx gazella" in page
    assert "No events were reviewed by a person" in page
    assert "Species richness" in page
