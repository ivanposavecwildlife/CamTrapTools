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
    runner.invoke(app, ["ingest", str(tmp_path / "images"), "--out", str(tmp_path / "package")])

    default = runner.invoke(app, ["check", str(tmp_path / "package")])
    strict = runner.invoke(app, ["check", str(tmp_path / "package"), "--max-gap-days", "7"])

    assert "  OK" in default.output
    assert "no images for 10.0 days" in strict.output
