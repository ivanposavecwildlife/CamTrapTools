import pandas as pd

from camtraptools.ingest import run as ingest
from camtraptools.report import (
    LABEL_WIDTH,
    build_report,
    chart_range,
    issues_html,
    run,
    summary_table_html,
    timeline_svg,
    x_position,
)

FIRST = pd.Timestamp("2024-03-01")
LAST = pd.Timestamp("2024-03-11")


def make_package(make_image, tmp_path):
    make_image("images/camera_A/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    make_image("images/camera_A/IMG_0002.JPG", date_time_original="2024:03:02 23:00:00")
    make_image("images/camera_B/IMG_0001.JPG", date_time_original="2000:01:01 00:05:00")
    make_image("images/camera_B/IMG_0002.JPG", date_time_original="2024:03:05 12:00:00")
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(
        "deploymentID,latitude,longitude,deploymentStart,deploymentEnd\n"
        "camera_A,52.7,23.85,2024-03-01,2024-04-10\n"
        "camera_B,52.7,23.85,2024-03-01,2024-03-10\n"
    )
    ingest(tmp_path / "images", tmp_path / "package", sheet)
    return tmp_path / "package"


def test_x_position_maps_first_and_last_to_chart_edges():
    assert x_position(FIRST, FIRST, LAST) == LABEL_WIDTH
    assert x_position(LAST, FIRST, LAST) > x_position(FIRST + pd.Timedelta(days=5), FIRST, LAST)


def test_chart_range_ignores_implausible_dates():
    media = pd.DataFrame({"timestamp": pd.to_datetime(["2000-01-01", "2024-03-05"])})
    deployments = pd.DataFrame(
        {"deploymentStart": pd.to_datetime(["2024-03-01"]), "deploymentEnd": pd.to_datetime(["2024-03-11"])}
    )

    assert chart_range(media, deployments) == (FIRST, LAST)


def test_timeline_has_a_row_per_deployment_and_marks_gaps(make_image, tmp_path):
    package = make_package(make_image, tmp_path)
    media = pd.read_csv(package / "media.csv", parse_dates=["timestamp"])
    deployments = pd.read_csv(package / "deployments.csv", parse_dates=["deploymentStart", "deploymentEnd"])

    svg = timeline_svg(media, deployments.set_index("deploymentID"), max_gap_days=21)

    assert svg.startswith("<svg")
    assert svg.count('class="label"') == 2
    assert svg.count('class="image"') == 3
    assert 'class="gap"' in svg
    assert "⚠ 39 d" in svg


def test_timeline_without_dates_says_so():
    media = pd.DataFrame({"deploymentID": ["camera_A"], "timestamp": [pd.NaT]})

    assert timeline_svg(media, pd.DataFrame(), max_gap_days=21) == "<p>No dated images to chart.</p>"


def test_summary_table_shows_numbers_and_status():
    summary = pd.DataFrame(
        {"days": [10.0], "images": [2], "imagesPerDay": [0.2], "nightShare": [0.5], "longestGapDays": [5.5]},
        index=["camera_A"],
    )

    table = summary_table_html(summary, {"camera_A": []})

    assert "<td class=\"num\">50%</td>" in table
    assert "✓</span> OK" in table


def test_issues_are_escaped_and_grouped():
    html = issues_html({"camera_<A>": ["clock went back at <IMG>"], "camera_B": []})

    assert "<h3>camera_&lt;A&gt;</h3>" in html
    assert "clock went back at &lt;IMG&gt;" in html
    assert "camera_B" not in html


def test_issues_without_problems():
    assert "No problems found." in issues_html({"camera_A": []})


def test_build_report_is_a_complete_page(make_image, tmp_path):
    package = make_package(make_image, tmp_path)

    page = build_report(package)

    assert page.startswith("<!doctype html>")
    assert "<strong>2</strong><span>deployments</span>" in page
    assert "<strong>2</strong><span>with issues</span>" in page
    assert "1 image(s) dated before 2010" in page


def test_run_writes_report_file(make_image, tmp_path):
    package = make_package(make_image, tmp_path)

    run(package, tmp_path / "report.html")

    assert "Camera health report" in (tmp_path / "report.html").read_text(encoding="utf-8")
