"""Command-line interface. Each module of the toolkit is one subcommand."""

from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import typer

from camtraptools import (
    __version__,
    check,
    detect,
    events,
    export,
    importer,
    ingest,
    report,
    review,
)

app = typer.Typer(
    help="CamTrapTools: camera-trap toolkit, from SD card to ecological answers."
)


def _version(show: bool) -> None:
    if show:
        typer.echo(f"CamTrapTools {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True)
    ] = False,
) -> None:
    pass


@app.command("ingest")
def ingest_cmd(
    images: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder of SD-card images, one subfolder per camera.",
        ),
    ],
    out: Annotated[
        Path, typer.Option(help="Where to write the Camtrap DP package.")
    ] = Path("package"),
    field_sheet: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="CSV or Excel field sheet with coordinates and camera details.",
        ),
    ] = None,
) -> None:
    """Scan images and build deployments.csv and media.csv."""
    try:
        media, deployments, issues = ingest.run(images, out, field_sheet)
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Found {len(media)} images from {len(deployments)} deployments.")
    if issues:
        typer.echo(f"{len(issues)} field-sheet issue(s):")
        for issue in issues:
            typer.echo(f"  ! {issue}")
    typer.echo(f"Wrote {out / 'media.csv'} and {out / 'deployments.csv'}")


@app.command("check")
def check_cmd(
    package: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Camtrap DP package folder made by `ingest`.",
        ),
    ],
    max_gap_days: Annotated[
        float,
        typer.Option(help="Report periods longer than this many days without images."),
    ] = check.MAX_GAP_DAYS,
) -> None:
    """Check each deployment for camera and data problems."""
    try:
        results = check.run(package, max_gap_days)
        summary = check.summarize(package)
    except FileNotFoundError:
        typer.echo(
            f"Error: no media.csv in {package}; run `camtraptools ingest` first.",
            err=True,
        )
        raise typer.Exit(1)
    for deployment_id, issues in results.items():
        numbers = summary.loc[deployment_id]
        typer.echo(
            f"{deployment_id}: {numbers['days']} days, {int(numbers['images'])} images, "
            f"longest gap {numbers['longestGapDays']} days"
        )
        if issues:
            for issue in issues:
                typer.echo(f"  ! {issue}")
        else:
            typer.echo("  OK")


@app.command("report")
def report_cmd(
    package: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Camtrap DP package folder made by `ingest`.",
        ),
    ],
    out: Annotated[Path, typer.Option(help="Where to write the HTML report.")] = Path(
        "report.html"
    ),
    max_gap_days: Annotated[
        float,
        typer.Option(help="Report periods longer than this many days without images."),
    ] = check.MAX_GAP_DAYS,
) -> None:
    """Write an HTML health report with a timeline of every deployment."""
    try:
        report.run(package, out, max_gap_days)
    except FileNotFoundError:
        typer.echo(
            f"Error: no media.csv in {package}; run `camtraptools ingest` first.",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(f"Wrote {out}")


@app.command("detect")
def detect_cmd(
    package: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Camtrap DP package folder made by `ingest`."),
    ],
    images: Annotated[
        Path | None,
        typer.Option(exists=True, file_okay=False, help="The SD-card folder that was ingested into the package."),
    ] = None,
    country: Annotated[
        str | None, typer.Option(help="Three-letter country code (ISO 3166-1 alpha-3), e.g. ZAF, for geofencing.")
    ] = None,
    admin1_region: Annotated[
        str | None, typer.Option(help="State or province within the country (ISO 3166-2), e.g. CA.")
    ] = None,
    model: Annotated[str, typer.Option(help="SpeciesNet model to use.")] = detect.DEFAULT_MODEL,
    threshold: Annotated[
        float, typer.Option(help="Minimum detection confidence for keeping a detection as an observation.")
    ] = detect.DEFAULT_THRESHOLD,
    rerun: Annotated[bool, typer.Option(help="Run SpeciesNet even if results for these settings exist.")] = False,
) -> None:
    """Find and identify animals, people and vehicles in every image with SpeciesNet."""
    if not (package / "media.csv").exists():
        typer.echo(f"Error: no media.csv in {package}; run `camtraptools ingest` first.", err=True)
        raise typer.Exit(1)
    results_file = package / detect.RESULTS_FILE
    if rerun or not detect.has_results(results_file, model, country):
        if images is None:
            typer.echo("Error: --images is required to run SpeciesNet.", err=True)
            raise typer.Exit(1)
        try:
            results_file = detect.run(package, images, model, country, admin1_region)
        except RuntimeError as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo(f"Using existing results in {results_file} (pass --rerun to run SpeciesNet again).")
    report_detections(package, results_file, threshold)


def report_detections(package: Path, results_file: Path, threshold: float) -> None:
    """Print detection and species summaries and write observations.csv."""
    media = pd.read_csv(package / "media.csv")
    detections = detect.load_detections(results_file)
    typer.echo(f"Images with detections at confidence {threshold:g} or higher:")
    typer.echo(detect.summarize_detections(media, detections, threshold).to_string())
    species = detect.summarize_species(media, detections, threshold)
    if not species.empty:
        typer.echo("\nImages per species:")
        typer.echo(species.to_string())
    failed = detect.failed_images(results_file)
    if failed:
        typer.echo(f"{len(failed)} image(s) could not be fully processed, for example {failed[0]}")
    observations = detect.write_observations(package, threshold)
    typer.echo(f"Wrote {results_file} and {package / detect.OBSERVATIONS_FILE} ({len(observations)} observations)")


@app.command("import")
def import_cmd(
    package: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Camtrap DP package folder made by `ingest`."),
    ],
    results: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, help="Results from MegaDetector, AddaxAI, Timelapse or SpeciesNet (.json)."
        ),
    ],
    threshold: Annotated[
        float, typer.Option(help="Minimum detection confidence for keeping a detection as an observation.")
    ] = detect.DEFAULT_THRESHOLD,
) -> None:
    """Import detection and classification results made by another tool."""
    if not (package / "media.csv").exists():
        typer.echo(f"Error: no media.csv in {package}; run `camtraptools ingest` first.", err=True)
        raise typer.Exit(1)
    try:
        matched, ignored = importer.run(package, results)
    except (ValueError, KeyError) as error:
        typer.echo(f"Error: could not import {results.name}: {error}", err=True)
        raise typer.Exit(1)
    images = len(pd.read_csv(package / "media.csv"))
    typer.echo(f"Imported results for {matched} of {images} images.")
    if ignored:
        typer.echo(f"Ignored {ignored} images in {results.name} that are not in the package.")
    if matched < images:
        typer.echo(f"{images - matched} images in the package have no results and are marked unclassified.")
    report_detections(package, package / detect.RESULTS_FILE, threshold)


@app.command("events")
def events_cmd(
    package: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Camtrap DP package folder with observations from `detect`."),
    ],
    gap: Annotated[
        float, typer.Option(help="Start a new event after this many minutes without images.")
    ] = events.DEFAULT_GAP_MINUTES,
) -> None:
    """Group images into independent events and count them per species."""
    if not (package / detect.OBSERVATIONS_FILE).exists():
        typer.echo(f"Error: no observations.csv in {package}; run `camtraptools detect` first.", err=True)
        raise typer.Exit(1)
    event_observations = events.run(package, gap)
    images = len(pd.read_csv(package / "media.csv"))
    typer.echo(f"Grouped {images} images into {len(event_observations)} events ({gap:g}-minute gap).")
    table = events.summarize_events(event_observations)
    if not table.empty:
        typer.echo("\nIndependent animal events per taxon:")
        typer.echo(table.to_string())
    typer.echo(f"Wrote {package / detect.OBSERVATIONS_FILE}")


@app.command("export")
def export_cmd(
    package: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Package folder with observations from `detect` or `import`."),
    ],
    out: Annotated[Path, typer.Option(help="Folder for the Camtrap DP data package.")],
    title: Annotated[str, typer.Option(help="Title of the project.")],
    contributor: Annotated[
        list[str], typer.Option(help="Contributor name; repeat for several. The first is the contact.")
    ],
    timezone: Annotated[
        str, typer.Option(help="Time zone the camera clocks were set to, e.g. Africa/Johannesburg or Europe/Zagreb.")
    ],
    sampling_design: Annotated[
        str, typer.Option(help=f"One of: {', '.join(export.SAMPLING_DESIGNS)}.")
    ] = "opportunistic",
    license: Annotated[str, typer.Option(help="License of the data, e.g. CC0-1.0 or CC-BY-4.0.")] = "CC-BY-4.0",
    media_license: Annotated[str, typer.Option(help="License of the images.")] = "CC-BY-4.0",
) -> None:
    """Write a complete Camtrap DP data package, ready to validate and publish."""
    if not (package / detect.OBSERVATIONS_FILE).exists():
        typer.echo(f"Error: no observations.csv in {package}; run `camtraptools detect` or `import` first.", err=True)
        raise typer.Exit(1)
    if sampling_design not in export.SAMPLING_DESIGNS:
        typer.echo(f"Error: --sampling-design must be one of: {', '.join(export.SAMPLING_DESIGNS)}.", err=True)
        raise typer.Exit(1)
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        typer.echo(f"Error: unknown time zone {timezone!r}; use a name such as Africa/Johannesburg.", err=True)
        raise typer.Exit(1)
    try:
        left_out = export.run(package, out, title, contributor, timezone, sampling_design, license, media_license)
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    if left_out["media_without_timestamp"]:
        typer.echo(f"Left out {left_out['media_without_timestamp']} images without a timestamp.")
    if left_out["extra_columns"]:
        typer.echo(f"Left out deployment columns that are not part of Camtrap DP: {', '.join(left_out['extra_columns'])}")
    typer.echo(f"Wrote a Camtrap DP {export.CAMTRAP_DP_VERSION} data package to {out}")


@app.command("review")
def review_cmd(
    package: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Package folder with events from `camtraptools events`."),
    ],
    images: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False, help="The SD-card folder that was ingested into the package."),
    ],
    reviewer: Annotated[
        str | None, typer.Option(help="Name recorded with your review decisions (default: your user name).")
    ] = None,
    port: Annotated[int, typer.Option(help="Port for the local web server.")] = 8765,
    open_browser: Annotated[bool, typer.Option(help="Open the review page in the browser.")] = True,
) -> None:
    """Review the AI's observations in your browser, one event at a time."""
    import getpass
    import webbrowser
    from threading import Timer

    import uvicorn

    if not (package / detect.OBSERVATIONS_FILE).exists():
        typer.echo(f"Error: no observations.csv in {package}; run `camtraptools detect` first.", err=True)
        raise typer.Exit(1)
    try:
        app_ = review.create_app(package, images, reviewer or getpass.getuser())
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"Reviewing {package} at {url} (press Ctrl+C to stop)")
    if open_browser:
        Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app_, host="127.0.0.1", port=port, log_level="warning")
