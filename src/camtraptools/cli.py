"""Command-line interface. Each module of the toolkit is one subcommand."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from camtraptools import __version__, check, detect, ingest, report

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
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Camtrap DP package folder made by `ingest`.",
        ),
    ],
    images: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=False,
            help="The SD-card folder that was ingested into the package.",
        ),
    ],
    model: Annotated[
        str, typer.Option(help="MegaDetector model name or model file.")
    ] = detect.DEFAULT_MODEL,
    threshold: Annotated[
        float,
        typer.Option(
            help="Minimum confidence for counting a detection in the summary."
        ),
    ] = detect.DEFAULT_THRESHOLD,
) -> None:
    """Find animals, people and vehicles in every image with MegaDetector."""
    if not (package / "media.csv").exists():
        typer.echo(
            f"Error: no media.csv in {package}; run `camtraptools ingest` first.",
            err=True,
        )
        raise typer.Exit(1)
    try:
        results_file = detect.run(package, images, model)
    except RuntimeError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    media = pd.read_csv(package / "media.csv")
    summary = detect.summarize_detections(
        media, detect.load_detections(results_file), threshold
    )
    typer.echo(f"Images with detections at confidence {threshold:g} or higher:")
    typer.echo(summary.to_string())
    failed = detect.failed_images(results_file)
    if failed:
        typer.echo(
            f"{len(failed)} image(s) could not be processed, for example {failed[0]}"
        )
    typer.echo(f"Wrote {results_file}")
