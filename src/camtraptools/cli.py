"""Command-line interface. Each module of the toolkit is one subcommand."""

from pathlib import Path

import typer

from camtraptools import __version__, check, ingest

app = typer.Typer(
    help="CamTrapTools: camera-trap toolkit, from SD card to ecological answers."
)


def _version(show: bool) -> None:
    if show:
        typer.echo(f"CamTrapTools {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True),
) -> None:
    pass


@app.command("ingest")
def ingest_cmd(
    images: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Folder of SD-card images, one subfolder per camera.",
    ),
    out: Path = typer.Option(
        Path("package"), help="Where to write the Camtrap DP package."
    ),
    field_sheet: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="CSV or Excel field sheet with coordinates and camera details.",
    ),
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
    package: Path = typer.Argument(
        ..., help="Camtrap DP package folder made by `ingest`."
    ),
) -> None:
    """Write a health report for each camera deployment."""
    check.run(package)
