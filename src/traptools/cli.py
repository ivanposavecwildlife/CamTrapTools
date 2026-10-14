"""Command-line interface. Each module of the toolkit is one subcommand."""

from pathlib import Path

import typer

from traptools import __version__, check, ingest

app = typer.Typer(help="Camera-trap toolkit: from SD card to ecological answers.")


def _version(show: bool) -> None:
    if show:
        typer.echo(f"traptools {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True),
) -> None:
    pass


@app.command("ingest")
def ingest_cmd(
    images: Path = typer.Argument(..., help="Folder of SD-card images, one subfolder per camera."),
    out: Path = typer.Option(Path("package"), help="Where to write the Camtrap DP package."),
) -> None:
    """Scan images and build deployments.csv and media.csv."""
    ingest.run(images, out)


@app.command("check")
def check_cmd(
    package: Path = typer.Argument(..., help="Camtrap DP package folder made by `ingest`."),
) -> None:
    """Write a health report for each camera deployment."""
    check.run(package)
