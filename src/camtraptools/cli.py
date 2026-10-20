"""Command-line interface. Each module of the toolkit is one subcommand."""

from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import typer

from camtraptools import (
    __version__,
    activity,
    adapt,
    check,
    compare,
    crops,
    detect,
    ecology,
    evaluate,
    events,
    export,
    importer,
    ingest,
    labels,
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
            help="Folder of SD-card images, one subfolder per camera (see --folder-depth).",
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
    folder_depth: Annotated[
        int,
        typer.Option(
            min=1,
            help="Folder level that holds one deployment each, e.g. 2 for site/camera/ (deployment site_camera).",
        ),
    ] = 1,
) -> None:
    """Scan images and build deployments.csv and media.csv."""
    try:
        media, deployments, issues, corrections = ingest.run(images, out, field_sheet, folder_depth)
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Found {crops.plural(len(media), 'image')} from {crops.plural(len(deployments), 'deployment')}.")
    skipped = ingest.skipped_files(images, folder_depth)
    outside = skipped.pop(ingest.OUTSIDE_DEPLOYMENTS, 0)
    if skipped:
        kinds = ", ".join(f"{kind} ({count})" for kind, count in skipped.most_common())
        count = sum(skipped.values())
        what = "1 file that is not a JPEG image" if count == 1 else f"{count} files that are not JPEG images"
        typer.echo(f"Skipped {what}: {kinds}")
    if outside:
        typer.echo(
            f"Skipped {crops.plural(outside, 'JPEG image')} not inside a deployment folder; "
            f"check the folder layout or --folder-depth (now {folder_depth})."
        )
    for deployment_id, offset in corrections.items():
        typer.echo(f"Corrected the clock of {deployment_id} by {ingest.format_offset(offset)}.")
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
    adapted: Annotated[
        Path | None,
        typer.Option(exists=True, help="Folder with an adapted layer from `camtraptools adapt`, to identify animals."),
    ] = None,
    adapted_threshold: Annotated[
        float | None,
        typer.Option(help="Confidence above which the adapted layer's answer is used; default: the layer's own."),
    ] = None,
) -> None:
    """Find and identify animals, people and vehicles in every image with SpeciesNet."""
    if not (package / "media.csv").exists():
        typer.echo(f"Error: no media.csv in {package}; run `camtraptools ingest` first.", err=True)
        raise typer.Exit(1)
    if adapted is not None and images is None:
        typer.echo("Error: --adapted needs --images to crop the animals.", err=True)
        raise typer.Exit(1)
    results_file = package / detect.RESULTS_FILE
    if rerun or not detect.has_results(results_file, model, country):
        if images is None:
            typer.echo("Error: --images is required to run SpeciesNet.", err=True)
            raise typer.Exit(1)
        media = pd.read_csv(package / "media.csv")
        done = detect.saved_progress(package, detect.run_settings(media, images, model, country, admin1_region))
        if done:
            typer.echo(f"Continuing an interrupted run: {done} of {crops.plural(len(media), 'image')} already done.")
        else:
            typer.echo(f"Running SpeciesNet on {crops.plural(len(media), 'image')}; progress is saved every 10 minutes.")
        try:
            results_file = detect.run(package, images, model, country, admin1_region)
        except RuntimeError as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo(f"Using existing results in {results_file} (pass --rerun to run SpeciesNet again).")
    if adapted is not None:
        try:
            layer = adapt.load_adapted(adapted)
            answers = adapt.classify_detections(package, images, layer, threshold, adapted_threshold)
        except (RuntimeError, ValueError) as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1) from error
        report_adapted(answers, layer, adapted_threshold)
    report_detections(package, results_file, threshold, use_adapted=adapted is not None)


def report_adapted(answers: pd.DataFrame, layer: dict, threshold: float | None) -> None:
    """Print how often the adapted layer's answer replaced SpeciesNet's."""
    threshold = layer["threshold"] if threshold is None else threshold
    if threshold is None:
        typer.echo("The adapted layer never reached SpeciesNet's precision on validation, so it is not used.")
        return
    changed = answers["used"] & (answers["adapted"] != answers["speciesnet"])
    typer.echo(
        f"Adapted layer ({len(layer['classes'])} labels) at confidence {threshold:.3f} or higher: "
        f"used for {answers['used'].sum()} of {len(answers)} animal detections, "
        f"changing {changed.sum()} of SpeciesNet's answers."
    )
    if changed.any():
        changes = answers[changed].fillna({"speciesnet": "animal"}).value_counts(["speciesnet", "adapted"])
        typer.echo("Most common changes (SpeciesNet → adapted):")
        for (before, after), count in changes.head(10).items():
            typer.echo(f"  {before} → {after}: {count}")


def report_detections(package: Path, results_file: Path, threshold: float, use_adapted: bool = False) -> None:
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
    observations = detect.write_observations(package, threshold, use_adapted)
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
    order: Annotated[
        str, typer.Option(help="Order of events: uncertainty (least confident first) or time.")
    ] = "uncertainty",
    port: Annotated[int, typer.Option(help="Port for the local web server; another is used if it is busy.")] = 8765,
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
    if order not in review.ORDERS:
        typer.echo(f"Error: --order must be one of: {', '.join(review.ORDERS)}.", err=True)
        raise typer.Exit(1)
    try:
        app_ = review.create_app(package, images, reviewer or getpass.getuser(), order)
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)
    free = review.free_port(port)
    if free != port:
        typer.echo(f"Port {port} is in use by another program; using {free} instead.")
    port = free
    url = f"http://127.0.0.1:{port}/"
    typer.echo(f"Reviewing {package} at {url} (press Ctrl+C to stop)")
    if open_browser:
        Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app_, host="127.0.0.1", port=port, log_level="warning")


@app.command("review-stats")
def review_stats_cmd(
    package: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Package folder with reviews from `camtraptools review`."),
    ],
) -> None:
    """Summarize review progress, agreement with the AI and review speed."""
    from camtraptools import reviews as review_data

    if not (package / review_data.REVIEWS_FILE).exists():
        typer.echo(f"No reviews in {package} yet; start with `camtraptools review`.")
        raise typer.Exit(0)
    compared = review_data.compare_with_machine(package)
    observations = pd.read_csv(package / detect.OBSERVATIONS_FILE, dtype=str)
    total = ((observations["observationLevel"] == "event") & (observations["classificationMethod"] != "human")).sum()
    agreed = (compared["outcome"] == "agreed").sum()
    typer.echo(f"Reviewed {len(compared)} of {total} events; agreed with the AI on {agreed} ({agreed / len(compared):.0%}).")
    changes = compared["outcome"].value_counts().drop("agreed", errors="ignore")
    if not changes.empty:
        typer.echo("\nChanges:")
        for outcome, count in changes.items():
            typer.echo(f"  {outcome}: {count}")
        changed = compared[compared["outcome"] != "agreed"]
        typer.echo("\nCorrections (AI → reviewer):")
        for row in changed.itertuples():
            before = row.scientificNameAI if isinstance(row.scientificNameAI, str) else row.observationTypeAI
            after = row.scientificName if isinstance(row.scientificName, str) else row.observationType
            typer.echo(f"  {row.eventID}: {before} → {after}")
    typer.echo("\nBy AI confidence:")
    typer.echo(review_data.summarize_by_confidence(compared).to_string())
    speed = review_data.review_speed(compared["reviewedAt"])
    if speed["perMinute"]:
        typer.echo(f"\nSpeed: {speed['perMinute']} decisions per minute over {speed['minutes']} active minutes.")


@app.command("evaluate")
def evaluate_cmd(
    package: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Package folder with events from `camtraptools events`."),
    ],
    truth: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="CSV of image labels (filePath, scientificName); default: the reviews."
        ),
    ] = None,
    target_precision: Annotated[
        float, typer.Option(help="Precision each label's confidence threshold should reach.")
    ] = evaluate.DEFAULT_TARGET_PRECISION,
    report: Annotated[Path | None, typer.Option(help="Write an HTML report of the mistakes, with images.")] = None,
    images: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False, help="The SD-card folder, needed for --report.")
    ] = None,
    deployment: Annotated[
        list[str] | None,
        typer.Option(help="Only evaluate events from this deployment; repeat for several (e.g. held-out cameras)."),
    ] = None,
) -> None:
    """Measure the AI against ground truth: precision, recall, thresholds and mistakes."""
    if report is not None and images is None:
        typer.echo("Error: --report needs --images to show the photos.", err=True)
        raise typer.Exit(1)
    if not (package / detect.OBSERVATIONS_FILE).exists():
        typer.echo(f"Error: no observations.csv in {package}; run `camtraptools detect` first.", err=True)
        raise typer.Exit(1)
    compared = evaluate.run(package, truth, deployment)
    if compared.empty:
        source = truth.name if truth else "reviews"
        typer.echo(f"No events to evaluate: {source} has no labels for this package's events.")
        raise typer.Exit(1)
    source = f"labels in {truth.name}" if truth else "your reviews"
    typer.echo(f"Compared {len(compared)} events with {source}.")
    outcomes = compared["outcome"].value_counts().reindex(evaluate.OUTCOMES, fill_value=0)
    for outcome, count in outcomes.items():
        typer.echo(f"  {outcome}: {count} ({count / len(compared):.0%})")
    typer.echo("\nPer label:")
    typer.echo(evaluate.per_class_metrics(compared).to_string(na_rep="–"))
    typer.echo(f"\nConfidence thresholds for {target_precision:.0%} precision (⚠ = under ten answers, unreliable):")
    table = evaluate.thresholds(compared, target_precision)
    table["fewAnswers"] = table["fewAnswers"].map({True: "⚠", False: ""})
    table["threshold"] = table["threshold"].map(lambda value: "never" if pd.isna(value) else f"{value:.3f}")
    typer.echo(table.to_string(na_rep="–"))
    mistakes = compared[compared["outcome"] != "correct"]
    if not mistakes.empty:
        typer.echo("\nNot correct (truth → AI):")
        for row in mistakes.itertuples():
            typer.echo(f"  {row.eventID}: {row.truth} → {row.ai} ({row.outcome})")
    typer.echo(f"\nWrote {package / evaluate.EVALUATION_FILE}")
    if report is not None:
        report.write_text(evaluate.failure_report(package, images, compared, target_precision), encoding="utf-8")
        typer.echo(f"Wrote {report}")


@app.command("crops")
def crops_cmd(
    package: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Package folder with events from `camtraptools events`."),
    ],
    images: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="The SD-card folder that was ingested into the package.")
    ],
    out: Annotated[Path, typer.Option(help="New folder for the crops, in train/, val/ and test/ subfolders.")],
    truth: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="CSV of image labels (filePath, scientificName); default: the reviews."
        ),
    ] = None,
    padding: Annotated[
        float, typer.Option(help="Background to add around each box, as a fraction of its size.")
    ] = crops.DEFAULT_PADDING,
    val: Annotated[float, typer.Option(help="Target share of crops for validation.")] = crops.DEFAULT_FRACTIONS["val"],
    test: Annotated[float, typer.Option(help="Target share of crops for testing.")] = crops.DEFAULT_FRACTIONS["test"],
    seed: Annotated[int, typer.Option(help="Seed for breaking ties when splitting locations.")] = 0,
) -> None:
    """Crop every labelled animal detection into a training dataset, split by camera location."""
    if not (package / detect.OBSERVATIONS_FILE).exists():
        typer.echo(f"Error: no observations.csv in {package}; run `camtraptools detect` first.", err=True)
        raise typer.Exit(1)
    if out.exists() and any(out.iterdir()):
        typer.echo(f"Error: {out} is not empty; choose a new folder so old crops are not mixed in.", err=True)
        raise typer.Exit(1)
    if not (0 <= val < 1 and 0 <= test < 1 and val + test < 1):
        typer.echo("Error: --val and --test must be between 0 and 1 and leave room for training.", err=True)
        raise typer.Exit(1)
    fractions = {"train": 1 - val - test, "val": val, "test": test}
    dataset, skipped = crops.run(package, images, out, truth, padding, fractions, seed)
    source = f"labels in {truth.name}" if truth else "your reviews"
    typer.echo(f"Saved {crops.plural(len(dataset), 'crop')} labelled from {source}.")
    for reason, count in skipped.items():
        if count:
            typer.echo(f"  skipped {count}: {reason}")
    if dataset.empty:
        raise typer.Exit(1)
    typer.echo("\nLocations per split (each location is in one split only):")
    for split, group in dataset.groupby("split", sort=False):
        typer.echo(f"  {split}: {', '.join(sorted(group['location'].unique()))} ({crops.plural(len(group), 'crop')})")
    typer.echo("\nCrops per label:")
    typer.echo(crops.summarize(dataset).to_string())
    problems = crops.warnings(dataset)
    if problems:
        typer.echo("\nWarnings:")
        for problem in problems:
            typer.echo(f"  ⚠ {problem}")
    typer.echo(f"\nWrote {out / crops.CROPS_FILE}")


@app.command("adapt")
def adapt_cmd(
    crops_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Folder of crops made by `camtraptools crops`."),
    ],
    out: Annotated[Path, typer.Option(help="Folder for the adapted model and its predictions.")],
    epochs: Annotated[int, typer.Option(help="Most training passes over the crops.")] = adapt.DEFAULT_EPOCHS,
    learning_rate: Annotated[float, typer.Option(help="Step size of the optimiser.")] = adapt.DEFAULT_LEARNING_RATE,
    seed: Annotated[int, typer.Option(help="Random seed, for repeatable training.")] = 0,
    balanced: Annotated[
        bool,
        typer.Option(help="Weight rare species as much as common ones: better on rare species, worse overall."),
    ] = False,
    device: Annotated[
        str | None, typer.Option(help="cpu, mps (Apple GPU) or cuda (NVIDIA GPU); default: the fastest available.")
    ] = None,
) -> None:
    """Adapt SpeciesNet to your region: train a new final layer on labelled crops."""
    if not (crops_dir / crops.CROPS_FILE).exists():
        typer.echo(f"Error: no crops.csv in {crops_dir}; run `camtraptools crops` first.", err=True)
        raise typer.Exit(1)
    try:
        predictions, summary = adapt.run(
            crops_dir, out, epochs=epochs, learning_rate=learning_rate, seed=seed, device=device, balanced=balanced
        )
    except (RuntimeError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    stopped = (
        f"keeping epoch {summary['epochs']}, which had the lowest validation loss"
        if summary["validated"]
        else f"for {summary['epochs']} epochs (no validation crops with these labels)"
    )
    typer.echo(f"Trained a new final layer for {len(summary['classes'])} labels on {summary['device']}, {stopped}.")
    for name, count, known in zip(summary["classes"], summary["trainingCrops"], summary["knownToSpeciesNet"], strict=True):
        start = "starts from SpeciesNet" if known else "new to SpeciesNet"
        typer.echo(f"  {name}: {crops.plural(count, 'training crop')}, {start}")
    typer.echo("\nCrops named correctly, among labels the layer was trained on:")
    typer.echo(adapt.accuracy_table(predictions).to_string(na_rep="–"))
    typer.echo("(speciesnet: as run by `detect`; regionalList: SpeciesNet choosing only among these labels)")
    held_out = next((split for split in ["test", "val"] if (predictions["split"] == split).any()), None)
    if held_out is not None and predictions.loc[predictions["split"] == held_out, "inTraining"].any():
        typer.echo(f"\nCrops named correctly per label, {held_out} cameras:")
        typer.echo(adapt.label_table(predictions, held_out).to_string())
    unseen = predictions.loc[~predictions["inTraining"], "label"].value_counts()
    if not unseen.empty:
        held_out = crops.plural(int(unseen.sum()), "held-out crop")
        typer.echo(f"\n⚠ Labels with no training crops, which the adapted layer cannot name ({held_out}):")
        typer.echo("  " + ", ".join(f"{label} ({count})" for label, count in unseen.items()))
    if summary["threshold"] is None:
        typer.echo(
            f"\n⚠ No confidence reached SpeciesNet's precision on validation ({summary['speciesnetPrecision']:.0%}), "
            "so `detect --adapted` will not use this layer unless you pass --adapted-threshold."
        )
    elif summary["speciesnetPrecision"] is None:
        typer.echo(f"\nNo validation crops to choose a threshold on; `detect --adapted` will use {summary['threshold']}.")
    else:
        typer.echo(
            f"\nThreshold for `detect --adapted`: {summary['threshold']:.3f}, where the adapted layer is as precise "
            f"on validation as SpeciesNet's species answers ({summary['speciesnetPrecision']:.0%})."
        )
    typer.echo(f"\nWrote {out / adapt.MODEL_FILE} and {out / adapt.PREDICTIONS_FILE}")


@app.command("ecology")
def ecology_cmd(
    package: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Package folder with events from `camtraptools events`."),
    ],
    out: Annotated[Path, typer.Option(help="Folder for the detection histories, effort and occupancy tables.")],
    occasion_days: Annotated[
        int, typer.Option(min=1, help="Days per sampling occasion in the detection histories.")
    ] = ecology.DEFAULT_OCCASION_DAYS,
    taxon: Annotated[
        list[str] | None, typer.Option(help="Only this taxon (scientific name); repeat for several.")
    ] = None,
    all_taxa: Annotated[bool, typer.Option(help="Include higher taxa such as Bovidae, not only species.")] = False,
    min_activity_events: Annotated[
        int, typer.Option(min=2, help="Fewest events for a taxon's activity curve and overlaps.")
    ] = activity.DEFAULT_MIN_EVENTS,
) -> None:
    """Estimate occupancy (naive and modelled) and daily activity for each species."""
    if not (package / detect.OBSERVATIONS_FILE).exists():
        typer.echo(f"Error: no observations.csv in {package}; run `camtraptools detect` first.", err=True)
        raise typer.Exit(1)
    try:
        summary, left_out, overlaps = ecology.run(package, out, occasion_days, taxon, all_taxa, min_activity_events)
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    effort = pd.read_csv(out / ecology.EFFORT_FILE, index_col="site")
    typer.echo(
        f"{len(effort)} sites, {len(effort.columns)} occasions of {occasion_days} days from {effort.columns[0]}, "
        f"{int(effort.to_numpy().sum())} camera days."
    )
    missing = sorted(set(taxon or []) - set(summary.index))
    if missing:
        typer.echo(f"No events of: {', '.join(missing)}")
    if summary.empty:
        typer.echo("No animal events to analyse.")
        raise typer.Exit(1)
    typer.echo(
        "\nRecords and naive occupancy (rai: events per 100 camera-days; naiveOccupancy: share of surveyed "
        "sites where the taxon was recorded):"
    )
    naive = ["events", "rai", "sitesDetected", "sitesSurveyed", "naiveOccupancy", "detectionRate"]
    typer.echo(summary[naive].to_string(na_rep="–"))
    modelled = summary.dropna(subset=["occupancy"]) if "occupancy" in summary.columns else summary.iloc[0:0]
    if not modelled.empty:
        table = modelled[["naiveOccupancy", "occupancy"]].copy()
        table["95% interval"] = [
            f"{low:.2f}–{high:.2f}" for low, high in zip(modelled["occupancyLower"], modelled["occupancyUpper"], strict=True)
        ]
        table["detectionPerOccasion"] = modelled["detectionPerOccasion"]
        table["missedAtOccupiedSite"] = modelled["missedAtOccupiedSite"]
        table["warning"] = modelled["warning"].map(lambda text: f"⚠ {text}" if isinstance(text, str) else "")
        typer.echo(
            f"\nOccupancy model (taxa recorded at {ecology.MIN_SITES_FOR_MODEL} or more sites), "
            "correcting for the species being missed where it is present:"
        )
        typer.echo(table.to_string())
        typer.echo(
            "detectionPerOccasion: chance of recording the species in one full occasion where it is present; "
            "missedAtOccupiedSite: chance of never recording it over the survey."
        )
    report_activity(summary, overlaps, min_activity_events)
    if not left_out.empty and not taxon:
        reason = "not identified to species" if not all_taxa else "not identified"
        typer.echo(f"\nLeft out {crops.plural(int(left_out.sum()), 'event')} {reason}: "
                   + ", ".join(f"{name} ({count})" for name, count in left_out.head(8).items()))
    typer.echo(
        f"Wrote {out / ecology.SUMMARY_FILE}, {out / ecology.RAI_FILE}, {out / ecology.EFFORT_FILE} "
        f"and {out / ecology.HISTORIES_FOLDER}/"
    )
    if (out / activity.PLOT_FILE).exists() and "activityPeak" in summary.columns and summary["activityPeak"].notna().any():
        typer.echo(f"Wrote {out / activity.ACTIVITY_FILE}, {out / activity.OVERLAP_FILE} and {out / activity.PLOT_FILE}")


def report_activity(summary: pd.DataFrame, overlaps: pd.DataFrame, min_events: int) -> None:
    """Print each species' activity peak and the overlap between species' activity."""
    if "activityPeak" not in summary.columns or summary["activityPeak"].isna().all():
        typer.echo(f"\nNo taxon has {min_events} or more events, the fewest for an activity curve.")
        return
    peaks = summary.dropna(subset=["activityPeak"])
    typer.echo(f"\nDaily activity (taxa with {min_events} or more events), busiest time of day:")
    for name, row in peaks.iterrows():
        typer.echo(f"  {name}: {row['activityPeak']} ({int(row['events'])} events)")
    if not overlaps.empty:
        typer.echo("\nActivity overlap (0 = never active at the same time, 1 = identical), with 95% interval:")
        for row in overlaps.itertuples():
            typer.echo(
                f"  {row.first} & {row.second}: {row.overlap:.2f} ({row.overlapLower:.2f}–{row.overlapUpper:.2f}, "
                f"{row.estimator})"
            )


def parse_other(value: str) -> tuple[str, Path]:
    """Split a NAME=PACKAGE option into its name and folder."""
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise typer.BadParameter(f"expected NAME=PACKAGE, got {value!r}")
    if not (Path(path) / detect.OBSERVATIONS_FILE).exists():
        raise typer.BadParameter(f"no observations.csv in {path}")
    return name, Path(path)


@app.command("compare")
def compare_cmd(
    package: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="Package folder with events from `camtraptools events`."),
    ],
    out: Annotated[Path, typer.Option(help="Folder for each source's ecology results and the comparison.")],
    truth: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="CSV of image labels (filePath, scientificName) as ground truth."),
    ] = None,
    other: Annotated[
        list[str] | None,
        typer.Option(help="Another package of the same survey as NAME=FOLDER, e.g. adapted=survey_adapted; repeatable."),
    ] = None,
    occasion_days: Annotated[
        int, typer.Option(min=1, help="Days per sampling occasion in the detection histories.")
    ] = ecology.DEFAULT_OCCASION_DAYS,
    min_activity_events: Annotated[
        int, typer.Option(min=2, help="Fewest events for a taxon's activity curve and overlaps.")
    ] = activity.DEFAULT_MIN_EVENTS,
) -> None:
    """Compare occupancy and activity between label sources: machine, reviewed, truth and other packages."""
    if not (package / detect.OBSERVATIONS_FILE).exists():
        typer.echo(f"Error: no observations.csv in {package}; run `camtraptools detect` first.", err=True)
        raise typer.Exit(1)
    others = dict(parse_other(value) for value in other or [])
    try:
        reference, comparison, overlaps, scores = compare.run(
            package, out, truth, others, occasion_days, min_activity_events
        )
    except ValueError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    names = list(dict.fromkeys(comparison["source"])) if not comparison.empty else [reference]
    typer.echo(f"Sources: {', '.join(names)}; compared with {reference}.")
    if scores.empty:
        typer.echo("Only one source; add --truth, reviews or --other to compare.")
        return
    typer.echo("\nDifferences from the reference:")
    typer.echo(scores.to_string(na_rep="–"))
    typer.echo(
        "(siteRecords: species recorded at a site; errors are mean absolute differences over species, "
        "or over species pairs for overlap; activityDifference: 1 − overlap of each species' own activity curves)"
    )

    def wide(column: str) -> pd.DataFrame:
        return comparison.pivot_table(index="taxon", columns="source", values=column, aggfunc="first").reindex(
            columns=names
        )

    sites = wide("sitesDetected").fillna(0).astype(int)
    differing = sites[sites.nunique(axis=1) > 1].sort_values(reference, ascending=False)
    if not differing.empty:
        typer.echo("\nSites where each species was recorded, where sources differ:")
        typer.echo(differing.to_string())
    if "occupancy" in comparison.columns:
        occupancy = wide("occupancy").dropna(how="all")
        if not occupancy.empty:
            typer.echo("\nModelled occupancy (– where a source could not model the species):")
            typer.echo(occupancy.sort_values(reference, ascending=False).to_string(na_rep="–"))
    if "activityPeak" in comparison.columns:
        peaks = wide("activityPeak").dropna(how="all")
        if not peaks.empty:
            typer.echo("\nBusiest time of day:")
            typer.echo(peaks.to_string(na_rep="–"))
    if not overlaps.empty:
        table = overlaps.pivot_table(index=["first", "second"], columns="source", values="overlap", aggfunc="first")
        typer.echo("\nActivity overlap:")
        typer.echo(table.reindex(columns=[name for name in names if name in table.columns]).to_string(na_rep="–"))
    typer.echo(
        f"\nWrote {out / compare.SCORES_FILE}, {out / compare.COMPARISON_FILE}, "
        f"{out / compare.OVERLAP_COMPARISON_FILE} and one folder per source"
    )


@app.command("labels")
def labels_cmd(
    package: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Camtrap DP package folder made by `ingest`.")
    ],
    source: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Existing labels: a Camtrap DP package folder, a Wildlife Insights images.csv, or a CSV (e.g. Timelapse).",
        ),
    ],
    out: Annotated[Path, typer.Option(help="Truth file to write, for evaluate, crops and compare.")] = Path(
        "labels.csv"
    ),
    source_format: Annotated[
        str, typer.Option("--format", help="auto, camtrap-dp, wildlife-insights or csv.")
    ] = "auto",
    species_column: Annotated[
        str | None, typer.Option(help="CSV column with the species or label, e.g. Species.")
    ] = None,
    path_column: Annotated[str | None, typer.Option(help="CSV column with the image path.")] = None,
    file_column: Annotated[
        str | None, typer.Option(help="CSV column with the image file name (default: File, as in Timelapse).")
    ] = None,
    folder_column: Annotated[
        str | None, typer.Option(help="CSV column with the image folder (default: RelativePath, as in Timelapse).")
    ] = None,
    names: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="CSV mapping your labels to scientific names (label, scientificName)."),
    ] = None,
) -> None:
    """Import existing image labels (Camtrap DP, Wildlife Insights, Timelapse or any CSV) as a truth file."""
    if source_format not in labels.FORMATS:
        typer.echo(f"Error: --format must be one of {', '.join(labels.FORMATS)}.", err=True)
        raise typer.Exit(1)
    if not (package / "media.csv").exists():
        typer.echo(f"Error: no media.csv in {package}; run `camtraptools ingest` first.", err=True)
        raise typer.Exit(1)
    try:
        summary = labels.run(
            package, source, out, source_format, species_column, path_column, file_column, folder_column, names
        )
    except (ValueError, KeyError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(
        f"Read {crops.plural(summary['rows'], 'label')} ({summary['format']}); matched "
        f"{summary['matched']} of the package's {crops.plural(summary['packageImages'], 'image')}."
    )
    if summary["missing"]:
        typer.echo(f"  {crops.plural(summary['missing'], 'label')} matched no image in the package")
    if summary["ambiguous"]:
        typer.echo(
            f"  {crops.plural(summary['ambiguous'], 'label')} matched several images with the same file name; "
            "give a path column to tell them apart"
        )
    if summary["matched"]:
        typer.echo("\nImages per label:")
        for label, count in summary["labels"].head(20).items():
            typer.echo(f"  {label}: {count}")
    if summary["unknown"]:
        typer.echo(
            f"\n⚠ Not recognised as a scientific or common name: {', '.join(summary['unknown'][:10])}. "
            "Map them with --names (columns label, scientificName)."
        )
    typer.echo(f"\nWrote {out}; use it with --truth in evaluate, crops and compare.")
