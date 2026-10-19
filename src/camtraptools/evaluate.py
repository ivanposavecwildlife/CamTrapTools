"""Measure the AI's event-level decisions against ground truth."""

import base64
import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape
from PIL import Image

from camtraptools import __version__
from camtraptools.detect import OBSERVATIONS_FILE, RESULTS_FILE, parse_taxon
from camtraptools.importer import match_path
from camtraptools.reviews import load_reviews, order_events

EVALUATION_FILE = "evaluation.csv"
OUTCOMES = ["correct", "not to species", "unverified", "wrong"]
RANKS = ["genus", "family", "order", "class"]
NON_TAXON_LABELS = {"animal", "blank", "human", "vehicle", "unknown", "unclassified"}


def event_label(observation_type: str, scientific_name) -> str:
    """Return one comparable label for a decision: the scientific name for
    identified animals, otherwise the observation type (e.g. blank, animal)."""
    if observation_type == "animal" and isinstance(scientific_name, str) and scientific_name:
        return scientific_name
    return observation_type


def machine_labels(observations: pd.DataFrame) -> pd.Series:
    """Return the AI's label for every event, by event ID."""
    events = observations[(observations["observationLevel"] == "event") & (observations["classificationMethod"] != "human")]
    labels = [event_label(row.observationType, row.scientificName) for row in events.itertuples()]
    return pd.Series(labels, index=events["eventID"], name="ai")


def truth_from_reviews(package_dir: Path) -> pd.Series:
    """Return the reviewers' label for every reviewed event, by event ID."""
    reviews = load_reviews(package_dir)
    labels = [event_label(row.observationType, row.scientificName) for row in reviews.itertuples()]
    return pd.Series(labels, index=reviews["eventID"], name="truth")


def truth_from_file(truth_file: Path, package_dir: Path) -> pd.Series:
    """Return the true label for every event from a file of image labels.

    The file has a ``filePath`` (or ``mediaID``) column and a ``scientificName``
    column, empty for blank images. An event's label is the first non-blank label
    among its images, so a burst with one image of an animal is an animal event.
    """
    labels = pd.read_csv(truth_file, dtype=str)
    path_column = "filePath" if "filePath" in labels.columns else "mediaID"
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    media_events = observations.loc[observations["observationLevel"] == "media", ["mediaID", "eventID"]]
    media_events = media_events.dropna().drop_duplicates("mediaID")
    known = set(media_events["mediaID"])
    labels["mediaID"] = [match_path(path, known) for path in labels[path_column]]
    labels = labels.dropna(subset=["mediaID"]).merge(media_events, on="mediaID")
    labels["label"] = labels["scientificName"].fillna("").str.strip().replace("", "blank")
    first_animal = labels.sort_values("mediaID").groupby("eventID")["label"].agg(
        lambda values: next((value for value in values if value != "blank"), "blank")
    )
    return first_animal.rename("truth")


def ancestry_from_file(truth_file: Path) -> dict[str, set[str]]:
    """Return the higher taxa (genus, family, order, class) of each true label, from
    optional columns of those names in the truth file."""
    labels = pd.read_csv(truth_file, dtype=str)
    ranks = [rank for rank in RANKS if rank in labels.columns]
    ancestry = {}
    for row in labels.dropna(subset=["scientificName"]).drop_duplicates("scientificName").to_dict("records"):
        ancestry[row["scientificName"]] = {row[rank] for rank in ranks if isinstance(row[rank], str)}
    return ancestry


def ancestry_from_results(package_dir: Path) -> dict[str, set[str]]:
    """Return the higher taxa of every species SpeciesNet mentioned, read from the
    taxonomy in its labels (including runner-up classifications)."""
    results_file = package_dir / RESULTS_FILE
    if not results_file.exists():
        return {}
    ancestry = {}
    for prediction in json.loads(results_file.read_text(encoding="utf-8"))["predictions"]:
        for detection in prediction.get("detections") or []:
            labels = list((detection.get("raw_classifications") or {}).get("classes", []))
            labels += [(detection.get("processed_classification") or {}).get("class", "")]
            for label in labels:
                scientific_name, _ = parse_taxon(label) if label else (None, "")
                parts = label.split(";")
                if scientific_name and len(parts) == 7:
                    ancestry[scientific_name] = {part.capitalize() for part in parts[1:5] if part}
    return ancestry


def outcome(truth: str, ai: str, ancestry: dict[str, set[str]] | None = None) -> str:
    """Return whether the AI's label is correct, correct only above species level
    (e.g. ``Bovidae`` for a duiker), unverified (a higher taxon, but the truth's
    ancestry is unknown), or wrong."""
    if ai == truth:
        return "correct"
    ai_is_higher_taxon = ai not in NON_TAXON_LABELS and " " not in ai
    if not ai_is_higher_taxon or truth in NON_TAXON_LABELS:
        return "wrong"
    known = (ancestry or {}).get(truth)
    if known is None:
        return "unverified"
    return "not to species" if ai in known else "wrong"


def compare(truth: pd.Series, ai: pd.Series, ancestry: dict[str, set[str]] | None = None) -> pd.DataFrame:
    """Return one row per event with both labels and the outcome."""
    compared = pd.concat([truth, ai], axis=1, join="inner").rename_axis("eventID").reset_index()
    compared["outcome"] = [outcome(row.truth, row.ai, ancestry) for row in compared.itertuples()]
    return compared


RIGHT_OUTCOMES = {"correct", "not to species"}


def per_class_metrics(compared: pd.DataFrame) -> pd.DataFrame:
    """Return precision and recall for every label in the truth or the AI's output.

    Precision: of the events the AI gave this label, the share where the label was
    true (for a higher taxon such as ``Bovidae``, the animal was in that taxon).
    Recall: of the events that truly had this label, the share the AI named exactly.
    """
    rows = []
    for label in sorted(set(compared["truth"]) | set(compared["ai"])):
        true = compared["truth"] == label
        predicted = compared["ai"] == label
        right = predicted & compared["outcome"].isin(RIGHT_OUTCOMES)
        found = true & predicted
        rows.append(
            {
                "label": label,
                "events": int(true.sum()),
                "predicted": int(predicted.sum()),
                "right": int(right.sum()),
                "precision": round(right.sum() / predicted.sum(), 2) if predicted.any() else None,
                "recall": round(found.sum() / true.sum(), 2) if true.any() else None,
            }
        )
    return pd.DataFrame(rows).set_index("label")


DEFAULT_TARGET_PRECISION = 0.95
MIN_EVENTS_FOR_THRESHOLD = 10


def thresholds(compared: pd.DataFrame, target_precision: float = DEFAULT_TARGET_PRECISION) -> pd.DataFrame:
    """Return, for every label the AI used, the lowest confidence at which its
    answers reach ``target_precision``, and how many answers that keeps.

    Answers at or above the threshold can be trusted; the rest should be reviewed.
    A threshold is missing (NaN) if no confidence level reaches the target. Labels with
    fewer than ten answers are flagged, because their thresholds are unreliable.
    """
    rows = []
    for label, answers in compared.groupby("ai"):
        right = answers["outcome"].isin(RIGHT_OUTCOMES)
        chosen = None
        for threshold in sorted(answers["confidence"].unique()):
            kept = answers["confidence"] >= threshold
            if right[kept].mean() >= target_precision:
                chosen = threshold
                break
        kept = answers["confidence"] >= chosen if chosen is not None else answers["confidence"] > 1
        rows.append(
            {
                "label": label,
                "answers": len(answers),
                "precision": round(right.mean(), 2),
                "threshold": chosen,
                "kept": int(kept.sum()),
                "precisionKept": round(right[kept].mean(), 2) if kept.any() else None,
                "fewAnswers": len(answers) < MIN_EVENTS_FOR_THRESHOLD,
            }
        )
    return pd.DataFrame(rows).set_index("label")


def confusion_matrix(compared: pd.DataFrame) -> pd.DataFrame:
    """Return a table counting events by true label (rows) and AI label (columns)."""
    return pd.crosstab(compared["truth"], compared["ai"], rownames=["truth"], colnames=["AI"])


def run(package_dir: Path, truth_file: Path | None = None) -> pd.DataFrame:
    """Compare the AI with the reviews (or ``truth_file``), save the per-event
    comparison as ``evaluation.csv`` in the package, and return it."""
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    truth = truth_from_file(truth_file, package_dir) if truth_file else truth_from_reviews(package_dir)
    ancestry = ancestry_from_results(package_dir)
    if truth_file:
        ancestry.update(ancestry_from_file(truth_file))
    compared = compare(truth, machine_labels(observations), ancestry)
    machine = observations[observations["classificationMethod"] != "human"]
    confidence = order_events(machine[machine["observationLevel"] == "event"], machine, "uncertainty")
    compared = compared.merge(confidence[["eventID", "confidence"]], on="eventID", how="left")
    compared.to_csv(package_dir / EVALUATION_FILE, index=False)
    return compared


templates = Environment(loader=PackageLoader("camtraptools"), autoescape=select_autoescape())
THUMBNAIL_WIDTH = 480


def thumbnail(image_path: Path) -> str:
    """Return a small JPEG of the image as a data URI, so the report is one file."""
    with Image.open(image_path) as image:
        image.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_WIDTH))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def failure_report(
    package_dir: Path,
    images_dir: Path,
    compared: pd.DataFrame,
    target_precision: float = DEFAULT_TARGET_PRECISION,
) -> str:
    """Return a self-contained HTML report with the evaluation summary and every
    event the AI did not get exactly right, with its images and boxes."""
    media = pd.read_csv(package_dir / "media.csv", dtype=str).set_index("mediaID")
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    machine_media = observations[(observations["observationLevel"] == "media") & (observations["classificationMethod"] != "human")]
    failures = []
    for row in compared[compared["outcome"] != "correct"].sort_values(["outcome", "confidence"]).itertuples():
        event_rows = machine_media[machine_media["eventID"] == row.eventID]
        images = []
        for media_id, detections in event_rows.groupby("mediaID", sort=False):
            images.append(
                {
                    "name": media.loc[media_id, "fileName"],
                    "timestamp": media.loc[media_id, "timestamp"],
                    "src": thumbnail(images_dir / media.loc[media_id, "filePath"]),
                    "boxes": detections.dropna(subset=["bboxX"]).to_dict("records"),
                }
            )
        failures.append({"event": row, "images": images})
    outcomes = compared["outcome"].value_counts().reindex(OUTCOMES, fill_value=0)
    return templates.get_template("evaluation.html").render(
        package=package_dir.resolve().name,
        generated=f"{datetime.now():%Y-%m-%d %H:%M}",
        version=__version__,
        total=len(compared),
        outcomes=outcomes.to_dict(),
        metrics=per_class_metrics(compared).reset_index().to_dict("records"),
        thresholds=thresholds(compared, target_precision).reset_index().to_dict("records"),
        target_precision=target_precision,
        failures=failures,
    )
