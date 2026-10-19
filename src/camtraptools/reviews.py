"""Human review decisions: stored in ``reviews.csv`` and written to
``observations.csv`` as event-level observations classified by a human."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from camtraptools.detect import OBSERVATION_COLUMNS, OBSERVATIONS_FILE

REVIEWS_FILE = "reviews.csv"
REVIEW_COLUMNS = ["eventID", "observationType", "scientificName", "count", "reviewer", "reviewedAt"]
REVIEW_TYPES = ["animal", "blank", "human", "vehicle", "unknown"]
COUNTED_TYPES = {"animal", "human", "vehicle"}


def load_reviews(package_dir: Path) -> pd.DataFrame:
    """Return the package's review decisions, one row per reviewed event."""
    path = package_dir / REVIEWS_FILE
    if not path.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    return pd.read_csv(path, dtype=str)


def validate_review(observation_type: str, scientific_name: str | None, count: int | None) -> None:
    """Raise ValueError if a review decision is incomplete or contradictory."""
    if observation_type not in REVIEW_TYPES:
        raise ValueError(f"type must be one of: {', '.join(REVIEW_TYPES)}")
    if observation_type in COUNTED_TYPES and (count is None or count < 1):
        raise ValueError(f"a {observation_type} observation needs a count of at least 1")
    if observation_type not in COUNTED_TYPES and count is not None:
        raise ValueError(f"a {observation_type} observation has no count")
    if observation_type != "animal" and scientific_name:
        raise ValueError("only animal observations have a scientific name")


def human_observations(reviews: pd.DataFrame, machine_events: pd.DataFrame) -> pd.DataFrame:
    """Return one human event-level observation per review of an existing event."""
    times = machine_events[["eventID", "deploymentID", "eventStart", "eventEnd"]].drop_duplicates("eventID")
    reviewed = reviews.merge(times, on="eventID")
    observations = pd.DataFrame(
        {
            "observationID": reviewed["eventID"] + "-human",
            "deploymentID": reviewed["deploymentID"],
            "eventID": reviewed["eventID"],
            "eventStart": reviewed["eventStart"],
            "eventEnd": reviewed["eventEnd"],
            "observationLevel": "event",
            "observationType": reviewed["observationType"],
            "scientificName": reviewed["scientificName"],
            "count": reviewed["count"],
            "classificationMethod": "human",
            "classifiedBy": reviewed["reviewer"],
            "classificationTimestamp": reviewed["reviewedAt"],
        }
    )
    return observations.reindex(columns=OBSERVATION_COLUMNS)


def apply_reviews(package_dir: Path) -> tuple[int, int]:
    """Replace the human observations in ``observations.csv`` with the current
    reviews. Returns how many were applied and how many refer to events that no
    longer exist (for example after changing the event gap)."""
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    machine = observations[observations["classificationMethod"] != "human"]
    reviews = load_reviews(package_dir)
    humans = human_observations(reviews, machine[machine["observationLevel"] == "event"])
    combined = pd.concat([machine, humans], ignore_index=True)
    combined[OBSERVATION_COLUMNS].to_csv(package_dir / OBSERVATIONS_FILE, index=False)
    return len(humans), len(reviews) - len(humans)


def save_review(
    package_dir: Path,
    event_id: str,
    observation_type: str,
    scientific_name: str | None,
    count: int | None,
    reviewer: str,
) -> None:
    """Record a review decision for one event, replacing any earlier one."""
    validate_review(observation_type, scientific_name, count)
    review = {
        "eventID": event_id,
        "observationType": observation_type,
        "scientificName": scientific_name or None,
        "count": str(count) if count is not None else None,
        "reviewer": reviewer,
        "reviewedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    reviews = load_reviews(package_dir)
    reviews = pd.concat([reviews[reviews["eventID"] != event_id], pd.DataFrame([review])], ignore_index=True)
    reviews[REVIEW_COLUMNS].to_csv(package_dir / REVIEWS_FILE, index=False)
    apply_reviews(package_dir)


def prefer_human(observations: pd.DataFrame) -> pd.DataFrame:
    """Return the observations without machine event-level rows for events that a
    human has reviewed, so each event is counted once."""
    human_events = set(observations.loc[observations["classificationMethod"] == "human", "eventID"])
    replaced = (
        (observations["observationLevel"] == "event")
        & (observations["classificationMethod"] != "human")
        & observations["eventID"].isin(human_events)
    )
    return observations[~replaced]
