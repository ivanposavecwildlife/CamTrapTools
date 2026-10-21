"""Human review decisions: stored in ``reviews.csv`` and written to
``observations.csv`` as event-level observations classified by a human.

An event usually has one decision. A visit by several species, such as a mixed herd,
has one row per species: the first is the main one, used wherever an event needs a
single label, and every row becomes an observation."""

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


def main_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Return one review per event: the main species of visits with several."""
    return reviews.drop_duplicates("eventID")


def human_observations(reviews: pd.DataFrame, machine_events: pd.DataFrame) -> pd.DataFrame:
    """Return one human event-level observation per review row of an existing event:
    ``EVENT-human`` for the main species and ``EVENT-human-2`` and so on for others."""
    times = machine_events[["eventID", "deploymentID", "eventStart", "eventEnd"]].drop_duplicates("eventID")
    reviewed = reviews.merge(times, on="eventID")
    number = reviewed.groupby("eventID").cumcount() + 1
    suffix = number.map(lambda value: "" if value == 1 else f"-{value}")
    observations = pd.DataFrame(
        {
            "observationID": reviewed["eventID"] + "-human" + suffix,
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
    reviews. Returns how many reviewed events were applied and how many refer to
    events that no longer exist (for example after changing the event gap)."""
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    machine = observations[observations["classificationMethod"] != "human"]
    reviews = load_reviews(package_dir)
    humans = human_observations(reviews, machine[machine["observationLevel"] == "event"])
    combined = pd.concat([machine, humans], ignore_index=True)
    combined[OBSERVATION_COLUMNS].to_csv(package_dir / OBSERVATIONS_FILE, index=False)
    applied = humans["eventID"].nunique()
    return applied, reviews["eventID"].nunique() - applied


def save_review(
    package_dir: Path,
    event_id: str,
    observation_type: str,
    scientific_name: str | None,
    count: int | None,
    reviewer: str,
    others: list[tuple[str, int]] | None = None,
) -> None:
    """Record a review decision for one event, replacing any earlier one. ``others``
    adds more species seen in the same visit, each with its count, after the main
    one; only animal decisions can have them."""
    validate_review(observation_type, scientific_name, count)
    others = others or []
    if others and observation_type != "animal":
        raise ValueError("only animal observations can have more than one species")
    names = [scientific_name, *(name for name, _ in others)]
    if others and (not all(names) or len(set(names)) < len(names)):
        raise ValueError("each species of a visit needs its own, different scientific name")
    for name, other_count in others:
        validate_review("animal", name, other_count)
    reviewed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    decisions = [(observation_type, scientific_name, count), *(("animal", name, other_count) for name, other_count in others)]
    rows = [
        {
            "eventID": event_id,
            "observationType": kind,
            "scientificName": name or None,
            "count": str(number) if number is not None else None,
            "reviewer": reviewer,
            "reviewedAt": reviewed_at,
        }
        for kind, name, number in decisions
    ]
    reviews = load_reviews(package_dir)
    reviews = pd.concat([reviews[reviews["eventID"] != event_id], pd.DataFrame(rows)], ignore_index=True)
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


ORDERS = ["uncertainty", "plan", "time"]


def event_confidence(event: pd.Series, media_observations: pd.DataFrame) -> float:
    """Return how confident the AI was about an event, from 0 to 1.

    It is the event's vote share times the best classifier probability for the
    chosen label. Unknown events score 0, and animals not identified to species
    are capped at 0.5, so the events most likely to need a human come first.
    """
    if event["observationType"] in ("unknown", "unclassified"):
        return 0.0
    matching = media_observations[media_observations["observationType"] == event["observationType"]]
    if event["observationType"] == "animal":
        matching = matching[matching["scientificName"].fillna("") == (event["scientificName"] if isinstance(event["scientificName"], str) else "")]
    probabilities = pd.to_numeric(matching["classificationProbability"]).dropna()
    best = probabilities.max() if not probabilities.empty else (1.0 if event["observationType"] == "blank" else 0.5)
    share = pd.to_numeric(pd.Series([event["classificationProbability"]])).fillna(1.0).iloc[0]
    confidence = float(share * best)
    if event["observationType"] == "animal" and " " not in str(event["scientificName"] if isinstance(event["scientificName"], str) else ""):
        confidence = min(confidence, 0.5)
    return round(confidence, 3)


def order_events(events: pd.DataFrame, observations: pd.DataFrame, order: str) -> pd.DataFrame:
    """Return the events with a ``confidence`` column, sorted for review: by start
    time (``time``), otherwise least confident first (the review app puts the events
    of the review plan first for ``plan``)."""
    media_level = observations[observations["observationLevel"] == "media"]
    by_event = dict(list(media_level.groupby("eventID")))
    empty = media_level.iloc[0:0]
    events = events.assign(
        confidence=[event_confidence(event, by_event.get(event["eventID"], empty)) for _, event in events.iterrows()]
    )
    columns = ["eventStart", "eventID"] if order == "time" else ["confidence", "eventStart", "eventID"]
    return events.sort_values(columns).reset_index(drop=True)


CONFIDENCE_BANDS = [(0.0, 0.6, "below 0.6"), (0.6, 0.95, "0.6 to 0.95"), (0.95, 1.01, "0.95 and up")]
BREAK_SECONDS = 600


def compare_with_machine(package_dir: Path) -> pd.DataFrame:
    """Return one row per reviewed event with the AI's decision and confidence, the
    human decision, and the outcome: agreed, changed type, changed species or
    changed count."""
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    machine = observations[observations["classificationMethod"] != "human"]
    events = order_events(machine[machine["observationLevel"] == "event"], machine, "uncertainty")
    events = events[["eventID", "observationType", "scientificName", "count", "confidence"]]
    compared = main_reviews(load_reviews(package_dir)).merge(events, on="eventID", suffixes=("", "AI"))
    same_type = compared["observationType"] == compared["observationTypeAI"]
    same_name = compared["scientificName"].fillna("") == compared["scientificNameAI"].fillna("")
    same_count = compared["count"].fillna("") == compared["countAI"].fillna("")
    compared["outcome"] = "agreed"
    compared.loc[~same_count, "outcome"] = "changed count"
    compared.loc[~same_name, "outcome"] = "changed species"
    compared.loc[~same_type, "outcome"] = "changed type"
    return compared


def review_speed(reviewed_at: pd.Series) -> dict:
    """Return the number of decisions, minutes spent and decisions per minute.

    Pauses longer than ten minutes between decisions count as breaks.
    """
    times = pd.to_datetime(reviewed_at, format="ISO8601", utc=True).sort_values()
    gaps = times.diff().dt.total_seconds().dropna()
    active = gaps[gaps <= BREAK_SECONDS]
    minutes = float(active.sum()) / 60
    return {
        "decisions": len(times),
        "minutes": round(minutes, 1),
        "perMinute": round(len(active) / minutes, 1) if minutes else None,
    }


def summarize_by_confidence(compared: pd.DataFrame) -> pd.DataFrame:
    """Return, per AI confidence band, how many events were reviewed and how many
    the reviewer changed."""
    rows = []
    for low, high, label in CONFIDENCE_BANDS:
        band = compared[(compared["confidence"] >= low) & (compared["confidence"] < high)]
        rows.append({"confidence": label, "reviewed": len(band), "changed": int((band["outcome"] != "agreed").sum())})
    return pd.DataFrame(rows).set_index("confidence")
