"""Group images into independent events and summarize them per species."""

from pathlib import Path

import pandas as pd

from camtraptools.detect import OBSERVATION_COLUMNS, OBSERVATIONS_FILE
from camtraptools.reviews import apply_reviews

DEFAULT_GAP_MINUTES = 30
TYPE_PRIORITY = ["animal", "human", "vehicle", "unknown", "unclassified", "blank"]


def assign_events(media: pd.DataFrame, gap_minutes: float = DEFAULT_GAP_MINUTES) -> pd.Series:
    """Return an event ID for every image. Consecutive images at the same deployment
    belong to one event while each follows the previous one by at most
    ``gap_minutes``. Images without a timestamp get no event."""
    dated = media.assign(time=pd.to_datetime(media["timestamp"])).dropna(subset=["time"])
    dated = dated.sort_values(["deploymentID", "time"])
    gap = dated.groupby("deploymentID")["time"].diff()
    starts_event = gap.isna() | (gap > pd.Timedelta(minutes=gap_minutes))
    number = starts_event.groupby(dated["deploymentID"]).cumsum()
    event_ids = dated["deploymentID"] + "-" + number.map("{:04d}".format)
    return event_ids.reindex(media.index)


def summarize_event(event_id: str, observations: pd.DataFrame) -> dict:
    """Return the event-level observation for one event's media-level observations.

    The event type is the most important type seen (animal before human, vehicle,
    unknown and blank). For animals, the species with the highest summed
    classifier probability wins, and the count is the largest number of animals
    in any single image.
    """
    observation_type = next(t for t in TYPE_PRIORITY if (observations["observationType"] == t).any())
    event = {
        "observationID": f"{event_id}-event",
        "deploymentID": observations["deploymentID"].iloc[0],
        "eventID": event_id,
        "eventStart": observations["eventStart"].min(),
        "eventEnd": observations["eventEnd"].max(),
        "observationLevel": "event",
        "observationType": observation_type,
        "classificationMethod": "machine",
        "classifiedBy": observations["classifiedBy"].iloc[0],
        "classificationTimestamp": observations["classificationTimestamp"].iloc[0],
    }
    matching = observations[observations["observationType"] == observation_type]
    if observation_type in ("animal", "human", "vehicle"):
        event["count"] = matching.groupby("mediaID").size().max()
    if observation_type == "animal":
        probability = pd.to_numeric(matching["classificationProbability"])
        votes = probability.groupby(matching["scientificName"].fillna("")).sum()
        event["scientificName"] = votes.idxmax() or None
        event["classificationProbability"] = round(votes.max() / votes.sum(), 3)
    return event


def build_event_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Return one event-level observation per event in the media-level ``observations``."""
    with_event = observations.dropna(subset=["eventID"])
    rows = [summarize_event(event_id, group) for event_id, group in with_event.groupby("eventID")]
    events = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
    events["count"] = events["count"].astype("Int64")
    return events


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return the number of independent animal events per taxon and deployment."""
    animals = events[events["observationType"] == "animal"]
    taxa = animals["scientificName"].fillna("unidentified animal")
    return animals.groupby([taxa, animals["deploymentID"]]).size().unstack(fill_value=0)


def run(package_dir: Path, gap_minutes: float = DEFAULT_GAP_MINUTES) -> pd.DataFrame:
    """Add event IDs to the media-level observations in the package, append one
    event-level observation per event to ``observations.csv``, re-apply human
    reviews, and return the machine event-level observations."""
    media = pd.read_csv(package_dir / "media.csv", dtype=str)
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    observations = observations[observations["observationLevel"] == "media"].copy()
    event_ids = dict(zip(media["mediaID"], assign_events(media, gap_minutes), strict=True))
    observations["eventID"] = observations["mediaID"].map(event_ids)
    events = build_event_observations(observations)
    combined = pd.concat([observations, events], ignore_index=True)
    combined[OBSERVATION_COLUMNS].to_csv(package_dir / OBSERVATIONS_FILE, index=False)
    apply_reviews(package_dir)
    return events
