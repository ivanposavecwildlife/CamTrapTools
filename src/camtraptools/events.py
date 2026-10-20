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


def summarize_all(observations: pd.DataFrame) -> pd.DataFrame:
    """Return one event-level observation per event in the media-level
    ``observations``, computed for all events at once.

    The event type is the most important type seen (animal before human, vehicle,
    unknown and blank). For animals, the species with the highest summed classifier
    probability wins (or the most images, if no image has a probability), and the
    count is the largest number of animals in any single image.
    """
    rows = observations.dropna(subset=["eventID"]).copy()
    priority = {name: rank for rank, name in enumerate(TYPE_PRIORITY)}
    rows["rank"] = rows["observationType"].map(priority).fillna(priority["unknown"])
    rows["eventRank"] = rows.groupby("eventID")["rank"].transform("min")
    first = rows.drop_duplicates("eventID").set_index("eventID")
    events = pd.DataFrame(index=first.index.sort_values())
    events["observationID"] = events.index + "-event"
    events["deploymentID"] = first["deploymentID"]
    events["eventID"] = events.index
    events["eventStart"] = rows.groupby("eventID")["eventStart"].min()
    events["eventEnd"] = rows.groupby("eventID")["eventEnd"].max()
    events["observationLevel"] = "event"
    events["observationType"] = rows.groupby("eventID")["eventRank"].first().astype(int).map(dict(enumerate(TYPE_PRIORITY)))
    events["classificationMethod"] = "machine"
    events["classifiedBy"] = first["classifiedBy"]
    events["classificationTimestamp"] = first["classificationTimestamp"]

    matching = rows[rows["rank"] == rows["eventRank"]]
    counted = matching[matching["observationType"].isin(["animal", "human", "vehicle"])]
    events["count"] = counted.groupby(["eventID", "mediaID"]).size().groupby(level=0).max()

    animals = matching[matching["observationType"] == "animal"]
    if not animals.empty:
        keys = [animals["eventID"], animals["scientificName"].fillna("")]
        votes = pd.to_numeric(animals["classificationProbability"], errors="coerce").fillna(0).groupby(keys).sum()
        totals = votes.groupby(level=0).transform("sum")
        images = animals.groupby(keys).size()
        votes = votes.where(totals > 0, images)
        winners = votes.groupby(level=0).idxmax()
        events["scientificName"] = winners.map(lambda key: key[1] or None)
        events["classificationProbability"] = (votes.groupby(level=0).max() / votes.groupby(level=0).sum()).round(3)
    events = events.reset_index(drop=True).reindex(columns=OBSERVATION_COLUMNS)
    events["count"] = events["count"].astype("Int64")
    return events


def summarize_event(event_id: str, observations: pd.DataFrame) -> dict:
    """Return the event-level observation for one event's media-level observations
    (see ``summarize_all``), with missing values as ``None``."""
    event = summarize_all(observations.assign(eventID=event_id)).iloc[0].to_dict()
    return {key: None if pd.isna(value) else value for key, value in event.items()}


def build_event_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Return one event-level observation per event in the media-level ``observations``."""
    return summarize_all(observations)


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
