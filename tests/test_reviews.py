import pandas as pd
import pytest

from camtraptools import events, reviews
from camtraptools.detect import OBSERVATION_COLUMNS


@pytest.mark.parametrize(
    ("observation_type", "scientific_name", "count"),
    [("animal", "Canis mesomelas", 1), ("animal", None, 2), ("human", None, 1), ("blank", None, None), ("unknown", None, None)],
)
def test_validate_review_accepts_complete_decisions(observation_type, scientific_name, count):
    reviews.validate_review(observation_type, scientific_name, count)


@pytest.mark.parametrize(
    ("observation_type", "scientific_name", "count", "message"),
    [
        ("bird", None, 1, "type must be one of"),
        ("animal", "Canis mesomelas", None, "needs a count of at least 1"),
        ("vehicle", None, 0, "needs a count of at least 1"),
        ("blank", None, 1, "has no count"),
        ("human", "Homo sapiens", 1, "only animal observations have a scientific name"),
    ],
)
def test_validate_review_rejects_incomplete_or_contradictory_decisions(observation_type, scientific_name, count, message):
    with pytest.raises(ValueError, match=message):
        reviews.validate_review(observation_type, scientific_name, count)


def make_package(tmp_path):
    pd.DataFrame(
        {
            "mediaID": ["m1", "m2", "m3"],
            "deploymentID": ["camera_A", "camera_A", "camera_A"],
            "timestamp": ["2024-03-01T06:00:00", "2024-03-01T06:00:01", "2024-03-02T06:00:00"],
        }
    ).to_csv(tmp_path / "media.csv", index=False)
    observations = pd.DataFrame(columns=OBSERVATION_COLUMNS).astype(object)
    observations["observationID"] = ["obs1", "obs2", "obs3"]
    observations["deploymentID"] = "camera_A"
    observations["mediaID"] = ["m1", "m2", "m3"]
    observations["eventStart"] = observations["eventEnd"] = ["2024-03-01T06:00:00", "2024-03-01T06:00:01", "2024-03-02T06:00:00"]
    observations["observationLevel"] = "media"
    observations["observationType"] = ["animal", "animal", "blank"]
    observations["scientificName"] = ["Chlorocebus pygerythrus", "Mammalia", None]
    observations["count"] = ["1", "1", None]
    observations["classificationMethod"] = "machine"
    observations["classificationProbability"] = ["0.94", "0.67", None]
    observations.to_csv(tmp_path / "observations.csv", index=False)
    events.run(tmp_path)
    return tmp_path


def test_save_review_records_decision_and_human_observation(tmp_path):
    package = make_package(tmp_path)

    reviews.save_review(package, "camera_A-0001", "animal", "Canis mesomelas", 1, "Ivan Posavec")

    review = reviews.load_reviews(package).iloc[0]
    assert (review["eventID"], review["scientificName"], review["count"], review["reviewer"]) == (
        "camera_A-0001",
        "Canis mesomelas",
        "1",
        "Ivan Posavec",
    )
    assert review["reviewedAt"][-6] in "+-"
    observations = pd.read_csv(package / "observations.csv", dtype=str)
    human = observations[observations["classificationMethod"] == "human"].iloc[0]
    assert (human["observationID"], human["observationLevel"], human["classifiedBy"]) == (
        "camera_A-0001-human",
        "event",
        "Ivan Posavec",
    )
    assert (human["eventStart"], human["eventEnd"]) == ("2024-03-01T06:00:00", "2024-03-01T06:00:01")
    machine_event = observations[observations["observationID"] == "camera_A-0001-event"].iloc[0]
    assert machine_event["scientificName"] == "Chlorocebus pygerythrus"


def test_save_review_replaces_earlier_decision(tmp_path):
    package = make_package(tmp_path)
    reviews.save_review(package, "camera_A-0001", "animal", "Canis mesomelas", 1, "Ivan Posavec")

    reviews.save_review(package, "camera_A-0001", "animal", "Canis mesomelas", 2, "Ana")

    assert list(reviews.load_reviews(package)["count"]) == ["2"]
    observations = pd.read_csv(package / "observations.csv", dtype=str)
    assert list(observations.loc[observations["classificationMethod"] == "human", "classifiedBy"]) == ["Ana"]


def test_rerunning_events_keeps_reviews_and_reports_orphans(tmp_path):
    package = make_package(tmp_path)
    reviews.save_review(package, "camera_A-0002", "blank", None, None, "Ivan Posavec")

    events.run(package)
    assert reviews.apply_reviews(package) == (1, 0)

    events.run(package, gap_minutes=60 * 48)
    assert reviews.apply_reviews(package) == (0, 1)


def test_prefer_human_replaces_machine_event_rows_only_for_reviewed_events(tmp_path):
    package = make_package(tmp_path)
    reviews.save_review(package, "camera_A-0001", "animal", "Canis mesomelas", 1, "Ivan Posavec")

    preferred = reviews.prefer_human(pd.read_csv(package / "observations.csv", dtype=str))

    event_rows = preferred[preferred["observationLevel"] == "event"]
    assert list(event_rows["observationID"]) == ["camera_A-0002-event", "camera_A-0001-human"]
    assert (preferred["observationLevel"] == "media").sum() == 3


def test_compare_with_machine_classifies_each_outcome(tmp_path):
    package = make_package(tmp_path)
    reviews.save_review(package, "camera_A-0001", "animal", "Canis mesomelas", 1, "Ivan Posavec")
    reviews.save_review(package, "camera_A-0002", "blank", None, None, "Ivan Posavec")

    compared = reviews.compare_with_machine(package).set_index("eventID")

    assert compared.loc["camera_A-0001", "outcome"] == "changed species"
    assert compared.loc["camera_A-0001", "scientificNameAI"] == "Chlorocebus pygerythrus"
    assert compared.loc["camera_A-0002", "outcome"] == "agreed"
    assert 0 <= compared.loc["camera_A-0001", "confidence"] <= 1


@pytest.mark.parametrize(
    ("observation_type", "scientific_name", "count", "outcome"),
    [
        ("animal", "Chlorocebus pygerythrus", 1, "agreed"),
        ("animal", "Chlorocebus pygerythrus", 2, "changed count"),
        ("animal", "Canis mesomelas", 2, "changed species"),
        ("blank", None, None, "changed type"),
    ],
)
def test_compare_with_machine_outcomes(tmp_path, observation_type, scientific_name, count, outcome):
    package = make_package(tmp_path)
    reviews.save_review(package, "camera_A-0001", observation_type, scientific_name, count, "Ivan Posavec")

    assert reviews.compare_with_machine(package).iloc[0]["outcome"] == outcome


def test_review_speed_ignores_breaks():
    times = pd.Series(
        ["2026-10-19T20:00:00+02:00", "2026-10-19T20:00:05+02:00", "2026-10-19T20:00:15+02:00", "2026-10-19T21:00:00+02:00"]
    )

    assert reviews.review_speed(times) == {"decisions": 4, "minutes": 0.2, "perMinute": 8.0}


def test_review_speed_with_one_decision():
    assert reviews.review_speed(pd.Series(["2026-10-19T20:00:00+02:00"])) == {"decisions": 1, "minutes": 0.0, "perMinute": None}


def test_summarize_by_confidence_counts_changes_per_band():
    compared = pd.DataFrame({"confidence": [0.3, 0.5, 0.7, 0.99, 1.0], "outcome": ["changed species", "agreed", "agreed", "agreed", "changed type"]})

    summary = reviews.summarize_by_confidence(compared)

    assert summary.to_dict("index") == {
        "below 0.6": {"reviewed": 2, "changed": 1},
        "0.6 to 0.95": {"reviewed": 1, "changed": 0},
        "0.95 and up": {"reviewed": 2, "changed": 1},
    }
