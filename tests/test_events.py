import pandas as pd
import pytest

from camtraptools import events
from camtraptools.detect import OBSERVATION_COLUMNS


def make_media(*rows):
    return pd.DataFrame(rows, columns=["mediaID", "deploymentID", "timestamp"])


def test_assign_events_splits_on_gap_and_deployment():
    media = make_media(
        ("a1", "camera_A", "2024-03-01T06:00:00"),
        ("a2", "camera_A", "2024-03-01T06:00:00"),
        ("a3", "camera_A", "2024-03-01T06:29:00"),
        ("a4", "camera_A", "2024-03-01T07:00:00"),
        ("b1", "camera_B", "2024-03-01T06:10:00"),
    )

    assert list(events.assign_events(media)) == ["camera_A-0001"] * 3 + ["camera_A-0002", "camera_B-0001"]


def test_assign_events_follows_time_not_row_order():
    media = make_media(("late", "camera_A", "2024-03-02T06:00:00"), ("early", "camera_A", "2024-03-01T06:00:00"))

    assert list(events.assign_events(media)) == ["camera_A-0002", "camera_A-0001"]


def test_assign_events_uses_gap_minutes_and_skips_undated_images():
    media = make_media(
        ("a1", "camera_A", "2024-03-01T06:00:00"),
        ("a2", "camera_A", "2024-03-01T06:10:00"),
        ("a3", "camera_A", None),
    )

    event_ids = events.assign_events(media, gap_minutes=5)

    assert list(event_ids[:2]) == ["camera_A-0001", "camera_A-0002"]
    assert pd.isna(event_ids.iloc[2])


def make_observations(*rows):
    columns = ["mediaID", "observationType", "scientificName", "classificationProbability"]
    observations = pd.DataFrame(rows, columns=columns)
    observations["deploymentID"] = "camera_A"
    observations["eventStart"] = observations["eventEnd"] = [f"2024-03-01T06:0{i}:00" for i in range(len(rows))]
    observations["classifiedBy"] = "SpeciesNet 4.0.3a"
    observations["classificationTimestamp"] = "2026-10-19T10:00:00"
    return observations


def test_summarize_event_votes_by_summed_probability():
    observations = make_observations(
        ("m1", "animal", "Oryx gazella", 0.9),
        ("m2", "animal", "Bovidae", 0.7),
        ("m2", "animal", "Bovidae", 0.6),
        ("m3", "blank", None, None),
    )

    event = events.summarize_event("camera_A-0001", observations)

    assert event["observationType"] == "animal"
    assert event["scientificName"] == "Bovidae"
    assert event["classificationProbability"] == round(1.3 / 2.2, 3)
    assert event["count"] == 2
    assert (event["eventStart"], event["eventEnd"]) == ("2024-03-01T06:00:00", "2024-03-01T06:03:00")
    assert event["observationLevel"] == "event"


def test_summarize_event_prefers_animals_over_other_types():
    observations = make_observations(("m1", "human", None, 0.9), ("m2", "animal", None, 0.4))

    event = events.summarize_event("camera_A-0001", observations)

    assert (event["observationType"], event["scientificName"]) == ("animal", None)


@pytest.mark.parametrize(
    ("types", "expected"),
    [(["blank", "human"], "human"), (["blank", "vehicle"], "vehicle"), (["blank", "unknown"], "unknown"), (["blank"], "blank")],
)
def test_summarize_event_type_priority(types, expected):
    observations = make_observations(*[(f"m{i}", kind, None, None) for i, kind in enumerate(types)])

    assert events.summarize_event("camera_A-0001", observations)["observationType"] == expected


def test_build_event_observations_has_camtrap_dp_columns():
    observations = make_observations(("m1", "animal", "Oryx gazella", 0.9), ("m2", "blank", None, None))
    observations["eventID"] = ["camera_A-0001", "camera_A-0002"]

    built = events.build_event_observations(observations)

    assert list(built.columns) == OBSERVATION_COLUMNS
    assert list(built["observationID"]) == ["camera_A-0001-event", "camera_A-0002-event"]
    assert list(built["count"].fillna(0)) == [1, 0]


def test_summarize_events_counts_animal_events_per_taxon():
    built = pd.DataFrame(
        {
            "deploymentID": ["camera_A", "camera_A", "camera_B", "camera_B"],
            "observationType": ["animal", "animal", "animal", "blank"],
            "scientificName": ["Oryx gazella", "Oryx gazella", None, None],
        }
    )

    assert events.summarize_events(built).to_dict() == {
        "camera_A": {"Oryx gazella": 2, "unidentified animal": 0},
        "camera_B": {"Oryx gazella": 0, "unidentified animal": 1},
    }


def test_run_adds_event_ids_and_event_observations(tmp_path):
    make_media(
        ("a1", "camera_A", "2024-03-01T06:00:00"),
        ("a2", "camera_A", "2024-03-01T06:00:00"),
        ("a3", "camera_A", "2024-03-02T06:00:00"),
    ).to_csv(tmp_path / "media.csv", index=False)
    media_observations = pd.DataFrame(columns=OBSERVATION_COLUMNS)
    media_observations["observationID"] = ["obs000001", "obs000002", "obs000003"]
    media_observations["mediaID"] = ["a1", "a2", "a3"]
    media_observations["deploymentID"] = "camera_A"
    media_observations["eventStart"] = media_observations["eventEnd"] = ["2024-03-01T06:00:00"] * 2 + ["2024-03-02T06:00:00"]
    media_observations["observationLevel"] = "media"
    media_observations["observationType"] = ["animal", "animal", "blank"]
    media_observations["scientificName"] = ["Oryx gazella", "Oryx gazella", None]
    media_observations["count"] = pd.array([1, 1, None], dtype="Int64")
    media_observations["classificationProbability"] = [0.9, 0.8, None]
    media_observations.to_csv(tmp_path / "observations.csv", index=False)

    built = events.run(tmp_path)
    events.run(tmp_path)

    saved = pd.read_csv(tmp_path / "observations.csv")
    assert len(built) == 2
    assert list(saved["observationLevel"]) == ["media"] * 3 + ["event"] * 2
    assert list(saved["eventID"]) == ["camera_A-0001", "camera_A-0001", "camera_A-0002", "camera_A-0001", "camera_A-0002"]
    assert (tmp_path / "observations.csv").read_text(encoding="utf-8").splitlines()[1].split(",")[10] == "1"


def test_summarize_event_without_probabilities_counts_images():
    observations = make_observations(
        ("m1", "animal", "Oryx gazella", 0.0),
        ("m2", "animal", "Vulpes chama", 0.0),
        ("m3", "animal", "Vulpes chama", None),
    )

    event = events.summarize_event("camera_A-0001", observations)

    assert event["scientificName"] == "Vulpes chama"
    assert event["classificationProbability"] == round(2 / 3, 3)


def test_build_event_observations_summarizes_many_events_at_once():
    observations = make_observations(
        ("m1", "animal", "Oryx gazella", 0.9),
        ("m2", "human", None, 0.8),
        ("m3", "blank", None, None),
        ("m4", "animal", "Vulpes chama", 0.4),
    )
    observations["eventID"] = ["cam-0002", "cam-0001", "cam-0001", "cam-0003"]

    built = events.build_event_observations(observations).set_index("eventID")

    assert built["observationType"].to_dict() == {"cam-0001": "human", "cam-0002": "animal", "cam-0003": "animal"}
    assert built.loc["cam-0003", "scientificName"] == "Vulpes chama"
    assert built.loc["cam-0001", "count"] == 1
