import pandas as pd
import pytest

from camtraptools import compare, reviews
from camtraptools.detect import OBSERVATION_COLUMNS


def test_activity_difference_compares_each_species_own_curve():
    night = pd.DataFrame({"scientificName": "Canis mesomelas", "eventStart": [f"2024-03-01T{h:02d}:00:00" for h in [22, 23, 0, 1, 2] * 4]})
    day = night.assign(eventStart=[f"2024-03-01T{h:02d}:00:00" for h in [10, 11, 12, 13, 14] * 4])

    assert compare.activity_difference(night, night, min_events=20) == pytest.approx(0, abs=0.01)
    assert compare.activity_difference(night, day, min_events=20) > 0.8
    assert compare.activity_difference(night, day, min_events=21) is None


def test_species_sites_ignores_higher_taxa():
    events = pd.DataFrame({"scientificName": ["Vulpes chama", "Bovidae", None], "site": ["a", "b", "c"]})

    assert compare.species_sites(events) == {("Vulpes chama", "a")}


@pytest.mark.parametrize(
    ("names", "expected"),
    [(["machine", "reviewed", "truth"], "truth"), (["machine", "reviewed"], "reviewed"), (["machine", "x"], "machine")],
)
def test_reference_name(names, expected):
    assert compare.reference_name(names) == expected


def make_package(tmp_path, name="package"):
    """Three cameras; the machine calls a Cape fox a jackal at camera c."""
    package = tmp_path / name
    package.mkdir()
    pd.DataFrame(
        {
            "deploymentID": ["a", "b", "c"],
            "deploymentStart": "2024-03-01T12:00:00",
            "deploymentEnd": "2024-03-20T12:00:00",
        }
    ).to_csv(package / "deployments.csv", index=False)
    labels = ["Vulpes chama", "Vulpes chama", "Canis mesomelas", None]
    observations = pd.DataFrame(columns=OBSERVATION_COLUMNS).astype(object)
    media = [f"{camera}/{number}.JPG" for camera, number in [("a", 1), ("b", 1), ("c", 1), ("c", 2)]]
    event_ids = ["a-0001", "b-0001", "c-0001", "c-0002"]
    rows = []
    for media_id, event_id, label in zip(media, event_ids, labels, strict=True):
        common = {
            "deploymentID": event_id[0],
            "eventID": event_id,
            "eventStart": "2024-03-05T22:00:00",
            "observationType": "animal" if label else "blank",
            "scientificName": label,
            "classificationMethod": "machine",
            "classificationProbability": "0.9",
        }
        rows.append({**common, "observationID": f"m-{event_id}", "mediaID": media_id, "observationLevel": "media"})
        rows.append({**common, "observationID": f"e-{event_id}", "observationLevel": "event"})
    pd.concat([observations, pd.DataFrame(rows)]).to_csv(package / "observations.csv", index=False)
    return package


def write_truth(tmp_path):
    truth = tmp_path / "truth.csv"
    truth.write_text(
        "filePath,scientificName,genus,family,order,class\n"
        "a/1.JPG,Vulpes chama,Vulpes,Canidae,Carnivora,Mammalia\n"
        "b/1.JPG,,,,,\n"
        "c/1.JPG,Vulpes chama,Vulpes,Canidae,Carnivora,Mammalia\n"
        "c/2.JPG,Felis silvestris,Felis,Felidae,Carnivora,Mammalia\n"
    )
    return truth


def test_truth_observations_relabel_events(tmp_path):
    package = make_package(tmp_path)

    relabelled = compare.truth_observations(package, write_truth(tmp_path)).set_index("eventID")

    assert relabelled.loc["a-0001", "scientificName"] == "Vulpes chama"
    assert relabelled.loc["b-0001", "observationType"] == "blank"
    assert pd.isna(relabelled.loc["b-0001", "scientificName"])
    assert relabelled.loc["c-0002", ["observationType", "scientificName"]].tolist() == ["animal", "Felis silvestris"]


def test_sources_include_reviews_truth_and_other_packages(tmp_path):
    package = make_package(tmp_path)
    assert list(compare.sources(package)) == ["machine"]

    reviews.save_review(package, "c-0001", "animal", "Vulpes chama", 1, "Tester")
    other = make_package(tmp_path, "other")

    found = compare.sources(package, write_truth(tmp_path), {"adapted": other})

    assert list(found) == ["machine", "reviewed", "truth", "adapted"]
    reviewed = found["reviewed"][0].set_index("eventID")
    assert reviewed.loc["c-0001", "scientificName"] == "Vulpes chama"


def test_run_scores_sources_against_truth(tmp_path):
    package = make_package(tmp_path)

    reference, comparison, _, scores = compare.run(package, tmp_path / "out", write_truth(tmp_path))

    assert reference == "truth"
    machine = scores.loc["machine"]
    assert (machine["speciesMissed"], machine["speciesAdded"]) == (1, 1)
    assert (machine["siteRecordsMissed"], machine["siteRecordsAdded"]) == (2, 2)
    assert machine["naiveOccupancyError"] == pytest.approx(0.165)
    assert set(comparison["source"]) == {"machine", "truth"}
    for name in [compare.SCORES_FILE, compare.COMPARISON_FILE, compare.OVERLAP_COMPARISON_FILE]:
        assert (tmp_path / "out" / name).exists()
    assert (tmp_path / "out" / "truth" / "occupancy.csv").exists()
