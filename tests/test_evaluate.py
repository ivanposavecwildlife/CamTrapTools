import json

import pandas as pd
import pytest

from camtraptools import evaluate, reviews
from camtraptools.detect import OBSERVATION_COLUMNS


@pytest.mark.parametrize(
    ("observation_type", "scientific_name", "expected"),
    [("animal", "Oryx gazella", "Oryx gazella"), ("animal", None, "animal"), ("blank", None, "blank"), ("human", None, "human")],
)
def test_event_label(observation_type, scientific_name, expected):
    assert evaluate.event_label(observation_type, scientific_name) == expected


ANCESTRY = {"Sylvicapra grimmia": {"Sylvicapra", "Bovidae", "Artiodactyla", "Mammalia"}, "Vulpes chama": {"Vulpes", "Canidae", "Carnivora", "Mammalia"}}


@pytest.mark.parametrize(
    ("truth", "ai", "expected"),
    [
        ("Oryx gazella", "Oryx gazella", "correct"),
        ("blank", "blank", "correct"),
        ("Sylvicapra grimmia", "Bovidae", "not to species"),
        ("Vulpes chama", "Bovidae", "wrong"),
        ("Parahyaena brunnea", "Hyaenidae", "unverified"),
        ("Canis mesomelas", "Chlorocebus pygerythrus", "wrong"),
        ("blank", "Bovidae", "wrong"),
        ("Oryx gazella", "blank", "wrong"),
        ("Oryx gazella", "animal", "wrong"),
    ],
)
def test_outcome(truth, ai, expected):
    assert evaluate.outcome(truth, ai, ANCESTRY) == expected


def test_compare_keeps_events_present_in_both():
    truth = pd.Series({"e1": "Oryx gazella", "e2": "blank", "e3": "Vulpes chama"}, name="truth")
    ai = pd.Series({"e1": "Oryx gazella", "e2": "Bovidae", "e4": "blank"}, name="ai")

    compared = evaluate.compare(truth, ai, ANCESTRY)

    assert list(compared["eventID"]) == ["e1", "e2"]
    assert list(compared["outcome"]) == ["correct", "wrong"]


def test_per_class_metrics():
    compared = pd.DataFrame(
        {
            "truth": ["Oryx gazella", "Oryx gazella", "Oryx gazella", "blank", "blank"],
            "ai": ["Oryx gazella", "Oryx gazella", "Bovidae", "blank", "Oryx gazella"],
        }
    )

    metrics = evaluate.per_class_metrics(compared)

    assert metrics.loc["Oryx gazella"].to_dict() == {"events": 3, "predicted": 3, "correct": 2, "precision": 0.67, "recall": 0.67}
    assert metrics.loc["blank"].to_dict() == {"events": 2, "predicted": 1, "correct": 1, "precision": 1.0, "recall": 0.5}
    assert pd.isna(metrics.loc["Bovidae", "recall"])


def test_confusion_matrix_counts_true_against_predicted():
    compared = pd.DataFrame({"truth": ["Oryx gazella", "Oryx gazella", "blank"], "ai": ["Oryx gazella", "Bovidae", "blank"]})

    matrix = evaluate.confusion_matrix(compared)

    assert matrix.loc["Oryx gazella", "Bovidae"] == 1
    assert matrix.loc["Oryx gazella", "Oryx gazella"] == 1
    assert matrix.loc["blank", "blank"] == 1


def make_package(tmp_path):
    observations = pd.DataFrame(columns=OBSERVATION_COLUMNS).astype(object)
    observations["observationID"] = ["o1", "o2", "o3", "o4", "e1", "e2", "e3"]
    observations["mediaID"] = ["cam/a.JPG", "cam/b.JPG", "cam/c.JPG", "cam/d.JPG", None, None, None]
    observations["eventID"] = ["cam-0001", "cam-0001", "cam-0002", "cam-0003", "cam-0001", "cam-0002", "cam-0003"]
    observations["observationLevel"] = ["media"] * 4 + ["event"] * 3
    observations["observationType"] = ["blank", "animal", "animal", "blank", "animal", "animal", "blank"]
    observations["scientificName"] = [None, "Bovidae", "Oryx gazella", None, "Bovidae", "Oryx gazella", None]
    observations["classificationMethod"] = "machine"
    observations.to_csv(tmp_path / "observations.csv", index=False)
    return tmp_path


def test_truth_from_file_takes_first_animal_in_each_event(tmp_path):
    package = make_package(tmp_path)
    truth_file = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "filePath": ["survey/cam/a.JPG", "survey/cam/b.JPG", "survey/cam/c.JPG", "survey/cam/d.JPG", "other/x.JPG"],
            "scientificName": [None, "Sylvicapra grimmia", "Oryx gazella", None, "Vulpes chama"],
        }
    ).to_csv(truth_file, index=False)

    truth = evaluate.truth_from_file(truth_file, package)

    assert truth.to_dict() == {"cam-0001": "Sylvicapra grimmia", "cam-0002": "Oryx gazella", "cam-0003": "blank"}


def test_ancestry_from_file_and_results(tmp_path):
    truth_file = tmp_path / "truth.csv"
    pd.DataFrame(
        {"filePath": ["a.JPG"], "scientificName": ["Vulpes chama"], "genus": ["Vulpes"], "family": ["Canidae"], "order": ["Carnivora"], "class": ["Mammalia"]}
    ).to_csv(truth_file, index=False)
    label = "uuid;mammalia;artiodactyla;bovidae;sylvicapra;grimmia;common duiker"
    results = {"predictions": [{"filepath": "a.JPG", "detections": [{"raw_classifications": {"classes": [label]}}]}]}
    (tmp_path / "speciesnet.json").write_text(json.dumps(results), encoding="utf-8")

    assert evaluate.ancestry_from_file(truth_file) == {"Vulpes chama": {"Vulpes", "Canidae", "Carnivora", "Mammalia"}}
    assert evaluate.ancestry_from_results(tmp_path) == {"Sylvicapra grimmia": {"Mammalia", "Artiodactyla", "Bovidae", "Sylvicapra"}}


def test_run_with_truth_file_writes_evaluation(tmp_path):
    package = make_package(tmp_path)
    truth_file = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "filePath": ["cam/a.JPG", "cam/b.JPG", "cam/c.JPG", "cam/d.JPG"],
            "scientificName": [None, "Sylvicapra grimmia", "Oryx gazella", None],
            "family": [None, "Bovidae", "Bovidae", None],
        }
    ).to_csv(truth_file, index=False)

    compared = evaluate.run(package, truth_file)

    assert dict(zip(compared["eventID"], compared["outcome"], strict=True)) == {
        "cam-0001": "not to species",
        "cam-0002": "correct",
        "cam-0003": "correct",
    }
    assert (package / "evaluation.csv").exists()


def test_run_with_reviews_as_truth(tmp_path):
    package = make_package(tmp_path)
    pd.DataFrame(
        {
            "eventID": ["cam-0001", "cam-0003"],
            "observationType": ["animal", "animal"],
            "scientificName": ["Bovidae", "Canis mesomelas"],
            "count": ["1", "1"],
            "reviewer": "Ivan Posavec",
            "reviewedAt": "2026-10-19T20:00:00+02:00",
        }
    ).to_csv(package / reviews.REVIEWS_FILE, index=False)

    compared = evaluate.run(package)

    assert dict(zip(compared["eventID"], compared["outcome"], strict=True)) == {"cam-0001": "correct", "cam-0003": "wrong"}
