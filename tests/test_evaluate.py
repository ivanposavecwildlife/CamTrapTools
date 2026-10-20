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
        ("Bovidae", "Sylvicapra grimmia", "more specific"),
        ("Mammalia", "Vulpes chama", "more specific"),
        ("Canidae", "Sylvicapra grimmia", "wrong"),
        ("Rodentia", "animal", "wrong"),
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
            "truth": ["Oryx gazella", "Oryx gazella", "Oryx gazella", "blank", "blank", "Vulpes chama"],
            "ai": ["Oryx gazella", "Oryx gazella", "Bovidae", "blank", "Oryx gazella", "Bovidae"],
            "outcome": ["correct", "correct", "not to species", "correct", "wrong", "wrong"],
        }
    )

    metrics = evaluate.per_class_metrics(compared)

    assert metrics.loc["Oryx gazella"].to_dict() == {"events": 3, "predicted": 3, "right": 2, "precision": 0.67, "recall": 0.67}
    assert metrics.loc["blank"].to_dict() == {"events": 2, "predicted": 1, "right": 1, "precision": 1.0, "recall": 0.5}
    assert metrics.loc["Bovidae", "precision"] == 0.5
    assert pd.isna(metrics.loc["Bovidae", "recall"])


def test_thresholds_find_lowest_confidence_reaching_target():
    compared = pd.DataFrame(
        {
            "ai": ["Oryx gazella"] * 4 + ["Bovidae"] * 2,
            "outcome": ["wrong", "correct", "correct", "correct", "not to species", "wrong"],
            "confidence": [0.4, 0.6, 0.8, 0.9, 0.5, 0.5],
        }
    )

    table = evaluate.thresholds(compared, target_precision=0.95)

    assert table.loc["Oryx gazella"].to_dict() == {
        "answers": 4,
        "precision": 0.75,
        "threshold": 0.6,
        "kept": 3,
        "precisionKept": 1.0,
        "fewAnswers": True,
    }
    assert pd.isna(table.loc["Bovidae", "threshold"])
    assert table.loc["Bovidae", "kept"] == 0


def test_thresholds_trust_labels_with_enough_answers():
    compared = pd.DataFrame({"ai": ["blank"] * 12, "outcome": ["correct"] * 12, "confidence": [0.99] * 12})

    assert not evaluate.thresholds(compared).loc["blank", "fewAnswers"]


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
    assert compared["confidence"].between(0, 1).all()
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


def test_thumbnail_is_a_small_embedded_jpeg(make_image, tmp_path):
    from PIL import Image

    path = make_image("big.JPG")
    Image.new("RGB", (2000, 1500), "green").save(path)

    uri = evaluate.thumbnail(path)

    assert uri.startswith("data:image/jpeg;base64,")
    import base64
    import io

    with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as small:
        assert small.size == (480, 360)


def report_package(make_image, tmp_path):
    package = make_package(tmp_path)
    for name in ["a", "b", "c", "d"]:
        make_image(f"images/cam/{name}.JPG")
    pd.DataFrame(
        {
            "mediaID": ["cam/a.JPG", "cam/b.JPG", "cam/c.JPG", "cam/d.JPG"],
            "filePath": ["cam/a.JPG", "cam/b.JPG", "cam/c.JPG", "cam/d.JPG"],
            "fileName": ["a.JPG", "b.JPG", "c.JPG", "d.JPG"],
            "timestamp": ["2024-03-01T22:00:00"] * 4,
        }
    ).to_csv(package / "media.csv", index=False)
    return package


def test_failure_report_shows_mistakes_with_images(make_image, tmp_path):
    package = report_package(make_image, tmp_path)
    compared = pd.DataFrame(
        {
            "eventID": ["cam-0001", "cam-0002"],
            "truth": ["Vulpes <chama>", "Oryx gazella"],
            "ai": ["Bovidae", "Oryx gazella"],
            "outcome": ["wrong", "correct"],
            "confidence": [0.5, 0.99],
        }
    )

    page = evaluate.failure_report(package, tmp_path / "images", compared)

    assert page.startswith("<!doctype html>")
    assert "Events the AI did not get exactly right (1)" in page
    assert "cam-0001 · true: Vulpes &lt;chama&gt; · AI: Bovidae · wrong" in page
    assert "cam-0002 · true" not in page
    assert page.count("data:image/jpeg;base64,") == 2


def test_failure_report_when_everything_is_right(make_image, tmp_path):
    package = report_package(make_image, tmp_path)
    compared = pd.DataFrame({"eventID": ["cam-0002"], "truth": ["Oryx gazella"], "ai": ["Oryx gazella"], "outcome": ["correct"], "confidence": [0.99]})

    assert "The AI got every event exactly right." in evaluate.failure_report(package, tmp_path / "images", compared)


def test_run_can_compare_only_some_deployments(tmp_path):
    package = make_package(tmp_path)
    observations = pd.read_csv(package / "observations.csv", dtype=str)
    observations["deploymentID"] = ["cam", "cam", "cam", "cam", "north", "south", "north"]
    observations.to_csv(package / "observations.csv", index=False)
    truth_file = tmp_path / "truth.csv"
    truth_file.write_text("filePath,scientificName\ncam/a.JPG,\ncam/b.JPG,Oryx gazella\ncam/c.JPG,Oryx gazella\ncam/d.JPG,\n")

    compared = evaluate.run(package, truth_file, deployments=["north"])

    assert compared["eventID"].tolist() == ["cam-0001", "cam-0003"]
