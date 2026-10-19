import json
import sys
import types

import pandas as pd
import pytest

from camtraptools import detect
from camtraptools.ingest import run as ingest

GEMSBOK = "6038ccfc-94bd-4f04-9ec8-77a4f1fb78fe;mammalia;artiodactyla;bovidae;oryx;gazella;gemsbok"
BOVIDAE = "uuid-1;mammalia;artiodactyla;bovidae;;;bovidae family"
BLANK = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
HUMAN = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"
UNKNOWN = "f2efdae9-efb8-48fb-8a91-eccf79ab4ffb;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result"


def detection(label, conf, bbox, classification=None, score=0.9):
    result = {"category": "1", "label": label, "conf": conf, "bbox": bbox}
    if classification:
        result["processed_classification"] = {"class": classification, "score": score, "source": "classifier"}
    return result


RESULTS = {
    "info": {"format_version": "20261015", "model": detect.DEFAULT_MODEL, "country": "ZAF", "completion_time": "2026-10-19T10:00:00"},
    "predictions": [
        {
            "filepath": "camera_A/IMG_0001.JPG",
            "detections": [detection("animal", 0.98, [0.1, 0.2, 0.3, 0.4], GEMSBOK, 0.99)],
            "model_version": "4.0.3a",
        },
        {
            "filepath": "camera_A/IMG_0002.JPG",
            "detections": [
                detection("human", 0.9, [0.5, 0.5, 0.2, 0.4], HUMAN),
                detection("animal", 0.3, [0.0, 0.0, 0.1, 0.1], BOVIDAE, 0.7),
                detection("animal", 0.05, [0.6, 0.6, 0.1, 0.1]),
            ],
            "model_version": "4.0.3a",
        },
        {
            "filepath": "camera_B/IMG_0001.JPG",
            "detections": [detection("vehicle", 0.4, [0.0, 0.0, 1.0, 0.8], BLANK, 0.999)],
            "model_version": "4.0.3a",
        },
        {
            "filepath": "camera_B/IMG_0002.JPG",
            "detections": [detection("animal", 0.25, [0.4, 0.4, 0.1, 0.1], UNKNOWN, 0.6)],
            "model_version": "4.0.3a",
        },
        {"filepath": "camera_B/IMG_0003.JPG", "detections": [], "model_version": "4.0.3a"},
        {"filepath": "camera_B/IMG_0004.JPG", "failures": ["DETECTOR", "CLASSIFIER"], "model_version": "4.0.3a"},
    ],
}


@pytest.fixture
def results_file(tmp_path):
    path = tmp_path / "speciesnet.json"
    path.write_text(json.dumps(RESULTS), encoding="utf-8")
    return path


def make_media():
    files = [prediction["filepath"] for prediction in RESULTS["predictions"]]
    return pd.DataFrame(
        {
            "mediaID": files,
            "deploymentID": [file.split("/")[0] for file in files],
            "timestamp": [f"2024-03-0{day}T06:00:00" for day in range(1, 6)] + [None],
            "filePath": files,
        }
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (GEMSBOK, ("Oryx gazella", "gemsbok")),
        (BOVIDAE, ("Bovidae", "bovidae family")),
        ("uuid-2;aves;;;;;bird", ("Aves", "bird")),
        (BLANK, (None, "blank")),
        (HUMAN, (None, "human")),
        (UNKNOWN, (None, "no cv result")),
    ],
)
def test_parse_taxon(label, expected):
    assert detect.parse_taxon(label) == expected


def test_load_detections_has_one_row_per_detection(results_file):
    detections = detect.load_detections(results_file)

    assert list(detections["category"]) == ["animal", "human", "animal", "animal", "vehicle", "animal"]
    first = detections.loc[0].to_dict()
    assert first["filePath"] == "camera_A/IMG_0001.JPG"
    assert (first["x"], first["y"], first["width"], first["height"]) == (0.1, 0.2, 0.3, 0.4)
    assert (first["label"], first["score"]) == (GEMSBOK, 0.99)
    assert pd.isna(detections.loc[3, "label"])


def test_failed_images_are_listed(results_file):
    assert detect.failed_images(results_file) == ["camera_B/IMG_0004.JPG"]


def test_has_results_checks_model_and_country(results_file, tmp_path):
    assert detect.has_results(results_file, detect.DEFAULT_MODEL, "ZAF")
    assert not detect.has_results(results_file, detect.DEFAULT_MODEL, "BWA")
    assert not detect.has_results(results_file, "another-model", "ZAF")
    assert not detect.has_results(tmp_path / "missing.json", detect.DEFAULT_MODEL, "ZAF")


def test_summarize_detections_counts_images_per_deployment(results_file):
    summary = detect.summarize_detections(make_media(), detect.load_detections(results_file), threshold=0.2)

    assert summary.loc["camera_A"].to_dict() == {"images": 2, "animal": 2, "human": 1, "vehicle": 0, "empty": 0}
    assert summary.loc["camera_B"].to_dict() == {"images": 4, "animal": 1, "human": 0, "vehicle": 1, "empty": 2}


def test_summarize_species_counts_images_per_taxon(results_file):
    species = detect.summarize_species(make_media(), detect.load_detections(results_file), threshold=0.2)

    assert species.to_dict() == {"camera_A": {"bovidae family": 1, "gemsbok": 1}}


def test_build_observations_follows_camtrap_dp(results_file):
    observations = detect.build_observations(make_media(), results_file, threshold=0.2)

    assert list(observations.columns) == detect.OBSERVATION_COLUMNS
    assert list(observations["observationType"]) == ["animal", "human", "animal", "blank", "unknown", "blank", "unclassified"]
    assert list(observations["scientificName"].fillna("")) == ["Oryx gazella", "", "Bovidae", "", "", "", ""]
    assert list(observations["count"].fillna(0)) == [1, 1, 1, 0, 0, 0, 0]
    assert set(observations["classifiedBy"]) == {"SpeciesNet 4.0.3a"}
    assert set(observations["classificationTimestamp"]) == {"2026-10-19T10:00:00"}


def test_build_observations_copies_box_and_species_score(results_file):
    gemsbok = detect.build_observations(make_media(), results_file, threshold=0.2).iloc[0]

    assert gemsbok["mediaID"] == "camera_A/IMG_0001.JPG"
    assert gemsbok["eventStart"] == gemsbok["eventEnd"] == "2024-03-01T06:00:00"
    assert (gemsbok["bboxX"], gemsbok["bboxY"], gemsbok["bboxWidth"], gemsbok["bboxHeight"]) == (0.1, 0.2, 0.3, 0.4)
    assert gemsbok["classificationProbability"] == 0.99


def test_unclassified_detection_keeps_detector_category(results_file):
    observations = detect.build_observations(make_media(), results_file, threshold=0.01)

    weak = observations[(observations["mediaID"] == "camera_A/IMG_0002.JPG") & (observations["bboxX"] == 0.6)].iloc[0]
    assert weak["observationType"] == "animal"
    assert pd.isna(weak["scientificName"])
    assert weak["classificationProbability"] == 0.05


def test_write_observations_creates_csv(results_file):
    package = results_file.parent
    make_media().to_csv(package / "media.csv", index=False)

    detect.write_observations(package, threshold=0.2)

    lines = (package / "observations.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == detect.OBSERVATION_COLUMNS
    assert lines[1].split(",")[9:11] == ["Oryx gazella", "1"]


def test_run_explains_missing_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "speciesnet", None)

    with pytest.raises(RuntimeError, match="uv sync --extra detect"):
        detect.run(tmp_path, tmp_path)


def fake_speciesnet(monkeypatch, calls):
    class SpeciesNet:
        def __init__(self, model):
            calls["model"] = model

        def predict(self, filepaths, country, admin1_region):
            calls.update(filepaths=filepaths, country=country, admin1_region=admin1_region)
            predictions = [{"filepath": file, "detections": [], "model_version": "4.0.3a"} for file in filepaths]
            return {"info": {"format_version": "20261015"}, "predictions": predictions}

    module = types.ModuleType("speciesnet")
    module.SpeciesNet = SpeciesNet
    monkeypatch.setitem(sys.modules, "speciesnet", module)


def test_run_processes_exactly_the_ingested_images(make_image, monkeypatch, tmp_path):
    make_image("images/camera_A/DCIM/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00")
    (tmp_path / "images/camera_A/DCIM/._IMG_0001.JPG").write_bytes(b"macOS metadata")
    ingest(tmp_path / "images", tmp_path / "package")
    calls = {}
    fake_speciesnet(monkeypatch, calls)

    results_file = detect.run(tmp_path / "package", tmp_path / "images", country="ZAF")

    assert calls["model"] == detect.DEFAULT_MODEL
    assert calls["filepaths"] == [str(tmp_path / "images/camera_A/DCIM/IMG_0001.JPG")]
    assert (calls["country"], calls["admin1_region"]) == ("ZAF", None)
    results = json.loads(results_file.read_text(encoding="utf-8"))
    assert results["predictions"][0]["filepath"] == "camera_A/DCIM/IMG_0001.JPG"
    assert results["info"]["country"] == "ZAF"
    assert detect.has_results(results_file, detect.DEFAULT_MODEL, "ZAF")
