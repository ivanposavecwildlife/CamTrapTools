import json
import sys
import types

import pandas as pd
import pytest

from camtraptools import detect
from camtraptools.ingest import run as ingest

RESULTS = {
    "images": [
        {
            "file": "camera_A/IMG_0001.JPG",
            "detections": [
                {"category": "1", "conf": 0.98, "bbox": [0.1, 0.2, 0.3, 0.4]}
            ],
        },
        {
            "file": "camera_A/IMG_0002.JPG",
            "detections": [
                {"category": "2", "conf": 0.9, "bbox": [0.5, 0.5, 0.2, 0.4]},
                {"category": "1", "conf": 0.05, "bbox": [0.0, 0.0, 0.1, 0.1]},
            ],
        },
        {"file": "camera_B/IMG_0001.JPG", "detections": []},
        {
            "file": "camera_B/IMG_0002.JPG",
            "failure": "Failure image access",
            "detections": None,
        },
    ],
    "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
    "info": {"format_version": "1.6", "detector": "MDV5A"},
}


@pytest.fixture
def results_file(tmp_path):
    path = tmp_path / "megadetector.json"
    path.write_text(json.dumps(RESULTS), encoding="utf-8")
    return path


def test_load_detections_has_one_row_per_detection(results_file):
    detections = detect.load_detections(results_file)

    assert list(detections["category"]) == ["animal", "person", "animal"]
    assert detections.loc[0].to_dict() == {
        "filePath": "camera_A/IMG_0001.JPG",
        "category": "animal",
        "conf": 0.98,
        "x": 0.1,
        "y": 0.2,
        "width": 0.3,
        "height": 0.4,
    }


def test_failed_images_are_listed(results_file):
    assert detect.failed_images(results_file) == ["camera_B/IMG_0002.JPG"]


def test_summarize_detections_counts_images_per_deployment(results_file):
    media = pd.DataFrame(
        {
            "deploymentID": ["camera_A", "camera_A", "camera_B", "camera_B"],
            "filePath": [image["file"] for image in RESULTS["images"]],
        }
    )

    summary = detect.summarize_detections(
        media, detect.load_detections(results_file), threshold=0.2
    )

    assert summary.loc["camera_A"].to_dict() == {
        "images": 2,
        "animal": 1,
        "person": 1,
        "vehicle": 0,
        "empty": 0,
    }
    assert summary.loc["camera_B"].to_dict() == {
        "images": 2,
        "animal": 0,
        "person": 0,
        "vehicle": 0,
        "empty": 2,
    }


def test_low_threshold_counts_weak_detections(results_file):
    media = pd.DataFrame(
        {"deploymentID": ["camera_A"], "filePath": ["camera_A/IMG_0002.JPG"]}
    )

    summary = detect.summarize_detections(
        media, detect.load_detections(results_file), threshold=0.01
    )

    assert summary.loc["camera_A", "animal"] == 1


def test_run_explains_missing_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "megadetector", None)

    with pytest.raises(RuntimeError, match="uv sync --extra detect"):
        detect.run(tmp_path, tmp_path)


def fake_megadetector(monkeypatch, calls):
    def load_and_run_detector_batch(model, image_files, quiet):
        calls["model"] = model
        calls["image_files"] = image_files
        return [{"file": file, "detections": []} for file in image_files]

    def write_results_to_file(results, output_file, relative_path_base, detector_file):
        calls["relative_path_base"] = relative_path_base
        images = [
            {"file": r["file"].removeprefix(relative_path_base + "/"), "detections": []}
            for r in results
        ]
        output = {
            "images": images,
            "detection_categories": RESULTS["detection_categories"],
            "info": {},
        }
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(output, file)

    module = types.ModuleType("megadetector.detection.run_detector_batch")
    module.load_and_run_detector_batch = load_and_run_detector_batch
    module.write_results_to_file = write_results_to_file
    for name in ["megadetector", "megadetector.detection"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules, "megadetector.detection.run_detector_batch", module
    )


def test_run_processes_exactly_the_ingested_images(make_image, monkeypatch, tmp_path):
    make_image(
        "images/camera_A/DCIM/IMG_0001.JPG", date_time_original="2024:03:01 06:00:00"
    )
    (tmp_path / "images/camera_A/DCIM/._IMG_0001.JPG").write_bytes(b"macOS metadata")
    ingest(tmp_path / "images", tmp_path / "package")
    calls = {}
    fake_megadetector(monkeypatch, calls)

    results_file = detect.run(tmp_path / "package", tmp_path / "images", model="MDV5A")

    assert calls["model"] == "MDV5A"
    assert calls["image_files"] == [str(tmp_path / "images/camera_A/DCIM/IMG_0001.JPG")]
    assert results_file == tmp_path / "package/megadetector.json"
    assert detect.failed_images(results_file) == []


def make_media():
    return pd.DataFrame(
        {
            "mediaID": [image["file"] for image in RESULTS["images"]],
            "deploymentID": ["camera_A", "camera_A", "camera_B", "camera_B"],
            "timestamp": [
                "2024-03-01T06:00:00",
                "2024-03-01T07:00:00",
                "2024-03-02T06:00:00",
                None,
            ],
            "filePath": [image["file"] for image in RESULTS["images"]],
        }
    )


def test_has_results_checks_file_and_model(results_file, tmp_path):
    assert detect.has_results(results_file, "MDV5A")
    assert not detect.has_results(results_file, "MDV1000-REDWOOD")
    assert not detect.has_results(tmp_path / "missing.json", "MDV5A")


def test_build_observations_follows_camtrap_dp(results_file):
    observations = detect.build_observations(make_media(), results_file, threshold=0.2)

    assert list(observations.columns) == detect.OBSERVATION_COLUMNS
    assert list(observations["observationType"]) == [
        "animal",
        "human",
        "blank",
        "unclassified",
    ]
    assert list(observations["observationID"]) == [
        "obs000001",
        "obs000002",
        "obs000003",
        "obs000004",
    ]
    assert set(observations["observationLevel"]) == {"media"}
    assert set(observations["classificationMethod"]) == {"machine"}


def test_build_observations_copies_box_and_confidence(results_file):
    animal = detect.build_observations(make_media(), results_file, threshold=0.2).iloc[
        0
    ]

    assert animal["mediaID"] == "camera_A/IMG_0001.JPG"
    assert animal["eventStart"] == animal["eventEnd"] == "2024-03-01T06:00:00"
    assert (
        animal["bboxX"],
        animal["bboxY"],
        animal["bboxWidth"],
        animal["bboxHeight"],
    ) == (0.1, 0.2, 0.3, 0.4)
    assert animal["classificationProbability"] == 0.98
    assert animal["count"] == 1
    assert animal["classifiedBy"] == "MegaDetector MDV5A"


def test_build_observations_keeps_every_confident_detection(results_file):
    observations = detect.build_observations(make_media(), results_file, threshold=0.01)

    image_2 = observations[observations["mediaID"] == "camera_A/IMG_0002.JPG"]
    assert list(image_2["observationType"]) == ["human", "animal"]


def test_write_observations_creates_csv(results_file):
    package = results_file.parent
    make_media().to_csv(package / "media.csv", index=False)

    detect.write_observations(package, threshold=0.2)

    lines = (package / "observations.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == detect.OBSERVATION_COLUMNS
    assert lines[1].split(",")[10] == "1"
