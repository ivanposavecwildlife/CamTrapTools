import json

import pandas as pd
import pytest

from camtraptools import detect, importer

SPECIESNET_GEMSBOK = "6038ccfc;mammalia;artiodactyla;bovidae;oryx;gazella;gemsbok"

MEGADETECTOR_RESULTS = {
    "info": {
        "format_version": "1.3",
        "detector": "md_v5a.0.0.pt",
        "detector_metadata": {"megadetector_version": "v5a.0.0"},
        "detection_completion_time": "2022-08-28 21:47:09",
    },
    "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
    "classification_categories": {"0": "gemsbok", "1": "fox"},
    "classification_category_descriptions": {
        "0": SPECIESNET_GEMSBOK,
        "1": "animalia;chordata;mammalia;carnivora;canidae;vulpesvulpes",
    },
    "images": [
        {
            "file": "KGA_S1/camera_A/IMG_0001.JPG",
            "detections": [
                {"category": "1", "conf": 0.98, "bbox": [0.1, 0.2, 0.3, 0.4], "classifications": [["0", 0.95], ["1", 0.03]]}
            ],
        },
        {
            "file": "KGA_S1/camera_A/IMG_0002.JPG",
            "detections": [
                {"category": "2", "conf": 0.9, "bbox": [0.5, 0.5, 0.2, 0.4]},
                {"category": "1", "conf": 0.6, "bbox": [0.0, 0.0, 0.2, 0.2], "classifications": [["1", 0.8]]},
            ],
        },
        {"file": "KGA_S1/camera_B/IMG_0001.JPG", "failure": "Failure image access", "detections": None},
        {"file": "KGA_S1/elsewhere/IMG_9999.JPG", "detections": []},
    ],
}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("camera_A/IMG_0001.JPG", "camera_A/IMG_0001.JPG"),
        ("KGA_S1/camera_A/IMG_0001.JPG", "camera_A/IMG_0001.JPG"),
        ("/data/KGA_S1/camera_A/IMG_0001.JPG", "camera_A/IMG_0001.JPG"),
        ("KGA_S1\\camera_A\\IMG_0001.JPG", "camera_A/IMG_0001.JPG"),
        ("camera_C/IMG_0001.JPG", None),
    ],
)
def test_match_path_removes_leading_folders(path, expected):
    assert importer.match_path(path, {"camera_A/IMG_0001.JPG", "camera_B/IMG_0001.JPG"}) == expected


def test_classification_label_keeps_speciesnet_taxonomy_only():
    assert importer.classification_label("0", MEGADETECTOR_RESULTS) == SPECIESNET_GEMSBOK
    assert importer.classification_label("1", MEGADETECTOR_RESULTS) == ";;;;;;fox"


def test_from_megadetector_format_translates_detections():
    predictions, classified_by = importer.from_megadetector_format(MEGADETECTOR_RESULTS)

    assert classified_by == "MegaDetector v5a.0.0"
    first, second, failed, _ = predictions
    assert first["detections"][0]["processed_classification"] == {
        "class": SPECIESNET_GEMSBOK,
        "score": 0.95,
        "source": "imported",
    }
    assert [detection["label"] for detection in second["detections"]] == ["human", "animal"]
    assert "processed_classification" not in second["detections"][0]
    assert failed["failures"] == ["Failure image access"]


def write_package(tmp_path, *file_paths):
    pd.DataFrame(
        {
            "mediaID": list(file_paths),
            "deploymentID": [path.split("/")[0] for path in file_paths],
            "timestamp": ["2024-03-01T06:00:00"] * len(file_paths),
            "filePath": list(file_paths),
        }
    ).to_csv(tmp_path / "media.csv", index=False)


def write_results(tmp_path, results, name="results.json"):
    path = tmp_path / name
    path.write_text(json.dumps(results), encoding="utf-8")
    return path


def test_run_imports_matching_images(tmp_path):
    write_package(tmp_path, "camera_A/IMG_0001.JPG", "camera_A/IMG_0002.JPG", "camera_B/IMG_0001.JPG", "camera_B/IMG_0002.JPG")

    matched, ignored = importer.run(tmp_path, write_results(tmp_path, MEGADETECTOR_RESULTS))

    assert (matched, ignored) == (3, 1)
    imported = json.loads((tmp_path / detect.RESULTS_FILE).read_text(encoding="utf-8"))
    assert [prediction["filepath"] for prediction in imported["predictions"]] == [
        "camera_A/IMG_0001.JPG",
        "camera_A/IMG_0002.JPG",
        "camera_B/IMG_0001.JPG",
    ]
    assert imported["info"]["model"] == "imported from results.json"
    assert imported["info"]["completion_time"] == "2022-08-28T21:47:09"


def test_imported_results_become_observations(tmp_path):
    write_package(tmp_path, "camera_A/IMG_0001.JPG", "camera_A/IMG_0002.JPG", "camera_B/IMG_0001.JPG", "camera_B/IMG_0002.JPG")
    importer.run(tmp_path, write_results(tmp_path, MEGADETECTOR_RESULTS))

    observations = detect.write_observations(tmp_path, threshold=0.2)

    assert list(observations["observationType"]) == ["animal", "human", "animal", "unclassified", "unclassified"]
    assert list(observations["scientificName"].fillna("")) == ["Oryx gazella", "", "", "", ""]
    assert set(observations["classifiedBy"]) == {"MegaDetector v5a.0.0"}


def test_run_imports_speciesnet_format(tmp_path):
    write_package(tmp_path, "camera_A/IMG_0001.JPG")
    results = {
        "predictions": [
            {"filepath": "/Users/someone/sd_cards/camera_A/IMG_0001.JPG", "detections": [], "model_version": "4.0.3a"}
        ]
    }

    assert importer.run(tmp_path, write_results(tmp_path, results)) == (1, 0)
    observations = detect.write_observations(tmp_path, threshold=0.2)
    assert (observations.loc[0, "observationType"], observations.loc[0, "classifiedBy"]) == ("blank", "SpeciesNet 4.0.3a")


def test_run_rejects_unknown_format(tmp_path):
    write_package(tmp_path, "camera_A/IMG_0001.JPG")

    with pytest.raises(ValueError, match="not in MegaDetector or SpeciesNet format"):
        importer.run(tmp_path, write_results(tmp_path, {"rows": []}))


def test_run_rejects_results_for_other_images(tmp_path):
    write_package(tmp_path, "camera_Z/IMG_0001.JPG")

    with pytest.raises(ValueError, match="none of the images"):
        importer.run(tmp_path, write_results(tmp_path, MEGADETECTOR_RESULTS))
    assert not (tmp_path / detect.RESULTS_FILE).exists()
