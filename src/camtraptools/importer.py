"""Import detection and classification results from other tools.

Results in the MegaDetector batch format (written by MegaDetector, AddaxAI,
Timelapse and run_md_and_speciesnet) and SpeciesNet's own format are translated
into ``speciesnet.json``, so every later step works the same as after ``detect``.
"""

import json
from pathlib import Path, PurePosixPath

import pandas as pd

from camtraptools.detect import RESULTS_FILE

DETECTOR_LABELS = {"animal": "animal", "person": "human", "human": "human", "vehicle": "vehicle"}


def match_path(path: str, known_paths: set[str]) -> str | None:
    """Return the package path that ``path`` refers to, or None.

    Tools write paths relative to different folders, e.g.
    ``KGA_S1/A02/A02_R1/IMG_0001.JPG`` for ``A02/A02_R1/IMG_0001.JPG``, so
    leading folders are removed one at a time until the path matches.
    """
    parts = PurePosixPath(path.replace("\\", "/").lstrip("/")).parts
    for start in range(len(parts)):
        candidate = "/".join(parts[start:])
        if candidate in known_paths:
            return candidate
    return None


def classification_label(category_id: str, results: dict) -> str:
    """Return a SpeciesNet-style label for a classification category.

    Taxonomy is used when the category description is already a SpeciesNet
    label (seven ``;``-separated fields); otherwise only the name is kept.
    """
    name = results["classification_categories"][category_id]
    description = results.get("classification_category_descriptions", {}).get(category_id, "")
    if description.count(";") == 6:
        return description
    return f";;;;;;{name}"


def from_megadetector_format(results: dict) -> tuple[list[dict], str]:
    """Translate MegaDetector-format results into SpeciesNet-style predictions and
    return them with the name of the tool that made them."""
    info = results.get("info", {})
    detector = info.get("detector_metadata", {}).get("megadetector_version") or info.get("detector", "")
    classified_by = info.get("classifier") or f"MegaDetector {detector}".strip()
    predictions = []
    for image in results["images"]:
        prediction = {"filepath": image["file"]}
        if image.get("failure"):
            prediction["failures"] = [image["failure"]]
        detections = []
        for detection in image.get("detections") or []:
            label = results["detection_categories"].get(detection["category"], "")
            converted = {
                "category": detection["category"],
                "label": DETECTOR_LABELS.get(label, label),
                "conf": detection["conf"],
                "bbox": detection["bbox"],
            }
            if detection.get("classifications"):
                category_id, score = detection["classifications"][0]
                converted["processed_classification"] = {
                    "class": classification_label(category_id, results),
                    "score": score,
                    "source": "imported",
                }
            detections.append(converted)
        prediction["detections"] = detections
        predictions.append(prediction)
    return predictions, classified_by


def completion_time(info: dict) -> str | None:
    """Return when the results were made, as an ISO date and time, if recorded."""
    time = info.get("classification_completion_time") or info.get("detection_completion_time")
    return pd.Timestamp(time).strftime("%Y-%m-%dT%H:%M:%S") if time else None


def run(package_dir: Path, results_file: Path) -> tuple[int, int]:
    """Import ``results_file`` into the package as ``speciesnet.json``.

    Returns the number of package images found in the file and the number of
    images in the file that are not in the package.
    """
    results = json.loads(results_file.read_text(encoding="utf-8"))
    if "predictions" in results:
        predictions, classifier_name = results["predictions"], "SpeciesNet"
    elif "images" in results:
        predictions, classifier_name = from_megadetector_format(results)
    else:
        raise ValueError(f"{results_file.name} is not in MegaDetector or SpeciesNet format")
    known_paths = set(pd.read_csv(package_dir / "media.csv")["filePath"])
    matched = []
    for prediction in predictions:
        path = match_path(prediction["filepath"], known_paths)
        if path is not None:
            matched.append({**prediction, "filepath": path})
    if not matched:
        raise ValueError(f"none of the images in {results_file.name} are in the package")
    imported = {
        "info": {
            "model": f"imported from {results_file.name}",
            "country": None,
            "classifier_name": classifier_name,
            "completion_time": completion_time(results.get("info", {})),
        },
        "predictions": matched,
    }
    (package_dir / RESULTS_FILE).write_text(json.dumps(imported, indent=1), encoding="utf-8")
    return len(matched), len(predictions) - len(matched)
