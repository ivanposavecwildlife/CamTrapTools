"""Find animals, people and vehicles in images with MegaDetector.

Detection needs the optional ``detect`` extra, which installs the
``megadetector`` package. MegaDetector's model inference code is GPL-licensed.
"""

import json
from pathlib import Path

import pandas as pd

DEFAULT_MODEL = "MDV5A"
DEFAULT_THRESHOLD = 0.2
RESULTS_FILE = "megadetector.json"
CATEGORIES = ["animal", "person", "vehicle"]


def run(package_dir: Path, images_dir: Path, model: str = DEFAULT_MODEL) -> Path:
    """Run MegaDetector on every image in the package and save the results, in the
    standard MegaDetector batch format, as ``megadetector.json`` in the package."""
    try:
        from megadetector.detection.run_detector_batch import (
            load_and_run_detector_batch,
            write_results_to_file,
        )
    except ImportError as error:
        raise RuntimeError(
            "Detection needs the optional 'detect' extra. Install it with: uv sync --extra detect"
        ) from error
    media = pd.read_csv(package_dir / "media.csv")
    image_files = [str(images_dir / file_path) for file_path in media["filePath"]]
    results = load_and_run_detector_batch(model, image_files, quiet=True)
    results_file = package_dir / RESULTS_FILE
    write_results_to_file(
        results,
        str(results_file),
        relative_path_base=str(images_dir),
        detector_file=model,
    )
    return results_file


def load_detections(results_file: Path) -> pd.DataFrame:
    """Return one row per detection: image, category, confidence and bounding box.

    Bounding boxes are ``x``, ``y`` (top left), ``width`` and ``height``, as
    fractions of the image size.
    """
    results = json.loads(results_file.read_text(encoding="utf-8"))
    categories = results["detection_categories"]
    rows = []
    for image in results["images"]:
        for detection in image.get("detections") or []:
            x, y, width, height = detection["bbox"]
            rows.append(
                {
                    "filePath": image["file"],
                    "category": categories[detection["category"]],
                    "conf": detection["conf"],
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )
    return pd.DataFrame(
        rows, columns=["filePath", "category", "conf", "x", "y", "width", "height"]
    )


def failed_images(results_file: Path) -> list[str]:
    """Return the images MegaDetector could not process."""
    results = json.loads(results_file.read_text(encoding="utf-8"))
    return [image["file"] for image in results["images"] if image.get("failure")]


def summarize_detections(
    media: pd.DataFrame, detections: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Return, per deployment, how many images contain each category at or above
    ``threshold``, and how many contain nothing."""
    confident = detections[detections["conf"] >= threshold]
    images = media[["deploymentID", "filePath"]].copy()
    for category in CATEGORIES:
        found = confident.loc[confident["category"] == category, "filePath"]
        images[category] = images["filePath"].isin(found)
    images["empty"] = ~images[CATEGORIES].any(axis=1)
    summary = images.groupby("deploymentID")[[*CATEGORIES, "empty"]].sum()
    summary.insert(0, "images", images.groupby("deploymentID").size())
    return summary


OBSERVATIONS_FILE = "observations.csv"
OBSERVATION_COLUMNS = [
    "observationID",
    "deploymentID",
    "mediaID",
    "eventID",
    "eventStart",
    "eventEnd",
    "observationLevel",
    "observationType",
    "cameraSetupType",
    "scientificName",
    "count",
    "lifeStage",
    "sex",
    "behavior",
    "individualID",
    "individualPositionRadius",
    "individualPositionAngle",
    "individualSpeed",
    "bboxX",
    "bboxY",
    "bboxWidth",
    "bboxHeight",
    "classificationMethod",
    "classifiedBy",
    "classificationTimestamp",
    "classificationProbability",
    "observationTags",
    "observationComments",
]
OBSERVATION_TYPES = {"animal": "animal", "person": "human", "vehicle": "vehicle"}


def has_results(results_file: Path, model: str) -> bool:
    """Return True if ``results_file`` exists and was made with ``model``."""
    if not results_file.exists():
        return False
    results = json.loads(results_file.read_text(encoding="utf-8"))
    return results.get("info", {}).get("detector") == model


def build_observations(
    media: pd.DataFrame, results_file: Path, threshold: float
) -> pd.DataFrame:
    """Return Camtrap DP media-level observations: one per detection at or above
    ``threshold``, one "blank" per image without any, and one "unclassified" per
    image MegaDetector could not process."""
    info = json.loads(results_file.read_text(encoding="utf-8")).get("info", {})
    version = info.get("detector_metadata", {}).get("megadetector_version") or info.get(
        "detector", ""
    )
    classified_at = info.get("detection_completion_time")
    if classified_at:
        classified_at = pd.Timestamp(classified_at).strftime("%Y-%m-%dT%H:%M:%S")
    detections = load_detections(results_file)
    confident = detections[detections["conf"] >= threshold]
    detections_by_image = dict(list(confident.groupby("filePath")))
    failed = set(failed_images(results_file))
    rows = []
    for image in media.itertuples():
        base = {
            "deploymentID": image.deploymentID,
            "mediaID": image.mediaID,
            "eventStart": image.timestamp,
            "eventEnd": image.timestamp,
            "observationLevel": "media",
            "classificationMethod": "machine",
            "classifiedBy": f"MegaDetector {version}".strip(),
            "classificationTimestamp": classified_at,
        }
        if image.filePath in failed:
            rows.append({**base, "observationType": "unclassified"})
        elif image.filePath in detections_by_image:
            for detection in detections_by_image[image.filePath].itertuples():
                rows.append(
                    {
                        **base,
                        "observationType": OBSERVATION_TYPES[detection.category],
                        "count": 1,
                        "bboxX": detection.x,
                        "bboxY": detection.y,
                        "bboxWidth": detection.width,
                        "bboxHeight": detection.height,
                        "classificationProbability": detection.conf,
                    }
                )
        else:
            rows.append({**base, "observationType": "blank"})
    observations = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
    observations["observationID"] = [
        f"obs{number:06d}" for number in range(1, len(observations) + 1)
    ]
    observations["count"] = observations["count"].astype("Int64")
    return observations


def write_observations(package_dir: Path, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Write ``observations.csv`` to the package from its MegaDetector results."""
    media = pd.read_csv(package_dir / "media.csv", dtype={"timestamp": str})
    observations = build_observations(media, package_dir / RESULTS_FILE, threshold)
    observations.to_csv(package_dir / OBSERVATIONS_FILE, index=False)
    return observations
