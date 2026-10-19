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
