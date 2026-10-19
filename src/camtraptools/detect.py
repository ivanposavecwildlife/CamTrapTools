"""Find and identify animals in images with SpeciesNet.

SpeciesNet runs MegaDetector to find animals, people and vehicles, then
classifies each animal and applies geofencing and taxonomic rollup. Detection
needs the optional ``detect`` extra, which installs the ``speciesnet`` package
(Apache-2.0). Its MegaDetector inference code is GPL-licensed.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
DEFAULT_THRESHOLD = 0.2
RESULTS_FILE = "speciesnet.json"
CATEGORIES = ["animal", "human", "vehicle"]
SPECIAL_LABELS = {"animal": "animal", "blank": "blank", "human": "human", "vehicle": "vehicle", "no cv result": "unknown"}


def run(
    package_dir: Path,
    images_dir: Path,
    model: str = DEFAULT_MODEL,
    country: str | None = None,
    admin1_region: str | None = None,
) -> Path:
    """Run SpeciesNet on every image in the package and save its predictions as
    ``speciesnet.json`` in the package, with paths relative to ``images_dir``."""
    try:
        from speciesnet import SpeciesNet
    except ImportError as error:
        raise RuntimeError(
            "Detection needs the optional 'detect' extra. Install it with: uv sync --extra detect"
        ) from error
    media = pd.read_csv(package_dir / "media.csv")
    image_files = [str(images_dir / file_path) for file_path in media["filePath"]]
    predictions = SpeciesNet(model).predict(filepaths=image_files, country=country, admin1_region=admin1_region)
    for prediction, file_path in zip(predictions["predictions"], media["filePath"], strict=True):
        prediction["filepath"] = file_path
    predictions["info"] = {
        **predictions.get("info", {}),
        "model": model,
        "country": country,
        "admin1_region": admin1_region,
        "completion_time": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    results_file = package_dir / RESULTS_FILE
    results_file.write_text(json.dumps(predictions, indent=1), encoding="utf-8")
    return results_file


def has_results(results_file: Path, model: str, country: str | None = None) -> bool:
    """Return True if ``results_file`` exists and was made with ``model`` for ``country``."""
    if not results_file.exists():
        return False
    info = json.loads(results_file.read_text(encoding="utf-8")).get("info", {})
    return info.get("model") == model and info.get("country") == country


def parse_taxon(label: str) -> tuple[str | None, str]:
    """Split a SpeciesNet label into a scientific name and a common name.

    Labels look like ``uuid;class;order;family;genus;species;common name``. The
    scientific name is the most specific rank given, e.g. ``Oryx gazella`` or
    ``Bovidae``. Special labels (``blank``, ``animal``, ``human``, ``vehicle`` and
    ``no cv result``, which means unknown) and labels without taxonomy have none.
    """
    parts = label.split(";")
    common_name = parts[-1]
    if common_name in SPECIAL_LABELS:
        return None, common_name
    taxon_class, order, family, genus, species = parts[1:6]
    if genus and species:
        return f"{genus.capitalize()} {species}", common_name
    rank = next((name for name in (genus, family, order, taxon_class) if name), None)
    return (rank.capitalize() if rank else None), common_name


def load_detections(results_file: Path) -> pd.DataFrame:
    """Return one row per detection: image, the detection's position in the image's
    results, category, confidence, bounding box and, for classified detections,
    SpeciesNet's final label and score.

    Bounding boxes are ``x``, ``y`` (top left), ``width`` and ``height``, as
    fractions of the image size.
    """
    results = json.loads(results_file.read_text(encoding="utf-8"))
    rows = []
    for prediction in results["predictions"]:
        for number, detection in enumerate(prediction.get("detections") or []):
            x, y, width, height = detection["bbox"]
            classification = detection.get("processed_classification") or {}
            rows.append(
                {
                    "filePath": prediction["filepath"],
                    "detection": number,
                    "category": detection["label"],
                    "conf": detection["conf"],
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "label": classification.get("class"),
                    "score": classification.get("score"),
                }
            )
    columns = ["filePath", "detection", "category", "conf", "x", "y", "width", "height", "label", "score"]
    return pd.DataFrame(rows, columns=columns)


def failed_images(results_file: Path) -> list[str]:
    """Return the images SpeciesNet could not fully process."""
    results = json.loads(results_file.read_text(encoding="utf-8"))
    return [prediction["filepath"] for prediction in results["predictions"] if prediction.get("failures")]


def summarize_detections(media: pd.DataFrame, detections: pd.DataFrame, threshold: float) -> pd.DataFrame:
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


def summarize_species(media: pd.DataFrame, detections: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Return the number of images per species (or higher taxon) and deployment,
    for animal detections at or above ``threshold``."""
    animals = detections[(detections["conf"] >= threshold) & detections["label"].notna()].copy()
    animals["species"] = [parse_taxon(label)[1] for label in animals["label"]]
    animals = animals[(animals["species"] == "animal") | ~animals["species"].isin(SPECIAL_LABELS)]
    animals = animals.merge(media[["filePath", "deploymentID"]], on="filePath")
    unique = animals.drop_duplicates(["filePath", "species"])
    return unique.groupby(["species", "deploymentID"]).size().unstack(fill_value=0)


OBSERVATIONS_FILE = "observations.csv"
ADAPTED_FILE = "adapted.csv"
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
def observation_type_and_name(detection) -> tuple[str, str | None]:
    """Return the Camtrap DP observation type and scientific name for a detection."""
    if not isinstance(detection.label, str):
        return detection.category, None
    scientific_name, common_name = parse_taxon(detection.label)
    return SPECIAL_LABELS.get(common_name, "animal"), scientific_name


def build_observations(
    media: pd.DataFrame, results_file: Path, threshold: float, adapted: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return Camtrap DP media-level observations: one per detection at or above
    ``threshold``, one "blank" per image without any, and one "unclassified" per
    image that was not processed or could not be.

    ``adapted`` holds answers from an adapted layer (see ``adapt.classify_detections``);
    detections where it is marked ``used`` take its scientific name and probability.
    """
    results = json.loads(results_file.read_text(encoding="utf-8"))
    completion_time = results.get("info", {}).get("completion_time")
    classifier = results.get("info", {}).get("classifier_name", "SpeciesNet")
    versions = {prediction["filepath"]: prediction.get("model_version") for prediction in results["predictions"]}
    detections = load_detections(results_file)
    confident = detections[detections["conf"] >= threshold]
    detections_by_image = dict(list(confident.groupby("filePath")))
    failed = set(failed_images(results_file))
    replacements = {}
    if adapted is not None:
        for row in adapted[adapted["used"]].itertuples():
            replacements[(row.filePath, row.detection)] = row
    rows = []
    for image in media.itertuples():
        base = {
            "deploymentID": image.deploymentID,
            "mediaID": image.mediaID,
            "eventStart": image.timestamp,
            "eventEnd": image.timestamp,
            "observationLevel": "media",
            "classificationMethod": "machine",
            "classifiedBy": f"{classifier} {versions.get(image.filePath) or ''}".strip(),
            "classificationTimestamp": completion_time,
        }
        if image.filePath in failed or image.filePath not in versions:
            rows.append({**base, "observationType": "unclassified"})
        elif image.filePath in detections_by_image:
            for detection in detections_by_image[image.filePath].itertuples():
                observation_type, scientific_name = observation_type_and_name(detection)
                probability = detection.score if isinstance(detection.label, str) else detection.conf
                classified_by = base["classifiedBy"]
                replacement = replacements.get((image.filePath, detection.detection))
                if replacement is not None:
                    observation_type, scientific_name = "animal", replacement.adapted
                    probability = replacement.adaptedConfidence
                    classified_by = f"{classified_by} with adapted layer {replacement.model}"
                rows.append(
                    {
                        **base,
                        "classifiedBy": classified_by,
                        "observationType": observation_type,
                        "scientificName": scientific_name,
                        "count": 1 if observation_type in CATEGORIES else None,
                        "bboxX": detection.x,
                        "bboxY": detection.y,
                        "bboxWidth": detection.width,
                        "bboxHeight": detection.height,
                        "classificationProbability": probability,
                    }
                )
        else:
            rows.append({**base, "observationType": "blank"})
    observations = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
    observations["observationID"] = [f"obs{number:06d}" for number in range(1, len(observations) + 1)]
    observations["count"] = observations["count"].astype("Int64")
    return observations


def write_observations(
    package_dir: Path, threshold: float = DEFAULT_THRESHOLD, use_adapted: bool = False
) -> pd.DataFrame:
    """Write ``observations.csv`` to the package from its SpeciesNet results and, with
    ``use_adapted``, the adapted layer's answers in ``adapted.csv``."""
    media = pd.read_csv(package_dir / "media.csv", dtype={"timestamp": str})
    adapted = pd.read_csv(package_dir / ADAPTED_FILE) if use_adapted else None
    observations = build_observations(media, package_dir / RESULTS_FILE, threshold, adapted)
    observations.to_csv(package_dir / OBSERVATIONS_FILE, index=False)
    return observations
