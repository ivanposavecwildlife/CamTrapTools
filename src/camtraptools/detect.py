"""Find and identify animals in images with SpeciesNet.

SpeciesNet runs MegaDetector to find animals, people and vehicles, then
classifies each animal and applies geofencing and taxonomic rollup. Detection
needs the optional ``detect`` extra, which installs the ``speciesnet`` package
(Apache-2.0). Its MegaDetector inference code is GPL-licensed.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from camtraptools import video

DEFAULT_MODEL = "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
DEFAULT_THRESHOLD = 0.2
RESULTS_FILE = "speciesnet.json"
INSTALL_DETECT = 'Install it with: pip install "camtraptools[detect]" (or uv sync --extra detect in a clone of the repository)'
CATEGORIES = ["animal", "human", "vehicle"]
SPECIAL_LABELS = {"animal": "animal", "blank": "blank", "human": "human", "vehicle": "vehicle", "no cv result": "unknown"}


PARTIAL_FILE = "speciesnet.partial.json"
PARTIAL_SETTINGS_FILE = "speciesnet.partial.settings.json"


def run_settings(
    media: pd.DataFrame,
    images_dir: Path,
    model: str,
    country,
    admin1_region,
    frame_interval: float = video.DEFAULT_FRAME_INTERVAL,
    max_frames: int = video.DEFAULT_MAX_FRAMES,
) -> dict:
    """Return what a SpeciesNet run depends on, so saved progress is only reused for
    exactly the same run: the model, region, images folder, list of images and how
    frames are sampled from videos."""
    files = sorted(media["filePath"])
    return {
        "model": model,
        "country": country,
        "admin1_region": admin1_region,
        "images": str(images_dir.resolve()),
        "imageCount": len(files),
        "imageList": hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest(),
        "frameInterval": frame_interval,
        "maxFrames": max_frames,
    }


def is_video_media(media: pd.DataFrame) -> pd.Series:
    """Return which media rows are videos."""
    if "fileMediatype" not in media.columns:
        return pd.Series(False, index=media.index)
    return media["fileMediatype"].fillna("").str.startswith("video/")


def best_frame(frames: list[dict]) -> dict | None:
    """Return the prediction of the frame whose most confident animal detection is
    the most confident of all frames, or of the frame with the most confident
    detection of any kind when no frame has an animal."""
    def confidence(prediction: dict, animals_only: bool) -> float:
        detections = [
            detection["conf"]
            for detection in prediction.get("detections") or []
            if not animals_only or detection.get("label") == "animal"
        ]
        return max(detections, default=-1.0)

    usable = [frame for frame in frames if not frame.get("failures")]
    if not usable:
        return None
    for animals_only in (True, False):
        best = max(usable, key=lambda frame: confidence(frame, animals_only))
        if confidence(best, animals_only) >= 0:
            return best
    return usable[0]


def saved_progress(package_dir: Path, settings: dict) -> int:
    """Return how many images an interrupted SpeciesNet run with the same settings has
    already saved, after deleting saved progress from a run with other settings."""
    partial = package_dir / PARTIAL_FILE
    settings_file = package_dir / PARTIAL_SETTINGS_FILE
    if not partial.exists():
        return 0
    if not settings_file.exists() or json.loads(settings_file.read_text(encoding="utf-8")) != settings:
        partial.unlink()
        settings_file.unlink(missing_ok=True)
        return 0
    try:
        return len(json.loads(partial.read_text(encoding="utf-8")).get("predictions", []))
    except json.JSONDecodeError:
        partial.unlink()
        return 0


def run(
    package_dir: Path,
    images_dir: Path,
    model: str = DEFAULT_MODEL,
    country: str | None = None,
    admin1_region: str | None = None,
    frame_interval: float = video.DEFAULT_FRAME_INTERVAL,
    max_frames: int = video.DEFAULT_MAX_FRAMES,
) -> Path:
    """Run SpeciesNet on every image in the package and save its predictions as
    ``speciesnet.json`` in the package, with paths relative to ``images_dir``.

    For videos, a frame is sampled every ``frame_interval`` seconds (at most
    ``max_frames``) into ``frames/`` in the package, and the video's result is that of
    its best frame (see ``best_frame``), whose path is recorded as ``frame``.

    SpeciesNet saves its progress every 10 minutes in ``speciesnet.partial.json``; if a
    run is interrupted, running it again with the same settings continues from there.
    """
    try:
        from speciesnet import SpeciesNet
    except ImportError as error:
        raise RuntimeError(
            f"Detection needs the optional 'detect' extra. {INSTALL_DETECT}"
        ) from error
    media = pd.read_csv(package_dir / "media.csv")
    settings = run_settings(media, images_dir, model, country, admin1_region, frame_interval, max_frames)
    saved_progress(package_dir, settings)
    partial = package_dir / PARTIAL_FILE
    (package_dir / PARTIAL_SETTINGS_FILE).write_text(json.dumps(settings), encoding="utf-8")
    videos = is_video_media(media)
    frames = {
        file_path: video.sample_frames(
            images_dir / file_path, package_dir / video.FRAMES_FOLDER / file_path, frame_interval, max_frames
        )
        for file_path in media.loc[videos, "filePath"]
    }
    image_files = [str(images_dir / file_path) for file_path in media.loc[~videos, "filePath"]]
    frame_files = [str(frame) for paths in frames.values() for frame in paths]
    SpeciesNet(model).predict(
        filepaths=image_files + frame_files,
        country=country,
        admin1_region=admin1_region,
        predictions_json=str(partial),
        progress_bars=True,
    )
    predictions = json.loads(partial.read_text(encoding="utf-8"))
    by_file = {prediction["filepath"]: prediction for prediction in predictions["predictions"]}
    combined = []
    for file_path, is_clip in zip(media["filePath"], videos, strict=True):
        if not is_clip:
            combined.append({**by_file[str(images_dir / file_path)], "filepath": file_path})
            continue
        clip_frames = [by_file[str(frame)] for frame in frames[file_path]]
        best = best_frame(clip_frames)
        if best is None:
            combined.append({"filepath": file_path, "failures": ["VIDEO"]})
            continue
        frame = Path(best["filepath"]).relative_to(package_dir).as_posix()
        combined.append({**best, "filepath": file_path, "frame": frame, "frames": len(clip_frames)})
    predictions["predictions"] = combined
    predictions["info"] = {
        **predictions.get("info", {}),
        "model": model,
        "country": country,
        "admin1_region": admin1_region,
        "completion_time": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    results_file = package_dir / RESULTS_FILE
    results_file.write_text(json.dumps(predictions, indent=1), encoding="utf-8")
    partial.unlink()
    (package_dir / PARTIAL_SETTINGS_FILE).unlink(missing_ok=True)
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
