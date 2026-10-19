"""Build a training dataset of animal crops from detections, split by camera location."""

import re
from pathlib import Path

import pandas as pd
from PIL import Image

from camtraptools.detect import OBSERVATIONS_FILE
from camtraptools.evaluate import (
    NON_TAXON_LABELS,
    media_labels_from_file,
    truth_from_reviews,
)

CROPS_FILE = "crops.csv"
SPLITS = ["train", "val", "test"]
DEFAULT_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}
DEFAULT_PADDING = 0.1
MIN_CROP_PIXELS = 32
MIN_TRAIN_CROPS = 20
JPEG_QUALITY = 95


CROP_COLUMNS = [
    "filePath", "split", "label", "observationID", "mediaID", "eventID",
    "deploymentID", "location", "aiLabel", "width", "height", "padding",
]


def crop_box(width: int, height: int, bbox: tuple[float, float, float, float], padding: float) -> tuple[int, ...]:
    """Return the pixel box (left, top, right, bottom) for a Camtrap DP bounding box
    (x, y, width, height as fractions of the image), widened by ``padding`` times its
    size on every side so the classifier sees some background, and kept inside the image."""
    x, y, box_width, box_height = bbox
    left = (x - padding * box_width) * width
    top = (y - padding * box_height) * height
    right = (x + box_width * (1 + padding)) * width
    bottom = (y + box_height * (1 + padding)) * height
    return (
        max(0, round(left)),
        max(0, round(top)),
        min(width, round(right)),
        min(height, round(bottom)),
    )


def location_of(deployments: pd.DataFrame) -> pd.Series:
    """Return each deployment's location, by deployment ID: its ``locationID``, else its
    ``locationName``, else the deployment itself. Deployments at the same place share a
    location, so they always end up in the same split."""
    location = deployments["deploymentID"].astype(str)
    for column in ["locationName", "locationID"]:
        if column in deployments.columns:
            location = deployments[column].fillna(location).astype(str)
    return pd.Series(location.to_numpy(), index=deployments["deploymentID"], name="location")


def split_locations(crops_per_location: pd.Series, fractions: dict[str, float], seed: int = 0) -> dict[str, str]:
    """Assign every location to one split, so no camera's background appears in more
    than one. Locations are handed out largest first, each to the split furthest below
    its target share of crops; ties are broken by a shuffle with ``seed``."""
    order = crops_per_location.sample(frac=1, random_state=seed).sort_values(ascending=False, kind="stable")
    total = order.sum()
    assigned = dict.fromkeys(fractions, 0)
    splits = {}
    for location, count in order.items():
        empty = [split for split in fractions if fractions[split] > 0 and assigned[split] == 0]
        if empty and len(order) - len(splits) <= len(empty):
            split = empty[0]
        else:
            split = max(fractions, key=lambda name: fractions[name] * total - assigned[name])
        splits[location] = split
        assigned[split] += count
    return splits


def crop_labels(package_dir: Path, observations: pd.DataFrame, truth_file: Path | None) -> pd.Series:
    """Return the true label of every media-level observation, by observation ID: the
    image's own label from ``truth_file``, or else the reviewed label of its event."""
    media = observations[observations["observationLevel"] == "media"]
    if truth_file is not None:
        by_media = media_labels_from_file(truth_file, set(media["mediaID"].dropna()))
        labels = media["mediaID"].map(by_media)
    else:
        labels = media["eventID"].map(truth_from_reviews(package_dir))
    return pd.Series(labels.to_numpy(), index=media["observationID"], name="label")


def folder_name(label: str) -> str:
    """Return a safe folder name for a label, e.g. ``Oryx_gazella``."""
    return re.sub(r"[^A-Za-z0-9-]+", "_", label).strip("_")


def run(
    package_dir: Path,
    images_dir: Path,
    out_dir: Path,
    truth_file: Path | None = None,
    padding: float = DEFAULT_PADDING,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Save a JPEG crop of every animal detection with a known true label, in
    ``out_dir/<split>/<label>/``, and a table of them in ``out_dir/crops.csv``.

    Returns that table and how many detections were skipped, by reason.
    """
    fractions = fractions or DEFAULT_FRACTIONS
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype={"eventID": str, "mediaID": str})
    deployments = pd.read_csv(package_dir / "deployments.csv", dtype=str)
    media = pd.read_csv(package_dir / "media.csv", dtype=str).set_index("mediaID")

    detections = observations[
        (observations["observationLevel"] == "media")
        & (observations["classificationMethod"] != "human")
        & observations["bboxX"].notna()
    ].copy()
    detections["label"] = detections["observationID"].map(crop_labels(package_dir, observations, truth_file))
    detections["location"] = detections["deploymentID"].map(location_of(deployments))

    skipped = {
        "no true label": int(detections["label"].isna().sum()),
        "true label is blank (false detection)": int((detections["label"] == "blank").sum()),
        "true label is not an animal taxon": int(
            (detections["label"].isin(NON_TAXON_LABELS) & (detections["label"] != "blank")).sum()
        ),
        f"smaller than {MIN_CROP_PIXELS} pixels": 0,
    }
    detections = detections[detections["label"].notna() & ~detections["label"].isin(NON_TAXON_LABELS)]
    splits = split_locations(detections.groupby("location").size(), fractions, seed)
    detections["split"] = detections["location"].map(splits)

    rows = []
    for row in detections.itertuples():
        with Image.open(images_dir / media.loc[row.mediaID, "filePath"]) as image:
            box = crop_box(*image.size, (row.bboxX, row.bboxY, row.bboxWidth, row.bboxHeight), padding)
            if min(box[2] - box[0], box[3] - box[1]) < MIN_CROP_PIXELS:
                skipped[f"smaller than {MIN_CROP_PIXELS} pixels"] += 1
                continue
            path = Path(row.split) / folder_name(row.label) / f"{row.observationID}.jpg"
            (out_dir / path).parent.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").crop(box).save(out_dir / path, quality=JPEG_QUALITY)
        rows.append(
            {
                "filePath": path.as_posix(),
                "split": row.split,
                "label": row.label,
                "observationID": row.observationID,
                "mediaID": row.mediaID,
                "eventID": row.eventID,
                "deploymentID": row.deploymentID,
                "location": row.location,
                "aiLabel": row.scientificName if isinstance(row.scientificName, str) else row.observationType,
                "width": box[2] - box[0],
                "height": box[3] - box[1],
                "padding": padding,
            }
        )
    crops = pd.DataFrame(rows, columns=CROP_COLUMNS)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops.to_csv(out_dir / CROPS_FILE, index=False)
    return crops, skipped


def summarize(crops: pd.DataFrame) -> pd.DataFrame:
    """Return crops per label and split, with a total column."""
    table = pd.crosstab(crops["label"], crops["split"]).reindex(columns=SPLITS, fill_value=0)
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def plural(count: int, noun: str) -> str:
    """Return e.g. ``1 crop`` or ``3 crops``."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def warnings(crops: pd.DataFrame) -> list[str]:
    """Return problems a classifier trained on these crops would have: labels that
    are only in validation or test, and labels with few training examples."""
    table = summarize(crops)
    messages = []
    for label, row in table.iterrows():
        held_out = row["val"] + row["test"]
        if row["train"] == 0:
            messages.append(f"{label}: {plural(held_out, 'held-out crop')} but none for training")
        elif row["train"] < MIN_TRAIN_CROPS:
            messages.append(f"{label}: only {plural(row['train'], 'training crop')} (fewer than {MIN_TRAIN_CROPS})")
        if held_out == 0:
            messages.append(f"{label}: no crops in val or test, so it cannot be evaluated")
    return messages
