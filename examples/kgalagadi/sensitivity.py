"""How much do AI mistakes change ecological results, and how much do review and
adaptation help? The sensitivity study of the Kgalagadi case study.

Starting from a package with SpeciesNet results and events, crops from
``camtraptools crops`` and the truth file, it builds one package per labelling
strategy and compares their occupancy and activity results with the volunteer labels:

- SpeciesNet alone, and with the least confident 5% to 50% of events reviewed. The
  reviewer is simulated with the volunteer labels, so review is assumed to be correct.
- SpeciesNet with an adapted layer, cross-fitted: the cameras are split into four
  groups, and each group is labelled by a layer trained and tuned on the other groups
  only, so no camera is labelled by a layer that has seen it. Also with review.

Usage (with the ``detect`` extra installed, after the steps in the case study):

    uv run python examples/kgalagadi/sensitivity.py kgalagadi

Writes ``kgalagadi/sensitivity/`` with one package per strategy, the comparison from
``camtraptools compare`` and ``scores.csv``.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from camtraptools import adapt, compare, crops, detect, events, reviews
from camtraptools.evaluate import NON_TAXON_LABELS, truth_from_file

FOLDS = 4
REVIEW_BUDGETS = [0.05, 0.1, 0.2, 0.3, 0.5]
REVIEWED_BUDGETS_WITH_ADAPTATION = [0.1, 0.2]


def copy_package(source: Path, target: Path) -> Path:
    """Copy a package's tables, without reviews or adapted answers."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in ["media.csv", "deployments.csv", detect.RESULTS_FILE]:
        shutil.copy(source / name, target / name)
    return target


def label_package(package: Path, use_adapted: bool) -> None:
    """Write observations from SpeciesNet's results (and adapted answers) and events."""
    detect.write_observations(package, detect.DEFAULT_THRESHOLD, use_adapted)
    events.run(package)


def cross_fitted_answers(folder: Path, package: Path) -> pd.DataFrame:
    """Return adapted answers for every animal detection, each from a layer that was
    trained and tuned without the detection's camera."""
    crop_table = pd.read_csv(folder / "crops" / crops.CROPS_FILE)
    backbone, weight, bias, names = adapt.load_speciesnet()
    device = adapt.best_device()
    crop_features = adapt.cached_features(
        backbone, folder / "crops", crop_table["filePath"].tolist(), adapt.DEFAULT_MODEL,
        folder / "sensitivity" / "crop_features.npz", device,
    )
    detections_file = folder / "sensitivity" / "detections.csv"
    features_file = folder / "sensitivity" / "detection_features.npy"
    padding = float(crop_table["padding"].iloc[0]) if "padding" in crop_table.columns else crops.DEFAULT_PADDING
    if detections_file.exists() and features_file.exists():
        animals, detection_feature_rows = pd.read_csv(detections_file), np.load(features_file)
    else:
        animals, detection_feature_rows = adapt.detection_features(
            package, folder / "sd_cards", padding, detect.DEFAULT_THRESHOLD, backbone, device
        )
        animals.to_csv(detections_file, index=False)
        np.save(features_file, detection_feature_rows)

    location = dict(zip(crop_table["deploymentID"], crop_table["location"], strict=True))
    camera_crops = crop_table.groupby("location").size()
    groups = crops.split_locations(camera_crops, {f"group{number}": 1 / FOLDS for number in range(FOLDS)})
    group_of_detection = animals["filePath"].str.split("/").str[0].map(location).map(groups)
    group_names = sorted(set(groups.values()))

    answers = []
    for index, held_out in enumerate(group_names):
        validation = group_names[(index + 1) % FOLDS]
        crop_groups = crop_table["location"].map(groups)
        training = crop_table[~crop_groups.isin([held_out, validation])]
        classes = sorted(training["label"].unique())
        class_index = {name: position for position, name in enumerate(classes)}
        labels = crop_table["label"].map(class_index).fillna(-1).astype(int).to_numpy()
        train = crop_table.index.isin(training.index)
        val = ((crop_groups == validation) & (labels >= 0)).to_numpy()
        start_weight, start_bias, _ = adapt.initial_layer(classes, weight, bias, names)
        layer_weight, layer_bias, _ = adapt.train_layer(
            crop_features[train], labels[train], start_weight, start_bias, crop_features[val], labels[val]
        )
        predictions = crop_table.assign(
            split=np.where(val, "val", "other"), inTraining=labels >= 0
        )
        predictions["adapted"], predictions["adaptedConfidence"] = adapt.predict(
            crop_features, layer_weight, layer_bias, classes
        )
        threshold, _ = adapt.choose_threshold(predictions)
        layer = {"classes": classes, "weight": layer_weight, "bias": layer_bias, "created": f"fold {held_out}"}
        rows = (group_of_detection == held_out).to_numpy()
        answers.append(adapt.answer_detections(animals[rows], detection_feature_rows[rows], layer, threshold))
        print(f"  {held_out}: {len(classes)} labels, threshold {threshold}, {int(rows.sum())} detections")
    return pd.concat(answers, ignore_index=True)


def simulate_review(package: Path, truth: pd.Series, budget: float) -> int:
    """Review the least confident share ``budget`` of the package's events with the
    truth labels, as a careful reviewer would, and return how many were reviewed."""
    observations = pd.read_csv(package / detect.OBSERVATIONS_FILE, dtype=str)
    machine = observations[observations["classificationMethod"] != "human"]
    ordered = reviews.order_events(machine[machine["observationLevel"] == "event"], machine, "uncertainty")
    ordered = ordered[ordered["eventID"].isin(truth.index)]
    chosen = ordered.head(round(budget * len(ordered)))
    labels = chosen["eventID"].map(truth)
    table = pd.DataFrame(
        {
            "eventID": chosen["eventID"],
            "observationType": labels.where(labels.isin(NON_TAXON_LABELS), "animal"),
            "scientificName": labels.where(~labels.isin(NON_TAXON_LABELS)),
            "count": chosen["count"].fillna("1").where(labels != "blank"),
            "reviewer": "volunteer labels",
            "reviewedAt": "",
        }
    )
    table[reviews.REVIEW_COLUMNS].to_csv(package / reviews.REVIEWS_FILE, index=False)
    reviews.apply_reviews(package)
    return len(table)


def main(folder: Path) -> None:
    out = folder / "sensitivity"
    out.mkdir(exist_ok=True)
    truth_file = folder / "truth.csv"

    speciesnet = copy_package(folder / "package", out / "speciesnet")
    label_package(speciesnet, use_adapted=False)
    truth = truth_from_file(truth_file, speciesnet)

    print("Cross-fitting adapted layers:")
    adapted = copy_package(folder / "package", out / "adapted")
    cross_fitted_answers(folder, adapted).to_csv(adapted / detect.ADAPTED_FILE, index=False)
    label_package(adapted, use_adapted=True)

    others = {"adapted": adapted}
    for budget in REVIEW_BUDGETS:
        name = f"speciesnet_review{round(budget * 100)}"
        reviewed = copy_package(folder / "package", out / name)
        label_package(reviewed, use_adapted=False)
        count = simulate_review(reviewed, truth, budget)
        print(f"{name}: reviewed {count} events")
        others[name] = reviewed
    for budget in REVIEWED_BUDGETS_WITH_ADAPTATION:
        name = f"adapted_review{round(budget * 100)}"
        reviewed = copy_package(folder / "package", out / name)
        shutil.copy(adapted / detect.ADAPTED_FILE, reviewed / detect.ADAPTED_FILE)
        label_package(reviewed, use_adapted=True)
        count = simulate_review(reviewed, truth, budget)
        print(f"{name}: reviewed {count} events")
        others[name] = reviewed

    _, _, _, scores = compare.run(speciesnet, out / "comparison", truth_file, others)
    scores = scores.rename(index={"machine": "speciesnet"})
    scores.to_csv(out / "scores.csv")
    print(scores.to_string(na_rep="–"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python sensitivity.py FOLDER")
    main(Path(sys.argv[1]))
