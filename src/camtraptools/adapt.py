"""Adapt SpeciesNet to a region by training a new final layer on its image features.

SpeciesNet's classifier is an EfficientNetV2-M backbone, which turns an animal crop
into a vector of features, followed by one linear layer that scores about 2,500
labels. Adapting keeps the backbone frozen and trains a new linear layer for only the
labels in the training crops, starting from SpeciesNet's own weights for the labels it
already knows. With a few hundred crops this trains in seconds and overfits far less
than retraining the whole network.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from camtraptools import __version__
from camtraptools.crops import CROPS_FILE, DEFAULT_PADDING, crop_box
from camtraptools.detect import (
    ADAPTED_FILE,
    DEFAULT_MODEL,
    RESULTS_FILE,
    load_detections,
    parse_taxon,
)

MODEL_FILE = "adapted.pt"
PREDICTIONS_FILE = "predictions.csv"
FEATURES_FILE = "features.npz"
IMAGE_SIZE = 480
FEATURE_BATCH_SIZE = 8
DEFAULT_EPOCHS = 300
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 0.01
PATIENCE = 30
FALLBACK_THRESHOLD = 0.9


def require_torch():
    """Return the torch module, or explain how to install it."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Adapting needs the optional 'detect' extra. Install it with: uv sync --extra detect"
        ) from error
    return torch


def best_device() -> str:
    """Return the fastest available device: an NVIDIA GPU, an Apple GPU, or the CPU."""
    torch = require_torch()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_classifier(model):
    """Split a classifier whose output is ``features @ weight + bias`` into a backbone
    that returns the features, and the final layer's weight (features × labels) and
    bias. Works on SpeciesNet's model, which is a ``torch.fx`` graph."""
    torch = require_torch()
    graph = model.graph.__deepcopy__()
    output = next(node for node in graph.nodes if node.op == "output")

    def operands(node):
        attributes = [arg for arg in node.args if getattr(arg, "op", None) == "get_attr"]
        inputs = [arg for arg in node.args if getattr(arg, "op", None) not in (None, "get_attr")]
        if len(attributes) != 1 or len(inputs) != 1:
            raise ValueError("the model does not end in a single linear layer")
        return inputs[0], _fetch(model, attributes[0].target)

    product, bias = operands(output.args[0])
    features, weight = operands(product)
    if weight.ndim != 2 or bias.shape != (weight.shape[1],):
        raise ValueError("the model does not end in a single linear layer")
    output.args = (features,)
    backbone = torch.fx.GraphModule(model, graph)
    backbone.graph.eliminate_dead_code()
    backbone.delete_all_unused_submodules()
    backbone.recompile()
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone, weight.detach().clone(), bias.detach().clone()


def _fetch(model, target: str):
    """Return the tensor a ``get_attr`` node refers to, e.g. ``initializers.weight``."""
    value = model
    for part in target.split("."):
        value = getattr(value, part)
    return value


def load_speciesnet(model: str = DEFAULT_MODEL):
    """Return SpeciesNet's backbone, final weight and bias, and the scientific name of
    each of its labels (``None`` for labels such as blank)."""
    torch = require_torch()
    try:
        from speciesnet.utils import ModelInfo
    except ImportError as error:
        raise RuntimeError(
            "Adapting needs the optional 'detect' extra. Install it with: uv sync --extra detect"
        ) from error
    info = ModelInfo(model)
    classifier = torch.load(info.classifier, map_location="cpu", weights_only=False)
    labels = Path(info.classifier_labels).read_text(encoding="utf-8").splitlines()
    names = [parse_taxon(label.strip())[0] for label in labels]
    backbone, weight, bias = split_classifier(classifier)
    if weight.shape[1] != len(names):
        raise ValueError(f"SpeciesNet has {weight.shape[1]} outputs but {len(names)} labels")
    return backbone, weight, bias, names


def preprocess(source: Path | Image.Image):
    """Return a crop (a file or an image) as SpeciesNet expects it: resized to
    480 × 480 pixels without keeping its shape, as height × width × colour values
    between 0 and 1."""
    torch = require_torch()
    if isinstance(source, Image.Image):
        pixels = np.asarray(source.convert("RGB"), dtype=np.float32) / 255
    else:
        with Image.open(source) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    resized = torch.nn.functional.interpolate(
        tensor[None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    return resized[0].permute(1, 2, 0)


def extract_features(backbone, images: list[Path | Image.Image], device: str) -> np.ndarray:
    """Return the backbone's features for every image (a file or an image), one row
    per image."""
    torch = require_torch()
    backbone = backbone.to(device)
    rows = []
    with torch.no_grad():
        for start in range(0, len(images), FEATURE_BATCH_SIZE):
            batch = torch.stack([preprocess(image) for image in images[start : start + FEATURE_BATCH_SIZE]])
            rows.append(backbone(batch.to(device)).float().cpu().numpy())
    return np.concatenate(rows) if rows else np.zeros((0, 0), dtype=np.float32)


def initial_layer(classes: list[str], weight, bias, speciesnet_names: list[str | None]):
    """Return a starting weight and bias for a layer over ``classes``. Each class that
    SpeciesNet already knows starts from SpeciesNet's weights for it; the others start
    at zero. Also returns which classes SpeciesNet knew."""
    torch = require_torch()
    index = {}
    for position, name in enumerate(speciesnet_names):
        if name is not None:
            index.setdefault(name, position)
    known = [name in index for name in classes]
    start_weight = torch.zeros(weight.shape[0], len(classes))
    start_bias = torch.zeros(len(classes))
    for column, name in enumerate(classes):
        if name in index:
            start_weight[:, column] = weight[:, index[name]]
            start_bias[column] = bias[index[name]]
    if any(known):
        start_bias[~torch.tensor(known)] = start_bias[torch.tensor(known)].mean()
    return start_weight, start_bias, known


def train_layer(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    start_weight,
    start_bias,
    val_features: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    seed: int = 0,
    balanced: bool = False,
):
    """Train the final layer with cross-entropy. If ``balanced``, each class is weighted
    by the inverse of its number of training crops, so common species do not drown out
    rare ones; otherwise every crop counts the same.

    With validation crops, keeps the layer from the epoch with the lowest validation
    loss, weighted the same way (epoch 0 is the starting layer), and stops after
    ``PATIENCE`` epochs without improvement. Returns the weight, bias and the epoch kept.
    """
    torch = require_torch()
    torch.manual_seed(seed)
    x = torch.from_numpy(train_features)
    y = torch.from_numpy(train_labels).long()
    counts = torch.bincount(y, minlength=start_weight.shape[1]).float()
    class_weights = torch.where(counts > 0, len(y) / (len(counts) * counts.clamp(min=1)), torch.zeros_like(counts))
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights if balanced else None)
    weight = start_weight.clone().requires_grad_(True)
    bias = start_bias.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([weight, bias], lr=learning_rate, weight_decay=weight_decay)
    has_val = val_features is not None and len(val_features) > 0
    if has_val:
        x_val = torch.from_numpy(val_features)
        y_val = torch.from_numpy(val_labels).long()

    def validation_loss() -> float:
        with torch.no_grad():
            return loss_function(x_val @ weight + bias, y_val).item()

    best = (validation_loss() if has_val else float("inf"), weight.detach().clone(), bias.detach().clone(), 0)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_function(x @ weight + bias, y)
        loss.backward()
        optimizer.step()
        if has_val:
            val_loss = validation_loss()
            if val_loss < best[0]:
                best = (val_loss, weight.detach().clone(), bias.detach().clone(), epoch)
            elif epoch - best[3] >= PATIENCE:
                break
    if has_val:
        return best[1], best[2], best[3]
    return weight.detach(), bias.detach(), epochs


def predict(features: np.ndarray, weight, bias, classes: list[str]) -> tuple[list[str], np.ndarray]:
    """Return the most likely class for every row of features, and its probability."""
    torch = require_torch()
    with torch.no_grad():
        probabilities = torch.softmax(torch.from_numpy(features) @ weight + bias, dim=1)
    confidence, best = probabilities.max(dim=1)
    return [classes[index] for index in best.tolist()], confidence.numpy()


def accuracy_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return, per split, how many crops have a label the layer was trained on, and
    how often each approach named them: SpeciesNet as run by ``detect``, SpeciesNet
    choosing only among the training labels (no training), and the adapted layer."""
    rows = []
    for split, group in predictions.groupby("split", sort=False):
        known = group[group["inTraining"]]
        rows.append(
            {
                "split": split,
                "crops": len(group),
                "trainedLabel": len(known),
                **{
                    column: round((known[source] == known["label"]).mean(), 2) if len(known) else None
                    for column, source in [("speciesnet", "aiLabel"), ("regionalList", "regionalList"), ("adapted", "adapted")]
                },
            }
        )
    return pd.DataFrame(rows).set_index("split")


def cached_features(backbone, crops_dir: Path, paths: list[str], model: str, cache: Path, device: str):
    """Return the features of every crop, reusing ``cache`` when it was made from the
    same crops and model, so retraining with other settings takes seconds."""
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if saved["paths"].tolist() == paths and str(saved["model"]) == model:
                return saved["features"]
    features = extract_features(backbone, [crops_dir / path for path in paths], device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, features=features, paths=np.array(paths), model=np.array(model))
    return features


def choose_threshold(predictions: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return the lowest confidence at which the adapted layer's answers on the
    validation crops are at least as precise as SpeciesNet's species-level answers on
    the same crops, and that precision. Above it, the adapted layer makes no more
    mistakes than SpeciesNet. The threshold is ``None`` if no confidence is precise
    enough, and ``FALLBACK_THRESHOLD`` if there are no validation crops to measure on.
    """
    val = predictions[(predictions["split"] == "val") & predictions["inTraining"]]
    named = val[val["aiLabel"].str.contains(" ", na=False)]
    if named.empty:
        return FALLBACK_THRESHOLD, None
    target = float((named["aiLabel"] == named["label"]).mean())
    for threshold in sorted(val["adaptedConfidence"].unique()):
        kept = val[val["adaptedConfidence"] >= threshold]
        if (kept["adapted"] == kept["label"]).mean() >= target:
            return float(threshold), target
    return None, target


def label_table(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return, for each trained label in ``split``, its number of crops and how often
    each approach named it, with the average over labels as the last row, so rare
    species count as much as common ones."""
    known = predictions[(predictions["split"] == split) & predictions["inTraining"]]
    approaches = {"speciesnet": "aiLabel", "regionalList": "regionalList", "adapted": "adapted"}
    table = pd.DataFrame(
        {column: (known[source] == known["label"]).groupby(known["label"]).mean() for column, source in approaches.items()}
    )
    table.insert(0, "crops", known["label"].value_counts())
    table = table.sort_values("crops", ascending=False)
    table.loc["average over labels"] = [len(known), *table[list(approaches)].mean()]
    return table.astype({"crops": int}).round(2)


def run(
    crops_dir: Path,
    out_dir: Path,
    model: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = 0,
    device: str | None = None,
    balanced: bool = False,
    speciesnet=None,
) -> tuple[pd.DataFrame, dict]:
    """Train an adapted final layer on the crops made by ``crops`` and save it as
    ``out_dir/adapted.pt``, with every crop's prediction in ``predictions.csv`` and the
    crops' SpeciesNet features in ``features.npz`` for later runs.

    Returns the predictions and a summary of the training. ``speciesnet`` can supply
    an already loaded backbone, weight, bias and label names.
    """
    torch = require_torch()
    crops = pd.read_csv(crops_dir / CROPS_FILE)
    classes = sorted(crops.loc[crops["split"] == "train", "label"].unique())
    if not classes:
        raise ValueError("there are no training crops")
    backbone, weight, bias, names = speciesnet or load_speciesnet(model)
    device = device or best_device()
    paths = crops["filePath"].tolist()
    features = cached_features(backbone, crops_dir, paths, model, out_dir / FEATURES_FILE, device)

    class_index = {name: position for position, name in enumerate(classes)}
    crops["inTraining"] = crops["label"].isin(class_index)
    labels = crops["label"].map(class_index).fillna(-1).astype(int).to_numpy()
    train = (crops["split"] == "train").to_numpy()
    val = ((crops["split"] == "val") & crops["inTraining"]).to_numpy()
    start_weight, start_bias, known = initial_layer(classes, weight, bias, names)
    new_weight, new_bias, epochs_used = train_layer(
        features[train], labels[train], start_weight, start_bias, features[val], labels[val],
        epochs, learning_rate, seed=seed, balanced=balanced,
    )

    crops["regionalList"], _ = predict(features, start_weight, start_bias, classes)
    crops["adapted"], crops["adaptedConfidence"] = predict(features, new_weight, new_bias, classes)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["filePath", "split", "label", "inTraining", "aiLabel", "regionalList", "adapted", "adaptedConfidence"]
    crops[columns].to_csv(out_dir / PREDICTIONS_FILE, index=False)
    threshold, speciesnet_precision = choose_threshold(crops)
    summary = {
        "classes": classes,
        "trainingCrops": crops.loc[train, "label"].value_counts().reindex(classes).astype(int).tolist(),
        "knownToSpeciesNet": known,
        "baseModel": model,
        "epochs": epochs_used,
        "balanced": balanced,
        "validated": bool(val.any()),
        "threshold": threshold,
        "speciesnetPrecision": speciesnet_precision,
        "padding": float(crops["padding"].iloc[0]) if "padding" in crops.columns else DEFAULT_PADDING,
        "device": device,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camtraptoolsVersion": __version__,
    }
    torch.save({**summary, "weight": new_weight.contiguous(), "bias": new_bias.contiguous()}, out_dir / MODEL_FILE)
    return crops, summary


def load_adapted(model_dir: Path) -> dict:
    """Return an adapted layer saved by ``run``: its classes, weight, bias, threshold
    and training details."""
    torch = require_torch()
    path = model_dir / MODEL_FILE if model_dir.is_dir() else model_dir
    if not path.exists():
        raise ValueError(f"no {MODEL_FILE} in {model_dir}; run `camtraptools adapt` first")
    return torch.load(path, weights_only=True)


def use_adapted(speciesnet_name, adapted: str, confidence: float, threshold: float | None, classes) -> bool:
    """Return whether the adapted answer replaces SpeciesNet's: when it is confident
    enough, unless SpeciesNet named a species the adapted layer was never trained on
    (it cannot name that species, so SpeciesNet's answer is kept)."""
    if threshold is None or confidence < threshold:
        return False
    named_species = isinstance(speciesnet_name, str) and " " in speciesnet_name
    return not named_species or speciesnet_name in classes


def classify_detections(
    package_dir: Path,
    images_dir: Path,
    adapted: dict,
    detection_threshold: float,
    threshold: float | None = None,
    device: str | None = None,
    speciesnet=None,
) -> pd.DataFrame:
    """Classify every animal detection at or above ``detection_threshold`` with an
    adapted layer, cropping it as the training crops were cropped, and save the
    answers as ``adapted.csv`` in the package.

    ``threshold`` overrides the layer's own threshold. Each row says whether the
    adapted answer is used instead of SpeciesNet's (see ``use_adapted``).
    """
    require_torch()
    threshold = adapted["threshold"] if threshold is None else threshold
    detections = load_detections(package_dir / RESULTS_FILE)
    animals = detections[(detections["category"] == "animal") & (detections["conf"] >= detection_threshold)]
    animals = animals.sort_values(["filePath", "detection"]).reset_index(drop=True)
    backbone, _, _, _ = speciesnet or load_speciesnet(adapted["baseModel"])
    device = device or best_device()
    batches, pending = [], []
    for file_path, group in animals.groupby("filePath", sort=False):
        with Image.open(images_dir / file_path) as image:
            for row in group.itertuples():
                box = crop_box(*image.size, (row.x, row.y, row.width, row.height), adapted["padding"])
                pending.append(image.crop(box))
        if len(pending) >= FEATURE_BATCH_SIZE:
            batches.append(extract_features(backbone, pending, device))
            pending = []
    if pending:
        batches.append(extract_features(backbone, pending, device))
    width = adapted["weight"].shape[0]
    features = np.concatenate(batches) if batches else np.zeros((0, width), dtype=np.float32)
    labels, confidence = predict(features, adapted["weight"], adapted["bias"], adapted["classes"])
    answers = animals[["filePath", "detection"]].copy()
    answers["speciesnet"] = [parse_taxon(label)[0] if isinstance(label, str) else None for label in animals["label"]]
    answers["adapted"] = labels
    answers["adaptedConfidence"] = confidence
    answers["used"] = [
        use_adapted(name, label, float(score), threshold, adapted["classes"])
        for name, label, score in zip(answers["speciesnet"], labels, confidence, strict=True)
    ]
    answers["model"] = f"({len(adapted['classes'])} labels, {adapted['created']})"
    answers.to_csv(package_dir / ADAPTED_FILE, index=False)
    return answers
