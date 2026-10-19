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
from camtraptools.crops import CROPS_FILE
from camtraptools.detect import DEFAULT_MODEL, parse_taxon

MODEL_FILE = "adapted.pt"
PREDICTIONS_FILE = "predictions.csv"
IMAGE_SIZE = 480
FEATURE_BATCH_SIZE = 8
DEFAULT_EPOCHS = 300
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEIGHT_DECAY = 0.01
PATIENCE = 30


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


def preprocess(image_path: Path):
    """Return a crop as SpeciesNet expects it: resized to 480 × 480 pixels without
    keeping its shape, as height × width × colour values between 0 and 1."""
    torch = require_torch()
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    resized = torch.nn.functional.interpolate(
        tensor[None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    return resized[0].permute(1, 2, 0)


def extract_features(backbone, image_paths: list[Path], device: str) -> np.ndarray:
    """Return the backbone's features for every image, one row per image."""
    torch = require_torch()
    backbone = backbone.to(device)
    rows = []
    with torch.no_grad():
        for start in range(0, len(image_paths), FEATURE_BATCH_SIZE):
            batch = torch.stack([preprocess(path) for path in image_paths[start : start + FEATURE_BATCH_SIZE]])
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
):
    """Train the final layer with cross-entropy, weighting each class by the inverse of
    its number of training crops so common species do not drown out rare ones.

    With validation crops, keeps the layer from the epoch with the lowest validation
    loss (epoch 0 is the starting layer) and stops after ``PATIENCE`` epochs without
    improvement. Returns the weight, bias and the epoch kept.
    """
    torch = require_torch()
    torch.manual_seed(seed)
    x = torch.from_numpy(train_features)
    y = torch.from_numpy(train_labels).long()
    counts = torch.bincount(y, minlength=start_weight.shape[1]).float()
    class_weights = torch.where(counts > 0, len(y) / (len(counts) * counts.clamp(min=1)), torch.zeros_like(counts))
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    weight = start_weight.clone().requires_grad_(True)
    bias = start_bias.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([weight, bias], lr=learning_rate, weight_decay=weight_decay)
    has_val = val_features is not None and len(val_features) > 0
    if has_val:
        x_val = torch.from_numpy(val_features)
        y_val = torch.from_numpy(val_labels).long()

    def validation_loss() -> float:
        with torch.no_grad():
            return torch.nn.functional.cross_entropy(x_val @ weight + bias, y_val).item()

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
    how often SpeciesNet (as run by ``detect``) and the adapted layer named them."""
    rows = []
    for split, group in predictions.groupby("split", sort=False):
        known = group[group["inTraining"]]
        rows.append(
            {
                "split": split,
                "crops": len(group),
                "trainedLabel": len(known),
                "speciesnet": round((known["aiLabel"] == known["label"]).mean(), 2) if len(known) else None,
                "adapted": round((known["adapted"] == known["label"]).mean(), 2) if len(known) else None,
            }
        )
    return pd.DataFrame(rows).set_index("split")


def run(
    crops_dir: Path,
    out_dir: Path,
    model: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = 0,
    device: str | None = None,
    speciesnet=None,
) -> tuple[pd.DataFrame, dict]:
    """Train an adapted final layer on the crops made by ``crops`` and save it as
    ``out_dir/adapted.pt``, with every crop's prediction in ``predictions.csv``.

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
    features = extract_features(backbone, [crops_dir / path for path in crops["filePath"]], device)

    class_index = {name: position for position, name in enumerate(classes)}
    crops["inTraining"] = crops["label"].isin(class_index)
    labels = crops["label"].map(class_index).fillna(-1).astype(int).to_numpy()
    train = (crops["split"] == "train").to_numpy()
    val = ((crops["split"] == "val") & crops["inTraining"]).to_numpy()
    start_weight, start_bias, known = initial_layer(classes, weight, bias, names)
    new_weight, new_bias, epochs_used = train_layer(
        features[train], labels[train], start_weight, start_bias, features[val], labels[val],
        epochs, learning_rate, seed=seed,
    )

    crops["adapted"], crops["adaptedConfidence"] = predict(features, new_weight, new_bias, classes)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["filePath", "split", "label", "inTraining", "aiLabel", "adapted", "adaptedConfidence"]
    crops[columns].to_csv(out_dir / PREDICTIONS_FILE, index=False)
    summary = {
        "classes": classes,
        "trainingCrops": crops.loc[train, "label"].value_counts().reindex(classes).astype(int).tolist(),
        "knownToSpeciesNet": known,
        "baseModel": model,
        "epochs": epochs_used,
        "validated": bool(val.any()),
        "device": device,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camtraptoolsVersion": __version__,
    }
    torch.save({**summary, "weight": new_weight.contiguous(), "bias": new_bias.contiguous()}, out_dir / MODEL_FILE)
    return crops, summary
