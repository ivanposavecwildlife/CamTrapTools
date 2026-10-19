import numpy as np
import pandas as pd
import pytest
from PIL import Image

from camtraptools.crops import CROPS_FILE

torch = pytest.importorskip("torch")
from camtraptools import adapt


class TinyClassifier(torch.nn.Module):
    """Stands in for SpeciesNet: average colour as features, then a linear layer."""

    def __init__(self, weight, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return torch.matmul(x.mean(dim=(1, 2)), self.weight) + self.bias


def tiny_speciesnet():
    weight = [[4.0, -4.0, 0.0], [-4.0, 4.0, 0.0], [0.0, 0.0, 1.0]]
    model = torch.fx.symbolic_trace(TinyClassifier(weight, [0.5, 0.25, 0.0]))
    backbone, weight, bias = adapt.split_classifier(model)
    return backbone, weight, bias, ["Oryx gazella", "Canis mesomelas", None]


def test_split_classifier_reproduces_the_model():
    model = torch.fx.symbolic_trace(TinyClassifier([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [0.1, 0.2]))
    x = torch.rand(2, 8, 8, 3)

    backbone, weight, bias = adapt.split_classifier(model)

    assert backbone(x).shape == (2, 3)
    assert weight.shape == (3, 2)
    assert torch.allclose(backbone(x) @ weight + bias, model(x))
    assert not any(parameter.requires_grad for parameter in backbone.parameters())


def test_split_classifier_rejects_other_models():
    class NoLinearEnd(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(1, 2)) + 1

    with pytest.raises(ValueError, match="single linear layer"):
        adapt.split_classifier(torch.fx.symbolic_trace(NoLinearEnd()))


def test_preprocess_resizes_to_speciesnet_input(tmp_path):
    path = tmp_path / "crop.jpg"
    Image.new("RGB", (100, 40), (255, 0, 0)).save(path)

    pixels = adapt.preprocess(path)

    assert pixels.shape == (480, 480, 3)
    assert 0.9 < pixels[..., 0].mean() <= 1.0
    assert pixels[..., 1].mean() < 0.1


def test_initial_layer_starts_known_labels_from_speciesnet():
    _, weight, bias, names = tiny_speciesnet()

    start_weight, start_bias, known = adapt.initial_layer(["Canis mesomelas", "Vulpes chama"], weight, bias, names)

    assert known == [True, False]
    assert torch.equal(start_weight[:, 0], weight[:, 1])
    assert torch.equal(start_weight[:, 1], torch.zeros(3))
    assert start_bias.tolist() == [0.25, 0.25]


def separable_features():
    rng = np.random.default_rng(0)
    features = np.concatenate([rng.normal(1, 0.1, (20, 2)), rng.normal(-1, 0.1, (20, 2))]).astype(np.float32)
    return features, np.array([0] * 20 + [1] * 20)


def test_train_layer_learns_separable_classes():
    features, labels = separable_features()

    weight, bias, epochs = adapt.train_layer(features, labels, torch.zeros(2, 2), torch.zeros(2), epochs=100)

    predicted, confidence = adapt.predict(features, weight, bias, ["a", "b"])
    assert predicted == ["a"] * 20 + ["b"] * 20
    assert confidence.min() > 0.5
    assert epochs == 100


def test_train_layer_keeps_the_epoch_with_lowest_validation_loss():
    features, labels = separable_features()

    _, _, epochs = adapt.train_layer(
        features, labels, torch.zeros(2, 2), torch.zeros(2), features, 1 - labels, epochs=2000
    )

    assert epochs == 0


def test_accuracy_table():
    predictions = pd.DataFrame(
        {
            "split": ["train", "train", "test", "test", "test"],
            "label": ["a", "b", "a", "b", "c"],
            "inTraining": [True, True, True, True, False],
            "aiLabel": ["a", "a", "a", "a", "c"],
            "adapted": ["a", "b", "a", "b", "a"],
        }
    )

    table = adapt.accuracy_table(predictions)

    assert table.loc["test"].to_dict() == {"crops": 3, "trainedLabel": 2, "speciesnet": 0.5, "adapted": 1.0}


def make_crops(tmp_path):
    colours = {"Oryx gazella": (255, 40, 40), "Vulpes chama": (40, 40, 255)}
    rows = []
    for split, location in [("train", "north"), ("val", "south"), ("test", "east")]:
        for label, colour in colours.items():
            for number in range(3):
                path = f"{split}/{label.replace(' ', '_')}/{location}{number}.jpg"
                (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (40, 40), colour).save(tmp_path / path)
                rows.append({"filePath": path, "split": split, "label": label, "aiLabel": "Oryx gazella"})
    pd.DataFrame(rows).to_csv(tmp_path / CROPS_FILE, index=False)
    return tmp_path


def test_run_trains_and_saves_an_adapted_layer(tmp_path):
    crops_dir = make_crops(tmp_path / "crops")

    predictions, summary = adapt.run(crops_dir, tmp_path / "model", device="cpu", speciesnet=tiny_speciesnet())

    assert summary["classes"] == ["Oryx gazella", "Vulpes chama"]
    assert summary["trainingCrops"] == [3, 3]
    assert summary["knownToSpeciesNet"] == [True, False]
    assert (predictions["adapted"] == predictions["label"]).all()
    saved = torch.load(tmp_path / "model" / adapt.MODEL_FILE, weights_only=True)
    assert saved["weight"].shape == (3, 2)
    assert saved["classes"] == summary["classes"]
    assert len(pd.read_csv(tmp_path / "model" / adapt.PREDICTIONS_FILE)) == 18


def test_run_needs_training_crops(tmp_path):
    crops_dir = make_crops(tmp_path / "crops")
    crops = pd.read_csv(crops_dir / CROPS_FILE)
    crops[crops["split"] != "train"].to_csv(crops_dir / CROPS_FILE, index=False)

    with pytest.raises(ValueError, match="no training crops"):
        adapt.run(crops_dir, tmp_path / "model", device="cpu", speciesnet=tiny_speciesnet())
