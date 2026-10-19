import pandas as pd
import pytest
from PIL import Image

from camtraptools import crops, reviews
from camtraptools.detect import OBSERVATION_COLUMNS


@pytest.mark.parametrize(
    ("bbox", "padding", "expected"),
    [
        ((0.25, 0.25, 0.5, 0.5), 0.0, (50, 25, 150, 75)),
        ((0.25, 0.25, 0.5, 0.5), 0.1, (40, 20, 160, 80)),
        ((0.0, 0.5, 1.0, 0.5), 0.1, (0, 45, 200, 100)),
    ],
)
def test_crop_box_pads_and_stays_inside_image(bbox, padding, expected):
    assert crops.crop_box(200, 100, bbox, padding) == expected


def test_location_of_prefers_location_id_then_name():
    deployments = pd.DataFrame(
        {
            "deploymentID": ["d1", "d2", "d3"],
            "locationID": ["L1", None, None],
            "locationName": ["North", "North", None],
        }
    )

    assert crops.location_of(deployments).to_dict() == {"d1": "L1", "d2": "North", "d3": "d3"}


def test_split_locations_keeps_each_location_in_one_split():
    counts = pd.Series({"A": 70, "B": 10, "C": 10, "D": 8, "E": 2})

    splits = crops.split_locations(counts, crops.DEFAULT_FRACTIONS)

    assert splits["A"] == "train"
    assert set(splits.values()) == {"train", "val", "test"}
    assert sorted(splits) == ["A", "B", "C", "D", "E"]


def test_split_locations_fills_every_split_when_locations_are_few():
    splits = crops.split_locations(pd.Series({"A": 100, "B": 90, "C": 1}), crops.DEFAULT_FRACTIONS)

    assert sorted(splits.values()) == ["test", "train", "val"]


def test_split_locations_is_repeatable():
    counts = pd.Series({name: 5 for name in "ABCDEFGH"})

    assert crops.split_locations(counts, crops.DEFAULT_FRACTIONS, seed=3) == crops.split_locations(
        counts, crops.DEFAULT_FRACTIONS, seed=3
    )


def test_folder_name():
    assert crops.folder_name("Oryx gazella") == "Oryx_gazella"
    assert crops.folder_name("Canis mesomelas (adult)") == "Canis_mesomelas_adult"


def make_package(tmp_path):
    images = tmp_path / "images"
    for camera in ["north", "south", "east"]:
        (images / camera).mkdir(parents=True)
        for name in ["a", "b"]:
            Image.new("RGB", (200, 100), "green").save(images / camera / f"{name}.JPG")
    package = tmp_path / "package"
    package.mkdir()
    paths = [f"{camera}/{name}.JPG" for camera in ["north", "south", "east"] for name in ["a", "b"]]
    pd.DataFrame({"mediaID": paths, "filePath": paths}).to_csv(package / "media.csv", index=False)
    pd.DataFrame({"deploymentID": ["north", "south", "east"]}).to_csv(package / "deployments.csv", index=False)

    observations = pd.DataFrame(columns=OBSERVATION_COLUMNS).astype(object)
    observations["observationID"] = [f"o{i}" for i in range(1, 7)] + ["e1", "e2", "e3"]
    observations["deploymentID"] = [path.split("/")[0] for path in paths] + ["north", "south", "east"]
    observations["mediaID"] = paths + [None] * 3
    observations["eventID"] = ["north-0001"] * 2 + ["south-0001"] * 2 + ["east-0001"] * 2 + [
        "north-0001",
        "south-0001",
        "east-0001",
    ]
    observations["observationLevel"] = ["media"] * 6 + ["event"] * 3
    observations["observationType"] = ["animal"] * 9
    observations["scientificName"] = ["Oryx gazella"] * 9
    observations["classificationMethod"] = "machine"
    observations.loc[:5, ["bboxX", "bboxY", "bboxWidth", "bboxHeight"]] = [0.25, 0.25, 0.5, 0.5]
    observations.loc[5, ["bboxWidth", "bboxHeight"]] = [0.05, 0.05]
    observations.to_csv(package / "observations.csv", index=False)
    return package, images


def test_run_crops_labelled_detections_by_location(tmp_path):
    package, images = make_package(tmp_path)
    truth = tmp_path / "truth.csv"
    truth.write_text(
        "filePath,scientificName\n"
        "north/a.JPG,Canis mesomelas\nnorth/b.JPG,\nsouth/a.JPG,Oryx gazella\n"
        "south/b.JPG,human\neast/a.JPG,Oryx gazella\neast/b.JPG,Oryx gazella\n"
    )

    dataset, skipped = crops.run(package, images, tmp_path / "out", truth)

    assert dataset[["observationID", "label", "aiLabel"]].values.tolist() == [
        ["o1", "Canis mesomelas", "Oryx gazella"],
        ["o3", "Oryx gazella", "Oryx gazella"],
        ["o5", "Oryx gazella", "Oryx gazella"],
    ]
    assert skipped == {
        "no true label": 0,
        "true label is blank (false detection)": 1,
        "true label is not an animal taxon": 1,
        "smaller than 32 pixels": 1,
    }
    assert dataset.groupby("location")["split"].nunique().max() == 1
    assert set(dataset["split"]) == {"train", "val", "test"}
    first = dataset.iloc[0]
    with Image.open(tmp_path / "out" / first["filePath"]) as crop:
        assert crop.size == (120, 60) == (first["width"], first["height"])
    assert first["filePath"] == f"{first['split']}/Canis_mesomelas/o1.jpg"
    assert pd.read_csv(tmp_path / "out" / crops.CROPS_FILE).shape == (3, len(crops.CROP_COLUMNS))


def test_run_uses_reviews_without_truth_file(tmp_path):
    package, images = make_package(tmp_path)
    reviews.save_review(package, "south-0001", "animal", "Raphicerus campestris", 1, "Tester")

    dataset, skipped = crops.run(package, images, tmp_path / "out")

    assert dataset[["observationID", "label"]].values.tolist() == [
        ["o3", "Raphicerus campestris"],
        ["o4", "Raphicerus campestris"],
    ]
    assert skipped["no true label"] == 4


def test_summarize_and_warnings():
    dataset = pd.DataFrame(
        {
            "label": ["Oryx gazella"] * 25 + ["Canis mesomelas"] * 2 + ["Vulpes chama"],
            "split": ["train"] * 24 + ["val"] + ["val", "test"] + ["train"],
        }
    )

    table = crops.summarize(dataset)

    assert table.loc["Oryx gazella"].to_dict() == {"train": 24, "val": 1, "test": 0, "total": 25}
    assert crops.warnings(dataset) == [
        "Canis mesomelas: 2 held-out crops but none for training",
        "Vulpes chama: only 1 training crop (fewer than 20)",
        "Vulpes chama: no crops in val or test, so it cannot be evaluated",
    ]
