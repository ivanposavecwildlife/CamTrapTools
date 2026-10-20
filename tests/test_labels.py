import pandas as pd
import pytest

from camtraptools import labels

TAXONOMY = pd.DataFrame(
    [
        ["Vulpes chama", "cape fox", "Vulpes", "Canidae", "Carnivora", "Mammalia"],
        ["Oryx gazella", "gemsbok", "Oryx", "Bovidae", "Artiodactyla", "Mammalia"],
    ],
    columns=["scientificName", "commonName", *labels.RANKS],
)


def make_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    pd.DataFrame(
        {
            "mediaID": ["cam_A/DCIM/IMG_0001.JPG", "cam_A/DCIM/IMG_0002.JPG", "cam_B/IMG_0001.JPG"],
            "deploymentID": ["cam_A", "cam_A", "cam_B"],
            "timestamp": ["2024-03-01T06:00:00", "2024-03-01T06:01:00", "2024-03-01T07:00:00"],
            "filePath": ["cam_A/DCIM/IMG_0001.JPG", "cam_A/DCIM/IMG_0002.JPG", "cam_B/IMG_0001.JPG"],
            "fileName": ["IMG_0001.JPG", "IMG_0002.JPG", "IMG_0001.JPG"],
        }
    ).to_csv(package / "media.csv", index=False)
    return package


def test_detect_format(tmp_path):
    (tmp_path / "dp").mkdir()
    (tmp_path / "images.csv").write_text("project_id,filename,deployment_id,is_blank,genus,species\n")
    (tmp_path / "timelapse.csv").write_text("File,RelativePath,Species\n")

    assert labels.detect_format(tmp_path / "dp") == "camtrap-dp"
    assert labels.detect_format(tmp_path / "images.csv") == "wildlife-insights"
    assert labels.detect_format(tmp_path / "timelapse.csv") == "csv"


def test_match_images_by_path_then_file_name(tmp_path):
    media = pd.read_csv(make_package(tmp_path) / "media.csv", dtype=str)
    found = pd.DataFrame(
        {
            "path": ["D:/cards/cam_A/DCIM/IMG_0001.JPG", "elsewhere/IMG_0002.JPG", "x/IMG_0001.JPG", "y/IMG_0001.JPG", "z/IMG_9.JPG"],
            "fileName": ["IMG_0001.JPG", "IMG_0002.JPG", "IMG_0001.JPG", "IMG_0001.JPG", "IMG_9.JPG"],
            "deployment": [None, None, "cam_B", None, None],
        }
    )

    matched, missing, ambiguous = labels.match_images(found, media)

    assert matched.tolist() == [
        "cam_A/DCIM/IMG_0001.JPG",
        "cam_A/DCIM/IMG_0002.JPG",
        "cam_B/IMG_0001.JPG",
        None,
        None,
    ]
    assert (missing, ambiguous) == (1, 1)


def test_resolve_names_uses_mapping_then_taxonomy(tmp_path):
    names_file = tmp_path / "names.csv"
    names_file.write_text("label,scientificName\nfoxy,Vulpes chama\nmisfire,\n")

    names, unknown = labels.resolve_names(
        pd.Series(["Gemsbok", "oryx GAZELLA", "foxy", "Empty", "Person", "Mystery", "misfire"]), names_file, TAXONOMY
    )

    resolved = names.set_index("label")
    assert resolved.loc["Gemsbok", ["scientificName", "family"]].tolist() == ["Oryx gazella", "Bovidae"]
    assert resolved.loc["oryx GAZELLA", "scientificName"] == "Oryx gazella"
    assert resolved.loc["foxy", ["scientificName", "genus"]].tolist() == ["Vulpes chama", "Vulpes"]
    assert resolved.loc["Empty", "scientificName"] == ""
    assert resolved.loc["misfire", "scientificName"] == ""
    assert resolved.loc["Person", "scientificName"] == "human"
    assert unknown == ["Mystery"]


def test_run_with_a_timelapse_export(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "speciesnet_taxonomy", lambda: TAXONOMY)
    package = make_package(tmp_path)
    source = tmp_path / "timelapse.csv"
    source.write_text(
        "RootFolder,RelativePath,File,Species\n"
        "cards,cam_A\\DCIM,IMG_0001.JPG,Cape fox\n"
        "cards,cam_A\\DCIM,IMG_0002.JPG,\n"
        "cards,cam_B,IMG_0001.JPG,Empty\n"
    )

    summary = labels.run(package, source, tmp_path / "truth.csv", species_column="Species")

    truth = pd.read_csv(tmp_path / "truth.csv")
    assert truth[["filePath", "scientificName", "family"]].fillna("").values.tolist() == [
        ["cam_A/DCIM/IMG_0001.JPG", "Vulpes chama", "Canidae"],
        ["cam_B/IMG_0001.JPG", "", ""],
    ]
    assert (summary["format"], summary["rows"], summary["matched"]) == ("csv", 2, 2)


def test_csv_needs_a_species_column(tmp_path):
    source = tmp_path / "labels.csv"
    source.write_text("path,label\na.JPG,fox\n")

    package = make_package(tmp_path)

    with pytest.raises(ValueError, match="--species-column"):
        labels.run(package, source, tmp_path / "truth.csv")
    with pytest.raises(ValueError, match="no column 'Species'"):
        labels.run(package, source, tmp_path / "truth.csv", species_column="Species")


def test_run_with_a_wildlife_insights_export(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "speciesnet_taxonomy", lambda: None)
    package = make_package(tmp_path)
    source = tmp_path / "images.csv"
    source.write_text(
        "project_id,deployment_id,image_id,filename,location,is_blank,class,order,family,genus,species,common_name\n"
        "1,cam_A,a,IMG_0001.JPG,gs://bucket/1/cam_A/IMG_0001.JPG,0,Mammalia,Carnivora,Canidae,Vulpes,chama,Cape Fox\n"
        "1,cam_A,b,IMG_0002.JPG,gs://bucket/1/cam_A/IMG_0002.JPG,1,,,,,,Blank\n"
        "1,cam_B,c,IMG_0001.JPG,gs://bucket/1/cam_B/IMG_0001.JPG,0,Mammalia,Artiodactyla,Bovidae,,,Bovidae Family\n"
    )

    summary = labels.run(package, source, tmp_path / "truth.csv")

    truth = pd.read_csv(tmp_path / "truth.csv").fillna("")
    assert summary["format"] == "wildlife-insights"
    assert truth.values.tolist() == [
        ["cam_A/DCIM/IMG_0001.JPG", "Vulpes chama", "Vulpes", "Canidae", "Carnivora", "Mammalia"],
        ["cam_A/DCIM/IMG_0002.JPG", "", "", "", "", ""],
        ["cam_B/IMG_0001.JPG", "Bovidae", "", "Bovidae", "Artiodactyla", "Mammalia"],
    ]


def test_run_with_a_camtrap_dp_package(tmp_path, monkeypatch):
    monkeypatch.setattr(labels, "speciesnet_taxonomy", lambda: None)
    package = make_package(tmp_path)
    source = tmp_path / "dp"
    source.mkdir()
    pd.DataFrame(
        {
            "mediaID": ["m1", "m2", "m3"],
            "deploymentID": ["cam_A", "cam_A", "cam_B"],
            "timestamp": ["2024-03-01T06:00:00+02:00", "2024-03-01T06:01:00+02:00", "2024-03-01T07:00:00+02:00"],
            "filePath": ["https://example.org/cam_A/IMG_0001.JPG", "https://example.org/cam_A/IMG_0002.JPG", "cam_B/IMG_0001.JPG"],
        }
    ).to_csv(source / "media.csv", index=False)
    pd.DataFrame(
        {
            "observationID": ["o1", "o2", "o3", "o4"],
            "deploymentID": ["cam_A", "cam_A", "cam_A", "cam_B"],
            "mediaID": ["m1", None, "m1", "m3"],
            "eventStart": [None, "2024-03-01T06:00:00+02:00", None, None],
            "eventEnd": [None, "2024-03-01T06:05:00+02:00", None, None],
            "observationType": ["animal", "animal", "animal", "blank"],
            "scientificName": ["Bovidae", "Oryx gazella", "Oryx gazella", None],
            "classificationMethod": ["machine", "human", "human", "human"],
        }
    ).to_csv(source / "observations.csv", index=False)

    summary = labels.run(package, source, tmp_path / "truth.csv")

    truth = pd.read_csv(tmp_path / "truth.csv").fillna("")
    assert truth[["filePath", "scientificName"]].values.tolist() == [
        ["cam_A/DCIM/IMG_0001.JPG", "Oryx gazella"],
        ["cam_A/DCIM/IMG_0002.JPG", "Oryx gazella"],
        ["cam_B/IMG_0001.JPG", ""],
    ]
    assert summary["format"] == "camtrap-dp"


def test_speciesnet_taxonomy_reads_a_downloaded_model(tmp_path, monkeypatch):
    folder = tmp_path / "models/google/speciesnet/pyTorch/v4.0.3a/1"
    folder.mkdir(parents=True)
    (folder / "taxonomy_release.txt").write_text(
        "u1;mammalia;carnivora;canidae;vulpes;chama;cape fox\nu2;;;;;;blank\nu3;aves;;;;;bird\n"
    )
    monkeypatch.setenv("KAGGLEHUB_CACHE", str(tmp_path))

    taxonomy = labels.speciesnet_taxonomy()

    assert taxonomy.values.tolist() == [
        ["Vulpes chama", "cape fox", "Vulpes", "Canidae", "Carnivora", "Mammalia"],
        ["Aves", "bird", "", "", "", "Aves"],
    ]
    monkeypatch.setenv("KAGGLEHUB_CACHE", str(tmp_path / "empty"))
    assert labels.speciesnet_taxonomy() is None
