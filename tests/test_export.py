import json
from zoneinfo import ZoneInfoNotFoundError

import pandas as pd
import pytest

from camtraptools import export
from camtraptools.detect import OBSERVATION_COLUMNS


def test_with_timezone_adds_offsets_including_daylight_saving():
    times = pd.Series(["2024-01-15T06:00:00", "2024-07-15T06:00:00", None])

    assert list(export.with_timezone(times, "Europe/Zagreb")) == [
        "2024-01-15T06:00:00+01:00",
        "2024-07-15T06:00:00+02:00",
        None,
    ]


def test_with_timezone_rejects_unknown_zone():
    with pytest.raises(ZoneInfoNotFoundError):
        export.with_timezone(pd.Series(["2024-01-15T06:00:00"]), "Mars/Olympus")


def make_deployments(**extra):
    return pd.DataFrame(
        {
            "deploymentID": ["camera_A", "camera_B"],
            "deploymentStart": ["2024-03-01T05:30:00", "2024-03-02T05:30:00"],
            "deploymentEnd": ["2024-03-20T07:00:00", "2024-03-25T07:00:00"],
            "latitude": ["52.70", "52.80"],
            "longitude": ["23.85", "23.95"],
            **extra,
        }
    )


def test_export_deployments_has_every_column_in_order():
    exported, extra = export.export_deployments(make_deployments(notes=["a", "b"]), "Africa/Johannesburg")

    assert list(exported.columns) == export.DEPLOYMENT_COLUMNS
    assert exported.loc[0, "deploymentStart"] == "2024-03-01T05:30:00+02:00"
    assert extra == ["notes"]


def test_export_deployments_without_coordinate_columns():
    deployments = make_deployments().drop(columns=["latitude", "longitude"])

    with pytest.raises(ValueError, match="deployments without coordinates: camera_A, camera_B"):
        export.export_deployments(deployments, "UTC")


def test_export_deployments_requires_coordinates():
    deployments = make_deployments()
    deployments.loc[1, "latitude"] = None

    with pytest.raises(ValueError, match="deployments without coordinates: camera_B"):
        export.export_deployments(deployments, "UTC")


def test_export_media_leaves_out_images_without_timestamp():
    media = pd.DataFrame(
        {
            "mediaID": ["m1", "m2"],
            "deploymentID": ["camera_A", "camera_A"],
            "timestamp": ["2024-03-01T06:00:00", None],
            "filePath": ["camera_A/m1.JPG", "camera_A/m2.JPG"],
            "fileName": ["m1.JPG", "m2.JPG"],
            "fileMediatype": ["image/jpeg", "image/jpeg"],
        }
    )

    exported = export.export_media(media, "UTC")

    assert list(exported.columns) == export.MEDIA_COLUMNS
    assert list(exported["mediaID"]) == ["m1"]
    assert exported.loc[0, "timestamp"] == "2024-03-01T06:00:00+00:00"
    assert exported.loc[0, "filePublic"] == "false"


def test_export_observations_keeps_exported_images_and_known_offsets():
    observations = pd.DataFrame(columns=OBSERVATION_COLUMNS)
    observations["observationID"] = ["obs1", "obs2", "event1"]
    observations["mediaID"] = ["m1", "m2", None]
    observations["eventStart"] = observations["eventEnd"] = ["2024-03-01T06:00:00"] * 3
    observations["classificationTimestamp"] = ["2026-10-19T10:00:00+02:00", "2026-10-19T10:00:00", None]

    exported = export.export_observations(observations, {"m1"}, "UTC")

    assert list(exported["observationID"]) == ["obs1", "event1"]
    assert list(exported["eventStart"]) == ["2024-03-01T06:00:00+00:00"] * 2
    assert exported.iloc[0]["classificationTimestamp"] == "2026-10-19T10:00:00+02:00"
    assert pd.isna(exported.iloc[1]["classificationTimestamp"])


def test_spatial_coverage_is_a_point_or_bounding_box():
    one = pd.DataFrame({"latitude": ["52.7"], "longitude": ["23.85"]})
    assert export.spatial_coverage(one) == {"type": "Point", "coordinates": [23.85, 52.7]}

    box = export.spatial_coverage(make_deployments())
    assert box["type"] == "Polygon"
    assert box["coordinates"][0][0] == box["coordinates"][0][-1] == [23.85, 52.7]


def test_descriptor_summarizes_the_package():
    deployments, _ = export.export_deployments(make_deployments(), "UTC")
    observations = pd.DataFrame(
        {
            "scientificName": ["Oryx gazella", None, "Bovidae", "Oryx gazella"],
            "observationLevel": ["media", "media", "event", "event"],
        }
    )

    package = export.descriptor(deployments, observations, "Test", ["Ivan", "Ana"], "targeted", "CC0-1.0", "CC-BY-4.0")

    assert package["profile"].endswith("/1.0.2/camtrap-dp-profile.json")
    assert package["temporal"] == {"start": "2024-03-01", "end": "2024-03-25"}
    assert package["taxonomic"] == [{"scientificName": "Bovidae"}, {"scientificName": "Oryx gazella"}]
    assert package["project"]["observationLevel"] == ["event", "media"]
    assert package["contributors"] == [{"title": "Ivan", "role": "contact"}, {"title": "Ana", "role": "contributor"}]
    assert [license["scope"] for license in package["licenses"]] == ["data", "media"]
    assert [resource["name"] for resource in package["resources"]] == ["deployments", "media", "observations"]


def make_package(tmp_path):
    tmp_path.mkdir()
    make_deployments(locationName=["Ridge", "Stream"]).to_csv(tmp_path / "deployments.csv", index=False)
    pd.DataFrame(
        {
            "mediaID": ["camera_A/m1.JPG", "camera_B/m2.JPG"],
            "deploymentID": ["camera_A", "camera_B"],
            "timestamp": ["2024-03-05T06:00:00", "2024-03-06T22:00:00"],
            "filePath": ["camera_A/m1.JPG", "camera_B/m2.JPG"],
            "fileName": ["m1.JPG", "m2.JPG"],
            "fileMediatype": ["image/jpeg", "image/jpeg"],
        }
    ).to_csv(tmp_path / "media.csv", index=False)
    observations = pd.DataFrame(columns=OBSERVATION_COLUMNS)
    observations["observationID"] = ["obs000001", "obs000002", "camera_A-0001-event"]
    observations["deploymentID"] = ["camera_A", "camera_B", "camera_A"]
    observations["mediaID"] = ["camera_A/m1.JPG", "camera_B/m2.JPG", None]
    observations["eventID"] = ["camera_A-0001", "camera_B-0001", "camera_A-0001"]
    observations["eventStart"] = observations["eventEnd"] = ["2024-03-05T06:00:00", "2024-03-06T22:00:00", "2024-03-05T06:00:00"]
    observations["observationLevel"] = ["media", "media", "event"]
    observations["observationType"] = ["animal", "blank", "animal"]
    observations["scientificName"] = ["Oryx gazella", None, "Oryx gazella"]
    observations["count"] = pd.array([1, None, 1], dtype="Int64")
    observations["bboxX"] = observations["bboxY"] = [0.1, None, None]
    observations["bboxWidth"] = observations["bboxHeight"] = [0.3, None, None]
    observations["classificationMethod"] = "machine"
    observations["classifiedBy"] = "SpeciesNet 4.0.3a"
    observations["classificationTimestamp"] = "2026-10-19T10:00:00+02:00"
    observations["classificationProbability"] = [0.99, None, 1.0]
    observations.to_csv(tmp_path / "observations.csv", index=False)
    return tmp_path


def test_run_writes_all_four_files(tmp_path):
    package = make_package(tmp_path / "work")

    left_out = export.run(package, tmp_path / "out", "Test", ["Ivan Posavec"], "Africa/Johannesburg")

    assert left_out == {"media_without_timestamp": 0, "extra_columns": []}
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "datapackage.json",
        "deployments.csv",
        "media.csv",
        "observations.csv",
    ]
    descriptor = json.loads((tmp_path / "out/datapackage.json").read_text(encoding="utf-8"))
    assert descriptor["spatial"]["type"] == "Polygon"


def can_reach_camtrap_dp():
    """Check the connection the way Frictionless makes it, with the requests library."""
    requests = pytest.importorskip("requests")
    try:
        requests.get(f"{export.CAMTRAP_DP_URL}/camtrap-dp-profile.json", timeout=10).raise_for_status()
    except requests.RequestException:
        return False
    return True


def test_exported_package_passes_frictionless_validation(tmp_path):
    frictionless = pytest.importorskip("frictionless")
    if not can_reach_camtrap_dp():
        pytest.skip("the Camtrap DP schemas could not be downloaded")
    package = make_package(tmp_path / "work")
    export.run(package, tmp_path / "out", "Test", ["Ivan Posavec"], "Africa/Johannesburg")

    report = frictionless.validate(tmp_path / "out/datapackage.json")

    assert report.valid, report.flatten(["type", "note"])


def test_run_exports_the_human_decision_for_reviewed_events(tmp_path):
    from camtraptools import reviews

    package = make_package(tmp_path / "work")
    reviews.save_review(package, "camera_A-0001", "animal", "Canis mesomelas", 1, "Ivan Posavec")

    export.run(package, tmp_path / "out", "Test", ["Ivan Posavec"], "Africa/Johannesburg")

    exported = pd.read_csv(tmp_path / "out/observations.csv")
    event_rows = exported[exported["observationLevel"] == "event"]
    assert list(event_rows["observationID"]) == ["camera_A-0001-human"]
    assert event_rows.iloc[0]["classificationMethod"] == "human"
