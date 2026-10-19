from datetime import date

import pandas as pd
import pytest

from camtraptools import ecology, reviews
from camtraptools.detect import OBSERVATION_COLUMNS


@pytest.mark.parametrize(
    ("name", "expected"),
    [("Oryx gazella", True), ("Bovidae", False), (None, False), ("Canis lupus familiaris", False)],
)
def test_is_species(name, expected):
    assert ecology.is_species(name) == expected


def deployments():
    return pd.DataFrame(
        {
            "deploymentID": ["north-1", "north-2", "south"],
            "locationName": ["North", "North", "South"],
            "deploymentStart": ["2024-03-01T18:00:00", "2024-03-10T09:00:00", "2024-03-04T12:00:00"],
            "deploymentEnd": ["2024-03-03T06:00:00", "2024-03-11T09:00:00", "2024-03-15T08:00:00"],
        }
    )


def test_active_days_combines_deployments_at_a_site():
    days = ecology.active_days(deployments())

    assert sorted(days["North"]) == [date(2024, 3, day) for day in [1, 2, 3, 10, 11]]
    assert len(days["South"]) == 12


def test_effort_matrix_counts_active_days_per_occasion():
    effort = ecology.effort_matrix(deployments(), occasion_days=7)

    assert effort.columns.tolist() == ["2024-03-01", "2024-03-08", "2024-03-15"]
    assert effort.loc["North"].tolist() == [3, 2, 0]
    assert effort.loc["South"].tolist() == [4, 7, 1]


def test_effort_matrix_needs_deployment_periods():
    with pytest.raises(ValueError, match="no deployment"):
        ecology.effort_matrix(deployments().assign(deploymentEnd=None))


def test_detection_history_marks_detections_surveys_and_gaps():
    effort = ecology.effort_matrix(deployments(), occasion_days=7)
    events = pd.DataFrame(
        {"site": ["South", "North", "North"], "day": [date(2024, 3, 9), date(2024, 3, 16), date(2024, 3, 2)]}
    )

    history = ecology.detection_history(events, effort, occasion_days=7)

    assert history.loc["North"].tolist() == [1, 0, 1]
    assert history.loc["South"].tolist() == [0, 1, 0]


def test_detection_history_without_records_has_zeros_and_gaps():
    effort = ecology.effort_matrix(deployments(), occasion_days=7)

    history = ecology.detection_history(pd.DataFrame(columns=["site", "day"]), effort, occasion_days=7)

    assert history.loc["North"].tolist()[:2] == [0, 0]
    assert pd.isna(history.loc["North", "2024-03-15"])


def test_summarize_reports_naive_occupancy_and_detection_rate():
    history = pd.DataFrame(
        {"o1": [1, 0, 0, None], "o2": [0, 0, None, None], "o3": [1, 0, 0, None]},
        index=["a", "b", "c", "d"],
    ).astype("Int64")
    events = pd.DataFrame({"scientificName": ["Oryx gazella"] * 2})

    summary = ecology.summarize({"Oryx gazella": history}, events)

    assert summary.loc["Oryx gazella"].to_dict() == {
        "events": 2,
        "sitesDetected": 1,
        "sitesSurveyed": 3,
        "naiveOccupancy": 0.33,
        "detectionRate": 0.67,
    }


def make_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    deployments().to_csv(package / "deployments.csv", index=False)
    observations = pd.DataFrame(columns=OBSERVATION_COLUMNS).astype(object)
    observations["observationID"] = ["e1", "e2", "e3", "e4", "e5"]
    observations["deploymentID"] = ["north-1", "south", "south", "north-2", "south"]
    observations["eventID"] = ["north-1-0001", "south-0001", "south-0002", "north-2-0001", "south-0003"]
    observations["eventStart"] = [
        "2024-03-02T22:00:00", "2024-03-05T03:00:00", "2024-03-12T20:00:00", "2024-03-10T10:00:00", "2024-03-14T10:00:00",
    ]
    observations["observationLevel"] = "event"
    observations["observationType"] = ["animal", "animal", "animal", "animal", "blank"]
    observations["scientificName"] = ["Vulpes chama", "Vulpes chama", "Bovidae", None, None]
    observations["classificationMethod"] = "machine"
    observations.to_csv(package / "observations.csv", index=False)
    return package


def test_run_writes_histories_effort_and_summary(tmp_path):
    package = make_package(tmp_path)

    summary, left_out = ecology.run(package, tmp_path / "out")

    assert summary.index.tolist() == ["Vulpes chama"]
    assert summary.loc["Vulpes chama", "naiveOccupancy"] == 1.0
    assert left_out.to_dict() == {"Bovidae": 1, "animal (unidentified)": 1}
    history = pd.read_csv(tmp_path / "out" / "histories" / "Vulpes_chama.csv", index_col="site")
    assert history.loc["North"].tolist()[:2] == [1, 0]
    assert (tmp_path / "out" / ecology.EFFORT_FILE).exists()
    assert (tmp_path / "out" / ecology.SUMMARY_FILE).exists()


def test_run_with_higher_taxa_and_chosen_taxa(tmp_path):
    package = make_package(tmp_path)

    everything, _ = ecology.run(package, tmp_path / "all", all_taxa=True)
    chosen, _ = ecology.run(package, tmp_path / "chosen", taxa=["Bovidae", "Panthera leo"])

    assert sorted(everything.index) == ["Bovidae", "Vulpes chama"]
    assert chosen.index.tolist() == ["Bovidae"]


def test_run_uses_reviewed_decisions(tmp_path):
    package = make_package(tmp_path)
    reviews.save_review(package, "south-0002", "animal", "Raphicerus campestris", 1, "Tester")

    summary, _ = ecology.run(package, tmp_path / "out")

    assert summary.loc["Raphicerus campestris", "sitesDetected"] == 1
