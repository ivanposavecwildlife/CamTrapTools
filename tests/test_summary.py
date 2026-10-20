import pandas as pd

from camtraptools import summary
from camtraptools.detect import OBSERVATION_COLUMNS


def test_species_rows_format_numbers_and_warnings():
    table = pd.DataFrame(
        {
            "events": [170, 3],
            "rai": [18.181, 0.32],
            "sitesDetected": [16, 3],
            "sitesSurveyed": [20, 20],
            "naiveOccupancy": [0.8, 0.15],
            "occupancy": [0.8, None],
            "occupancyLower": [0.6, None],
            "occupancyUpper": [0.94, None],
            "warning": [None, None],
            "activityPeak": ["12:00", None],
        },
        index=["Oryx gazella", "Mellivora capensis"],
    )

    rows = summary.species_rows(table)

    assert rows[0] == {
        "name": "Oryx gazella", "events": 170, "rai": "18.18", "sites": "16/20", "naive": "0.80",
        "modelled": "0.80", "interval": "0.60–0.94", "warning": "", "peak": "12:00",
    }
    assert (rows[1]["modelled"], rows[1]["interval"], rows[1]["peak"]) == ("–", "", "–")


def test_labelling_reports_classifiers_reviews_and_accuracy(tmp_path):
    observations = pd.DataFrame(columns=OBSERVATION_COLUMNS).astype(object)
    observations["observationID"] = ["m1", "e1", "e2", "e1-human"]
    observations["eventID"] = ["e1", "e1", "e2", "e1"]
    observations["observationLevel"] = ["media", "event", "event", "event"]
    observations["classificationMethod"] = ["machine", "machine", "machine", "human"]
    observations["classifiedBy"] = ["SpeciesNet 4.0.3a", None, None, "Ivan"]
    observations.to_csv(tmp_path / "observations.csv", index=False)
    pd.DataFrame({"eventID": ["e1"], "observationType": ["blank"]}).to_csv(tmp_path / "reviews.csv", index=False)
    pd.DataFrame({"eventID": ["e1", "e2"], "outcome": ["correct", "wrong"]}).to_csv(tmp_path / "evaluation.csv", index=False)

    labels = summary.labelling(tmp_path)

    assert labels == {
        "classifiers": ["SpeciesNet 4.0.3a"],
        "events": 2,
        "reviewed": 1,
        "accuracy": {"events": 2, "correct": 1, "higher": 0, "wrong": 1},
    }
