import numpy as np
import pandas as pd
import pytest

from camtraptools import ecology, reviews, uncertainty
from camtraptools.detect import OBSERVATION_COLUMNS

ANCESTRY = {
    "Oryx gazella": {"Oryx", "Bovidae", "Artiodactyla", "Mammalia"},
    "Sylvicapra grimmia": {"Sylvicapra", "Bovidae", "Artiodactyla", "Mammalia"},
    "Vulpes chama": {"Vulpes", "Canidae", "Carnivora", "Mammalia"},
}


def make_survey(tmp_path, events):
    """Write a package with three sites running two weeks from 1 March 2024 and one
    event per ``(eventID, site, day, type, name, confidence)``, each with one image."""
    package = tmp_path / "package"
    package.mkdir()
    pd.DataFrame(
        {
            "deploymentID": ["A", "B", "C"],
            "locationName": ["A", "B", "C"],
            "deploymentStart": "2024-03-01T00:00:00",
            "deploymentEnd": "2024-03-14T23:00:00",
        }
    ).to_csv(package / "deployments.csv", index=False)
    rows = []
    for event_id, site, day, kind, name, confidence in events:
        start = f"2024-03-{day:02d}T12:00:00"
        base = {"deploymentID": site, "eventID": event_id, "eventStart": start, "eventEnd": start,
                "observationType": kind, "scientificName": name, "classificationMethod": "machine"}
        rows.append(base | {"observationID": f"{event_id}-m", "mediaID": f"{event_id}.JPG", "observationLevel": "media",
                            "classificationProbability": confidence})
        rows.append(base | {"observationID": event_id, "observationLevel": "event", "classificationProbability": 1.0})
    pd.DataFrame(rows).reindex(columns=OBSERVATION_COLUMNS).to_csv(package / "observations.csv", index=False)
    return package


SURVEY = [
    ("a1", "A", 2, "animal", "Oryx gazella", 0.95),
    ("a2", "A", 9, "animal", "Oryx gazella", 0.4),
    ("b1", "B", 2, "animal", "Oryx gazella", 0.6),
    ("b2", "B", 3, "animal", None, 0.3),
    ("c1", "C", 4, "blank", None, None),
    ("c2", "C", 10, "animal", "Vulpes chama", 0.9),
]


def test_calibrate_without_reviews_takes_confidence_at_face_value():
    calibration = uncertainty.calibrate(np.array([]), np.array([]), np.array([]))

    assert (calibration.intercept, calibration.slope, calibration.offsets) == (0.0, 1.0, {})
    assert uncertainty.chance_right([0.3, 0.9], ["Oryx gazella", "blank"], calibration) == pytest.approx([0.3, 0.9])


def test_calibrate_learns_a_label_that_is_wrong_more_often():
    labels = np.array(["Oryx gazella"] * 20 + ["Vulpes chama"] * 20)
    right = np.array([True] * 19 + [False] + [False] * 15 + [True] * 5)

    calibration = uncertainty.calibrate(np.full(40, 0.9), labels, right)

    oryx, fox = uncertainty.chance_right([0.9, 0.9], ["Oryx gazella", "Vulpes chama"], calibration)
    assert oryx > 0.85
    assert fox < 0.5


@pytest.mark.parametrize(
    ("ai", "truth", "expected"),
    [
        ("Oryx gazella", "Oryx gazella", True),
        ("Bovidae", "Oryx gazella", True),
        ("animal", "Vulpes chama", True),
        ("animal", "blank", False),
        ("Oryx gazella", "Sylvicapra grimmia", False),
        ("blank", "Oryx gazella", False),
    ],
)
def test_is_right(ai, truth, expected):
    assert uncertainty.is_right(ai, truth, ANCESTRY) == expected


def test_right_shares_spread_higher_taxa_over_the_species_within():
    counts = pd.Series({"Oryx gazella": 3, "Sylvicapra grimmia": 1, "Vulpes chama": 4})

    assert uncertainty.right_shares("Oryx gazella", counts, ANCESTRY) == {"Oryx gazella": 1.0}
    assert uncertainty.right_shares("blank", counts, ANCESTRY) == {"blank": 1.0}
    assert uncertainty.right_shares("Bovidae", counts, ANCESTRY) == {"Oryx gazella": 0.75, "Sylvicapra grimmia": 0.25}
    assert uncertainty.right_shares("animal", counts, ANCESTRY)["Vulpes chama"] == 0.5
    assert uncertainty.right_shares("Felidae", counts, ANCESTRY) == {"Felidae": 1.0}


def test_wrong_shares_blend_the_label_corrections_with_all_corrections():
    corrections = pd.DataFrame(
        {"ai": ["blank", "blank", "Oryx gazella"], "truth": ["Oryx gazella", "Oryx gazella", "Sylvicapra grimmia"]}
    )
    overall = corrections["truth"].value_counts(normalize=True)

    shares = uncertainty.wrong_shares("blank", corrections, overall)

    assert shares["Oryx gazella"] == pytest.approx((2 + 2 * 2 / 3) / 4)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert uncertainty.wrong_shares("Oryx gazella", corrections, overall) == {"Sylvicapra grimmia": 1.0}
    assert uncertainty.wrong_shares("blank", corrections.iloc[0:0], overall.iloc[0:0]) == {}


def test_label_chances_are_certain_for_reviewed_events(tmp_path):
    package = make_survey(tmp_path, SURVEY)
    reviews.save_review(package, "b1", "animal", "Sylvicapra grimmia", 1, "Tester")
    deployments = pd.read_csv(package / "deployments.csv", dtype=str)

    chances = uncertainty.load(package, deployments)

    probabilities = chances.probabilities.set_index(chances.events["eventID"])
    assert chances.reviewed == 1
    assert probabilities.loc["b1", "Sylvicapra grimmia"] == 1.0
    assert probabilities.loc["b1"].sum() == 1.0
    assert 0 < probabilities.loc["a2", "Oryx gazella"] < 0.4
    assert probabilities.loc["c1", "Sylvicapra grimmia"] > 0
    assert (probabilities.sum(axis=1) <= 1 + 1e-9).all()


def test_cell_certainty_combines_the_events_in_a_cell():
    events = pd.DataFrame(
        {
            "site": ["A", "A", "B"],
            "day": pd.to_datetime(["2024-03-02", "2024-03-03", "2024-03-09"]).date,
        }
    )
    chances = uncertainty.LabelChances(
        events, pd.DataFrame({"Oryx gazella": [0.5, 0.4, 0.2]}), uncertainty.Calibration(), 0
    )
    effort = pd.DataFrame({"2024-03-01": [7, 7, 0], "2024-03-08": [7, 7, 7]}, index=["A", "B", "C"])

    certainty = uncertainty.cell_certainty(chances, "Oryx gazella", effort, 7)

    assert certainty.loc["A", "2024-03-01"] == pytest.approx(1 - 0.5 * 0.6)
    assert certainty.loc["B"].tolist() == [0.0, pytest.approx(0.2)]
    assert np.isnan(certainty.loc["C", "2024-03-01"])
    assert uncertainty.cell_certainty(chances, "Panthera leo", effort, 7).loc["A"].tolist() == [0.0, 0.0]


def test_site_doubt_counts_expected_false_and_missed_sites():
    history = pd.DataFrame({"w1": [1, 0, 0], "w2": [0, 0, 0]}, index=["A", "B", "C"])
    certainty = pd.DataFrame({"w1": [0.8, 0.5, 0.0], "w2": [0.0, 0.5, np.nan]}, index=["A", "B", "C"])

    false_sites, missed_sites = uncertainty.site_doubt(history, certainty)

    assert false_sites == pytest.approx(0.2)
    assert missed_sites == pytest.approx(0.75)


def test_ecology_reports_sites_in_doubt_after_enough_reviews(tmp_path, monkeypatch):
    package = make_survey(tmp_path, SURVEY)

    with pytest.raises(ValueError, match="at least 20 reviewed events"):
        ecology.run(package, tmp_path / "out", ai_errors=True)

    monkeypatch.setattr(uncertainty, "MIN_REVIEWS", 1)
    reviews.save_review(package, "a1", "animal", "Oryx gazella", 1, "Tester")
    summary, _, _ = ecology.run(package, tmp_path / "out", ai_errors=True)

    assert summary.loc["Oryx gazella", "expectedFalseSites"] > 0
    assert summary.loc["Vulpes chama", "expectedFalseSites"] == pytest.approx(0.1, abs=0.05)
