import math
from datetime import date

import pandas as pd
import pytest

from camtraptools import richness


def test_incidence_counts_camera_days_per_species():
    events = pd.DataFrame(
        {
            "site": ["a", "a", "a", "b"],
            "day": [date(2024, 3, 1), date(2024, 3, 1), date(2024, 3, 2), date(2024, 3, 1)],
            "scientificName": ["Vulpes chama", "Vulpes chama", "Vulpes chama", "Oryx gazella"],
        }
    )

    assert richness.incidence(events).to_dict() == {"Oryx gazella": 1, "Vulpes chama": 2}


def test_expected_species_matches_hand_calculation_and_ends():
    counts = pd.Series([1, 2])

    # in 2 of 4 camera-days: species 1 is missed with C(3,2)/C(4,2) = 1/2, species 2 with C(2,2)/C(4,2) = 1/6
    assert richness.expected_species(counts, 4, 2) == pytest.approx(0.5 + 5 / 6)
    assert richness.expected_species(counts, 4, 4) == pytest.approx(2)
    assert richness.expected_species(counts, 4, 1) == pytest.approx(3 / 4)
    assert richness.expected_species(counts, 4, 0) == 0


def test_accumulation_curve_rises_to_the_observed_species():
    curve = richness.accumulation(pd.Series([1, 1, 3, 10]), units=200, points=20)

    assert curve["cameraDays"].iloc[0] == 1 and curve["cameraDays"].iloc[-1] == 200
    assert curve["expectedSpecies"].is_monotonic_increasing
    assert curve["expectedSpecies"].iloc[-1] == 4


def test_chao2_with_duplicates_matches_hand_calculation():
    counts = pd.Series([1] * 7 + [2] * 2 + [5] * 16)

    estimate = richness.chao2(counts, 935)

    factor = 934 / 935
    assert estimate["estimatedSpecies"] == round(25 + factor * 49 / 4, 1)
    assert (estimate["estimatedLower"], estimate["estimatedUpper"]) == (27.2, 93.1)
    assert estimate["completeness"] == round(25 / (25 + factor * 49 / 4), 2)
    assert (estimate["uniques"], estimate["duplicates"]) == (7, 2)


def test_chao2_without_duplicates_uses_the_bias_corrected_form():
    estimate = richness.chao2(pd.Series([1, 1, 1, 4]), 100)

    assert estimate["estimatedSpecies"] == round(4 + 0.99 * 3 * 2 / 2, 1)
    assert estimate["estimatedLower"] >= 4


def test_chao2_without_rare_species_has_no_unseen_species():
    estimate = richness.chao2(pd.Series([5, 6, 7]), 50)

    assert (estimate["estimatedSpecies"], estimate["estimatedLower"], estimate["estimatedUpper"]) == (3, 3, 3)
    assert estimate["completeness"] == 1


def test_run_writes_curve_and_chart(tmp_path):
    events = pd.DataFrame(
        {
            "site": ["a", "b", "b"],
            "day": [date(2024, 3, day) for day in (1, 2, 3)],
            "scientificName": ["Vulpes chama", "Vulpes chama", "Oryx gazella"],
        }
    )

    estimate = richness.run(events, 30, tmp_path)

    assert estimate["observedSpecies"] == 2
    assert (tmp_path / richness.RICHNESS_FILE).exists()
    assert (tmp_path / richness.PLOT_FILE).read_text(encoding="utf-8").startswith("<svg")
    assert richness.run(events.iloc[0:0], 30, tmp_path / "empty") is None
    assert not math.isnan(estimate["completeness"])
