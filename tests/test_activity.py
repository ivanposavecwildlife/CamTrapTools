import math

import numpy as np
import pandas as pd
import pytest

from camtraptools import activity


def test_to_radians_places_times_on_the_day_circle():
    angles = activity.to_radians(pd.Series(["2024-03-01T00:00:00", "2024-03-01T06:00:00", "2024-03-02T18:30:00"]))

    assert angles == pytest.approx([0, math.pi / 2, 18.5 / 24 * 2 * math.pi])


@pytest.mark.parametrize(
    ("order", "value", "expected"),
    [(0, 1.0, 1.266066), (1, 1.0, 0.565159), (2, 2.0, 0.688948), (0, 0.0, 1.0)],
)
def test_scaled_bessel_matches_tables(order, value, expected):
    assert activity.scaled_bessel(order, value) * math.exp(value) == pytest.approx(expected, rel=1e-5)


def test_concentration_from_moment_inverts_bessel_ratio():
    length = activity.scaled_bessel(1, 2.5) / activity.scaled_bessel(0, 2.5)

    assert activity.concentration_from_moment(1, length) == pytest.approx(2.5, abs=0.01)


def von_mises(mean, kappa, size, seed=0):
    return np.mod(np.random.default_rng(seed).vonmises(mean, kappa, size), 2 * math.pi)


def test_two_opposite_peaks_are_not_mistaken_for_flat_activity():
    dawn_and_dusk = np.concatenate([von_mises(math.pi / 2, 8, 100, 1), von_mises(3 * math.pi / 2, 8, 100, 2)])

    assert abs(np.exp(1j * dawn_and_dusk).mean()) < 0.1
    assert activity.von_mises_concentration(dawn_and_dusk) > 3


def test_density_integrates_to_one():
    angles = von_mises(1, 2, 40)
    points = np.linspace(0, 2 * math.pi, 4000, endpoint=False)

    assert activity.density(angles, points, activity.bandwidth(angles)).mean() * 2 * math.pi == pytest.approx(1)


def test_overlap_is_high_for_the_same_pattern_and_low_for_opposite_ones():
    noon, midnight = von_mises(math.pi, 6, 40, 1), von_mises(0, 6, 40, 2)

    assert activity.overlap(noon, von_mises(math.pi, 6, 40, 3)) > 0.8
    assert activity.overlap(noon, midnight) < 0.15


def test_overlap_interval_is_repeatable_and_within_bounds():
    first, second = von_mises(1, 3, 30, 1), von_mises(2, 3, 30, 2)

    estimate, low, high = activity.overlap_interval(first, second, samples=200, seed=4)

    assert 0 <= low <= high <= 1
    assert low < estimate < high + 0.05
    assert activity.overlap_interval(first, second, samples=200, seed=4) == (estimate, low, high)


def test_overlaps_choose_the_estimator_by_sample_size():
    times = {"a": von_mises(1, 3, 80, 1), "b": von_mises(2, 3, 90, 2), "c": von_mises(3, 3, 20, 3)}

    table = activity.overlaps(times, samples=50)

    estimators = {(row.first, row.second): row.estimator for row in table.itertuples()}
    assert estimators == {("a", "b"): "Δ4", ("a", "c"): "Δ1", ("b", "c"): "Δ1"}


def test_activity_curves_are_shares_of_the_day():
    curves = activity.activity_curves({"Oryx gazella": von_mises(math.pi, 4, 50)})

    assert curves["Oryx gazella"].sum() * 24 / activity.GRID_POINTS == pytest.approx(1)
    assert activity.peak_hour(curves["Oryx gazella"], curves["hour"])[:2] in {"11", "12", "13"}


def events(name, hours, seed=0):
    rng = np.random.default_rng(seed)
    minutes = rng.integers(0, 60, len(hours))
    stamps = [f"2024-03-{day % 28 + 1:02d}T{hour:02d}:{minute:02d}:00" for day, (hour, minute) in enumerate(zip(hours, minutes))]
    return pd.DataFrame({"eventStart": stamps, "scientificName": name})


def test_run_writes_curves_overlaps_and_chart(tmp_path):
    data = pd.concat(
        [
            events("Oryx gazella", [11, 12, 12, 13, 10, 14] * 4),
            events("Canis mesomelas", [22, 23, 0, 1, 2, 21] * 4, seed=1),
            events("Vulpes chama", [23, 1]),
        ]
    )

    peaks, pairs = activity.run(data, tmp_path)

    assert peaks.index.tolist() == ["Oryx gazella", "Canis mesomelas"]
    assert peaks.loc["Oryx gazella", "events"] == 24
    assert pairs[["first", "second"]].values.tolist() == [["Oryx gazella", "Canis mesomelas"]]
    assert pairs.loc[0, "overlap"] < 0.3
    chart = (tmp_path / activity.PLOT_FILE).read_text(encoding="utf-8")
    assert chart.startswith("<svg") and "Canis mesomelas (24)" in chart and "Vulpes chama" not in chart
    assert list(pd.read_csv(tmp_path / activity.ACTIVITY_FILE).columns) == ["hour", "Oryx gazella", "Canis mesomelas"]


def test_run_without_enough_events(tmp_path):
    peaks, pairs = activity.run(events("Vulpes chama", [23, 1]), tmp_path)

    assert peaks.empty and pairs.empty
    assert not (tmp_path / activity.PLOT_FILE).exists()


def test_density_for_large_samples_matches_the_exact_one(monkeypatch):
    angles = np.concatenate([von_mises(1.5, 4, 300, 1), von_mises(4.8, 3, 300, 2)])
    points = activity.grid()
    concentration = activity.bandwidth(angles)
    exact = activity.density(angles, points, concentration)

    monkeypatch.setattr(activity, "DIRECT_LIMIT", 0)
    binned = activity.density(angles, points, concentration)

    assert np.abs(binned - exact).max() < 0.002 * exact.max()
    assert binned.mean() * 2 * math.pi == pytest.approx(1, abs=1e-3)
