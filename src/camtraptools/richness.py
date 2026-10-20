"""Species richness: how many species a survey found, and how many it likely missed.

The sampling unit is a camera-day: one camera site on one day it was running. The
species accumulation curve is the expected number of species found in a given number
of camera-days, computed exactly by rarefaction (Colwell et al. 2012). Chao2 (Chao
1987) estimates the total number of species from those recorded on only one or two
camera-days: many species seen only once suggest many were never seen.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

RICHNESS_FILE = "richness.csv"
PLOT_FILE = "richness.svg"
CURVE_POINTS = 60
Z_95 = 1.959964


def incidence(events: pd.DataFrame) -> pd.Series:
    """Return, for each species, the number of camera-days (site and day) on which it
    was recorded. ``events`` has ``site``, ``day`` and ``scientificName`` columns."""
    units = events.dropna(subset=["scientificName"]).drop_duplicates(["scientificName", "site", "day"])
    return units.groupby("scientificName").size()


def log_choose(n: int, k: int) -> float:
    """Return the natural logarithm of n choose k."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def expected_species(counts: pd.Series, units: int, sample: int) -> float:
    """Return the expected number of species found in ``sample`` of ``units``
    camera-days drawn without replacement, when each species was recorded on
    ``counts`` of them: the sum over species of 1 − C(units − count, sample) / C(units, sample)."""
    if sample <= 0:
        return 0.0
    total = 0.0
    for count in counts:
        if units - count < sample:
            total += 1
        else:
            total += 1 - math.exp(log_choose(units - count, sample) - log_choose(units, sample))
    return total


def accumulation(counts: pd.Series, units: int, points: int = CURVE_POINTS) -> pd.DataFrame:
    """Return the species accumulation curve: expected species for evenly spaced
    numbers of camera-days, from one to all of them."""
    samples = sorted(set(np.linspace(1, units, min(points, units)).round().astype(int).tolist()))
    return pd.DataFrame(
        {"cameraDays": samples, "expectedSpecies": [round(expected_species(counts, units, k), 2) for k in samples]}
    )


def chao2(counts: pd.Series, units: int) -> dict:
    """Return the Chao2 estimate of total species richness with its 95% interval
    (log-normal, Chao 1987), from the species recorded on exactly one camera-day
    (uniques) and two (duplicates). The interval's lower end is never below the
    species observed."""
    observed = len(counts)
    uniques, duplicates = int((counts == 1).sum()), int((counts == 2).sum())
    factor = (units - 1) / units if units else 0
    if duplicates > 0:
        unseen = factor * uniques**2 / (2 * duplicates)
        ratio = uniques / duplicates
        variance = duplicates * (factor / 4 * ratio**4 + factor * ratio**3 + factor / 2 * ratio**2)
    else:
        unseen = factor * uniques * (uniques - 1) / 2
        variance = (
            factor * uniques * (uniques - 1) / 2
            + factor**2 * uniques * (2 * uniques - 1) ** 2 / 4
            - factor**2 * uniques**4 / (4 * max(unseen + observed, 1))
        )
    estimate = observed + unseen
    if unseen > 0 and variance > 0:
        spread = math.exp(Z_95 * math.sqrt(math.log(1 + variance / unseen**2)))
        lower, upper = observed + unseen / spread, observed + unseen * spread
    else:
        lower = upper = estimate
    return {
        "observedSpecies": observed,
        "uniques": uniques,
        "duplicates": duplicates,
        "estimatedSpecies": round(estimate, 1),
        "estimatedLower": round(lower, 1),
        "estimatedUpper": round(upper, 1),
        "completeness": round(observed / estimate, 2) if estimate else None,
        "cameraDays": units,
    }


def plot(curve: pd.DataFrame, estimate: dict) -> str:
    """Return an SVG chart of the accumulation curve with the Chao2 estimate."""
    width, height, left, right, top, bottom = 720, 340, 56, 24, 20, 46
    inner_width, inner_height = width - left - right, height - top - bottom
    peak = max(min(estimate["estimatedUpper"], estimate["estimatedSpecies"] * 1.5), curve["expectedSpecies"].max(), 1) * 1.05
    most = max(int(curve["cameraDays"].max()), 1)

    def x(days: float) -> float:
        return left + days / most * inner_width

    def y(species: float) -> float:
        return top + inner_height * (1 - min(species, peak) / peak)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        (f'<rect x="{left}" y="{y(estimate["estimatedUpper"]):.1f}" width="{inner_width}" '
        f'height="{max(y(estimate["estimatedLower"]) - y(estimate["estimatedUpper"]), 0):.1f}" fill="#E69F00" '
        'fill-opacity="0.15"/>'),
        (f'<line x1="{left}" y1="{y(estimate["estimatedSpecies"]):.1f}" x2="{left + inner_width}" '
        f'y2="{y(estimate["estimatedSpecies"]):.1f}" stroke="#D55E00" stroke-dasharray="6 4"/>'),
        (f'<text x="{left + inner_width - 4}" y="{y(estimate["estimatedSpecies"]) - 6:.1f}" text-anchor="end" '
        f'fill="#D55E00">Chao2 estimate: {estimate["estimatedSpecies"]:g} species (shaded: 95% interval)</text>'),
    ]
    step = next((size for size in [1, 2, 5, 10, 20, 50, 100, 200] if peak / size <= 8), max(int(peak // 8), 1))
    for tick in range(0, int(peak) + 1, step):
        parts.append(f'<line x1="{left}" y1="{y(tick):.1f}" x2="{left + inner_width}" y2="{y(tick):.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick}</text>')
    day_step = next(
        (size for size in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000] if most / size <= 8), max(most // 8, 1)
    )
    for tick in range(0, most + 1, day_step):
        parts.append(f'<text x="{x(tick):.1f}" y="{height - 26}" text-anchor="middle">{tick}</text>')
    points = " ".join(f"{x(row.cameraDays):.1f},{y(row.expectedSpecies):.1f}" for row in curve.itertuples())
    parts += [
        f'<polyline points="{x(0):.1f},{y(0):.1f} {points}" fill="none" stroke="#0072B2" stroke-width="2.5"/>',
        f'<line x1="{left}" y1="{top + inner_height}" x2="{left + inner_width}" y2="{top + inner_height}" stroke="#444"/>',
        f'<text x="{left + inner_width / 2}" y="{height - 6}" text-anchor="middle">Camera-days</text>',
        f'<text transform="translate(14 {top + inner_height / 2}) rotate(-90)" text-anchor="middle">Species</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def run(events: pd.DataFrame, camera_days: int, out_dir: Path) -> dict | None:
    """Write the species accumulation curve (``richness.csv``) and its chart
    (``richness.svg``) and return the Chao2 estimate, or ``None`` without species."""
    counts = incidence(events)
    if counts.empty or camera_days <= 0:
        return None
    units = max(camera_days, int(counts.max()))
    curve = accumulation(counts, units)
    estimate = chao2(counts, units)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_dir / RICHNESS_FILE, index=False)
    (out_dir / PLOT_FILE).write_text(plot(curve, estimate), encoding="utf-8")
    return estimate
