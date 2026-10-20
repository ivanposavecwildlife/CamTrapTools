"""Compare the ecological results of different labels for the same survey.

Each source is a set of event labels: the machine's alone, the machine's with human
reviews, ground truth from a file of image labels, or another package such as one
whose detections used an adapted layer. ``ecology.analyse`` runs on every source,
and the results are compared with a reference source, so the effect of AI mistakes
on occupancy and activity can be measured.
"""

from pathlib import Path

import pandas as pd

from camtraptools import activity, ecology
from camtraptools.evaluate import NON_TAXON_LABELS, truth_from_file
from camtraptools.reviews import load_reviews

COMPARISON_FILE = "comparison.csv"
OVERLAP_COMPARISON_FILE = "overlap_comparison.csv"
SCORES_FILE = "scores.csv"
SCORE_COLUMNS = [
    "speciesMissed", "speciesAdded", "siteRecordsMissed", "siteRecordsAdded", "naiveOccupancyError",
    "occupancyError", "occupancySpecies", "activityDifference", "overlapError",
]


def truth_observations(package_dir: Path, truth_file: Path) -> pd.DataFrame:
    """Return the package's machine event-level observations relabelled with the truth
    file's labels (see ``evaluate.truth_from_file``). Events with no image in the
    truth file are left out."""
    observations = ecology.event_observations(package_dir, use_reviews=False)
    truth = truth_from_file(truth_file, package_dir)
    relabelled = observations[observations["eventID"].isin(truth.index)].copy()
    labels = relabelled["eventID"].map(truth)
    relabelled["observationType"] = labels.where(labels.isin(NON_TAXON_LABELS), "animal")
    relabelled["scientificName"] = labels.where(~labels.isin(NON_TAXON_LABELS))
    return relabelled


def sources(
    package_dir: Path, truth_file: Path | None = None, others: dict[str, Path] | None = None
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Return every source's animal events and deployments, by name: ``machine``,
    ``reviewed`` when the package has reviews, ``truth`` with a truth file, and each of
    ``others`` (with its own reviews)."""
    deployments = pd.read_csv(package_dir / "deployments.csv", dtype=str)
    found = {
        "machine": (
            ecology.animal_events(ecology.event_observations(package_dir, use_reviews=False), deployments),
            deployments,
        )
    }
    if not load_reviews(package_dir).empty:
        found["reviewed"] = (ecology.animal_events(ecology.event_observations(package_dir), deployments), deployments)
    if truth_file is not None:
        found["truth"] = (ecology.animal_events(truth_observations(package_dir, truth_file), deployments), deployments)
    for name, other in (others or {}).items():
        other_deployments = pd.read_csv(other / "deployments.csv", dtype=str)
        events = ecology.animal_events(ecology.event_observations(other), other_deployments)
        found[name] = (events, other_deployments)
    return found


def reference_name(names: list[str]) -> str:
    """Return the source the others are compared with: ``truth`` if there is one,
    otherwise ``reviewed``, otherwise the first."""
    return next((name for name in ["truth", "reviewed"] if name in names), names[0])


def species_sites(events: pd.DataFrame) -> set[tuple[str, str]]:
    """Return every (species, site) pair recorded in ``events``."""
    named = events[[ecology.is_species(name) for name in events["scientificName"]]]
    return set(zip(named["scientificName"], named["site"], strict=True))


def activity_difference(reference_events: pd.DataFrame, other_events: pd.DataFrame, min_events: int) -> float | None:
    """Return how different each species' daily activity is between two sources, as
    1 − Δ (the overlap of the two activity curves for that species), averaged over
    species with at least ``min_events`` events in both. 0 means identical patterns.
    Unlike the busiest time of day, it is stable when a species has two similar peaks."""
    differences = []
    for name in set(reference_events["scientificName"].dropna()) & set(other_events["scientificName"].dropna()):
        first = reference_events.loc[reference_events["scientificName"] == name, "eventStart"]
        second = other_events.loc[other_events["scientificName"] == name, "eventStart"]
        if ecology.is_species(name) and min(len(first), len(second)) >= min_events:
            differences.append(1 - activity.overlap(activity.to_radians(first), activity.to_radians(second)))
    return round(sum(differences) / len(differences), 3) if differences else None


def score(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    reference_events: pd.DataFrame,
    other_events: pd.DataFrame,
    reference_overlaps: pd.DataFrame,
    other_overlaps: pd.DataFrame,
    min_activity_events: int = activity.DEFAULT_MIN_EVENTS,
) -> dict:
    """Return how far one source's ecological results are from the reference's:
    species and (species, site) records it missed or added, the mean absolute
    difference in naive and modelled occupancy, how different each species' activity
    curve is, and the mean difference in overlaps between species, over the species or
    pairs both can estimate."""
    expected, found = species_sites(reference_events), species_sites(other_events)
    reference_species, other_species = {name for name, _ in expected}, {name for name, _ in found}
    naive = other["naiveOccupancy"].reindex(reference.index).fillna(0) if not other.empty else 0
    result = {
        "speciesMissed": len(reference_species - other_species),
        "speciesAdded": len(other_species - reference_species),
        "siteRecordsMissed": len(expected - found),
        "siteRecordsAdded": len(found - expected),
        "naiveOccupancyError": round(float((naive - reference["naiveOccupancy"]).abs().mean()), 3)
        if not reference.empty
        else None,
    }

    def reliable(table: pd.DataFrame) -> pd.Series:
        if "occupancy" not in table.columns:
            return pd.Series(dtype=float)
        return table.loc[table["occupancy"].notna() & table["warning"].isna(), "occupancy"]

    both = reliable(reference).index.intersection(reliable(other).index)
    result["occupancyError"] = (
        round(float((reliable(other)[both] - reliable(reference)[both]).abs().mean()), 3) if len(both) else None
    )
    result["occupancySpecies"] = len(both)
    result["activityDifference"] = activity_difference(reference_events, other_events, min_activity_events)
    pairs = reference_overlaps.merge(other_overlaps, on=["first", "second"], suffixes=("", "Other"))
    result["overlapError"] = (
        round(float((pairs["overlap"] - pairs["overlapOther"]).abs().mean()), 3) if not pairs.empty else None
    )
    return result


def pair_key(table: pd.DataFrame) -> pd.DataFrame:
    """Return overlaps with each pair's species in alphabetical order, so the same pair
    matches between sources."""
    ordered = table.copy()
    swap = ordered["first"] > ordered["second"]
    ordered.loc[swap, ["first", "second"]] = ordered.loc[swap, ["second", "first"]].to_numpy()
    return ordered


def run(
    package_dir: Path,
    out_dir: Path,
    truth_file: Path | None = None,
    others: dict[str, Path] | None = None,
    occasion_days: int = ecology.DEFAULT_OCCASION_DAYS,
    min_activity_events: int = activity.DEFAULT_MIN_EVENTS,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the ecology analyses for every source in ``out_dir/<source>/`` and compare
    them with the reference source.

    Writes and returns the reference's name, the results per taxon and source
    (``comparison.csv``), the overlaps per pair and source (``overlap_comparison.csv``)
    and each source's scores against the reference (``scores.csv``).
    """
    found = sources(package_dir, truth_file, others)
    results, overlaps = {}, {}
    for name, (events, deployments) in found.items():
        summary, _, pairs = ecology.analyse(
            events, deployments, out_dir / name, occasion_days, min_activity_events=min_activity_events
        )
        results[name], overlaps[name] = summary, pair_key(pairs)
    reference = reference_name(list(found))
    scores = pd.DataFrame(
        [
            {
                "source": name,
                **score(
                    results[reference], results[name], found[reference][0], found[name][0],
                    overlaps[reference], overlaps[name], min_activity_events,
                ),
            }
            for name in found
            if name != reference
        ],
        columns=["source", *SCORE_COLUMNS],
    ).set_index("source")
    comparison = pd.concat(
        [table.assign(source=name) for name, table in results.items() if not table.empty]
    ).rename_axis("taxon").reset_index()
    overlap_table = pd.concat([table.assign(source=name) for name, table in overlaps.items()], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(out_dir / COMPARISON_FILE, index=False)
    overlap_table.to_csv(out_dir / OVERLAP_COMPARISON_FILE, index=False)
    scores.to_csv(out_dir / SCORES_FILE)
    return reference, comparison, overlap_table, scores
