"""Does the review plan fix the results with less review, and does the toolkit know
how many sites are still wrong? A test against reference labels.

The reviewer is simulated with the reference labels, so review is assumed to be
correct. The results from the reference labels for every event are the target. For
each review budget (a share of the events), two strategies are compared:

- least confident first: the events the AI was least sure about;
- planned: the least confident 2% first, to calibrate the AI, then the review plan,
  recomputed after every quarter of the budget.

Finally the plan is followed to the end: planned events are reviewed until the plan
is empty, to see how much review the plan itself asks for.

Each strategy's results are scored by the sites that disagree with the reference
(false and missed records, over all species) next to the number ``ecology
--ai-errors`` expected; by how far occupancy is from the reference for species the
reference estimates reliably (no warning); and by how many species' 95% intervals
contain the reference occupancy.

Usage, with a package from ``detect`` and ``events`` and a truth file of image labels:

    uv run python examples/planned_review.py orinoquia/package orinoquia/truth.csv orinoquia/planned

Writes ``scores.csv`` and per-species occupancy in ``species.csv`` to the output
folder, and prints the scores.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from camtraptools import ecology, plan, reviews, uncertainty
from camtraptools.evaluate import NON_TAXON_LABELS, truth_from_file

BUDGETS = [0.05, 0.1, 0.2, 0.3]
CALIBRATION_SHARE = 0.02
ROUNDS = 4


def review_events(package: Path, event_ids: list[str], truth: pd.Series) -> None:
    """Write reviews of ``event_ids`` with their reference labels."""
    labels = truth.reindex(event_ids).dropna()
    animal = ~labels.isin(NON_TAXON_LABELS)
    decisions = pd.DataFrame(
        {
            "eventID": labels.index,
            "observationType": labels.where(~animal, "animal").to_numpy(),
            "scientificName": labels.where(animal).to_numpy(),
            "count": pd.Series("1", index=labels.index).where(animal | labels.isin(reviews.COUNTED_TYPES)).to_numpy(),
            "reviewer": "reference labels",
            "reviewedAt": pd.Timestamp("2026-10-21T12:00:00+02:00").isoformat(),
        }
    )
    decisions.to_csv(package / reviews.REVIEWS_FILE, index=False)
    reviews.apply_reviews(package)


def least_confident(package: Path, count: int, done: list[str]) -> list[str]:
    """Return ``done`` and then the least confident other events, ``count`` in all."""
    observations = pd.read_csv(package / "observations.csv", dtype=str)
    machine = observations[observations["classificationMethod"] != "human"]
    ordered = reviews.order_events(machine[machine["observationLevel"] == "event"], machine, "uncertainty")
    rest = [event for event in ordered["eventID"] if event not in set(done)]
    return (done + rest)[:count]


def planned(package: Path, count: int, truth: pd.Series) -> list[str]:
    """Return the events reviewed by the planned strategy with a budget of ``count``."""
    chosen = least_confident(package, max(1, round(CALIBRATION_SHARE * len(truth))), [])
    for round_number in range(1, ROUNDS + 1):
        review_events(package, chosen, truth)
        size = round(count * round_number / ROUNDS)
        _, events, _ = plan.run(package)
        additions = [event for event in events["eventID"] if event not in set(chosen)]
        chosen = (chosen + additions)[:size]
        if len(chosen) < size:
            chosen = least_confident(package, size, chosen)
    return chosen[:count]


def until_stable(package: Path, truth: pd.DataFrame) -> list[str]:
    """Return the events reviewed by following the plan to the end: the least
    confident 2%, then every planned event, planning again until the plan is empty
    (at most ``ROUNDS`` times)."""
    chosen = least_confident(package, max(1, round(CALIBRATION_SHARE * len(truth))), [])
    for _ in range(ROUNDS):
        review_events(package, chosen, truth)
        _, events, _ = plan.run(package)
        additions = [event for event in events["eventID"] if event not in set(chosen)]
        if not additions:
            break
        chosen += additions
    return chosen


def results(package: Path, folder: Path) -> pd.DataFrame:
    """Return each species' sites and occupancy, as is and allowing for AI mistakes."""
    reviewed = len(reviews.load_reviews(package))
    summary, _, _ = ecology.run(package, folder, ai_errors=reviewed >= uncertainty.MIN_REVIEWS)
    histories = folder / ecology.HISTORIES_FOLDER
    sites = {
        name: set(pd.read_csv(histories / ecology.file_name(name), index_col="site").eq(1).any(axis=1).loc[lambda x: x].index)
        for name in summary.index
    }
    summary["sites"] = pd.Series(sites)
    return summary


def score(found: pd.DataFrame, reference: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return how far ``found`` is from the ``reference`` results, overall and for
    each species the reference estimates."""
    wrong_sites = 0
    for name in set(found.index) | set(reference.index):
        mine = found["sites"].get(name, set()) if name in found.index else set()
        true = reference["sites"].get(name, set()) if name in reference.index else set()
        wrong_sites += len(mine ^ true)
    doubt = found.reindex(columns=["expectedFalseSites", "expectedMissedSites"])
    predicted = float(doubt.sum().sum()) if doubt.notna().any().any() else float("nan")
    modelled = reference.dropna(subset=["occupancy"])
    found = found.reindex(modelled.index)
    details = pd.DataFrame(
        {
            "reference": modelled["occupancy"],
            "reliable": modelled["warning"].isna(),
            "found": found["occupancy"].fillna(found["naiveOccupancy"]).fillna(0.0),
            "covered": modelled["occupancy"].between(found["occupancyLower"], found["occupancyUpper"]),
        }
    )
    reliable = details[details["reliable"]]
    return {
        "wrongSites": wrong_sites,
        "predictedWrongSites": round(predicted, 1),
        "occupancyError": round(float((reliable["found"] - reliable["reference"]).abs().mean()), 3),
        "covered": f"{int(details['covered'].sum())}/{len(details)}",
    }, details


def main() -> None:
    package, truth_file, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    truth = truth_from_file(truth_file, package)
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder) / "package"
        shutil.copytree(package, work, ignore=shutil.ignore_patterns(reviews.REVIEWS_FILE, "review_plan*.csv"))
        reviews.apply_reviews(work)

        review_events(work, list(truth.index), truth)
        reference = results(work, Path(folder) / "reference")
        rows = []
        review_events(work, [], truth)
        summary, details = score(results(work, Path(folder) / "none"), reference)
        rows.append({"budget": 0.0, "strategy": "none", "reviewed": 0, **summary})
        all_details = [details.assign(budget=0.0, strategy="none")]
        for budget in BUDGETS:
            count = round(budget * len(truth))
            for strategy in ["least confident", "planned"]:
                chosen = least_confident(work, count, []) if strategy == "least confident" else planned(work, count, truth)
                review_events(work, chosen, truth)
                summary, details = score(results(work, Path(folder) / f"{strategy}-{budget}"), reference)
                rows.append({"budget": budget, "strategy": strategy, "reviewed": len(chosen), **summary})
                all_details.append(details.assign(budget=budget, strategy=strategy))
                print(rows[-1], flush=True)
        chosen = until_stable(work, truth)
        review_events(work, chosen, truth)
        summary, details = score(results(work, Path(folder) / "until-stable"), reference)
        rows.append({"budget": round(len(chosen) / len(truth), 2), "strategy": "plan until stable", "reviewed": len(chosen), **summary})
        all_details.append(details.assign(budget=rows[-1]["budget"], strategy="plan until stable"))
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "scores.csv", index=False)
    pd.concat(all_details).rename_axis("taxon").to_csv(out / "species.csv")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
