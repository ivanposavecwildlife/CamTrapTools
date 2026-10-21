"""Plan the review: which events to check, and how many, until each species' results
no longer depend on the AI's mistakes.

The results of ``ecology`` rest on each species' detection history: whether it was
recorded at each site in each occasion. With the AI's calibrated mistakes (see
``uncertainty``), every cell of the history has a chance of being wrong: a record
that is truly another animal, or a species hidden in an event labelled as something
else. Their sum is the number of cells in doubt. Reviewing a cell's events settles it,
so cells are planned in order of doubt removed per event reviewed, until the doubt
left is at most ``tolerance`` times the species' records, or a quarter of a cell for
species with few records (a better than three in four chance that no cell is wrong).
Rare species therefore need their uncertain records checked, and common species with
confident records need none. Cells where reviewing an event has less than
``min_gain`` chance of correcting a record are left out: doubt spread thinly over
many events, each unlikely to change anything, is not worth the reviewer's time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from camtraptools import ecology, uncertainty
from camtraptools.reviews import load_reviews, main_reviews, review_speed

DEFAULT_TOLERANCE = 0.05
MIN_DOUBT_LEFT = 0.25
DEFAULT_MIN_GAIN = 0.05
MISSED_SITES_TO_LIST = 0.5
PLAN_FILE = "review_plan.csv"
SPECIES_FILE = "review_plan_species.csv"


def taxon_plan(
    chances: uncertainty.LabelChances,
    taxon: str,
    effort: pd.DataFrame,
    occasion_days: int,
    tolerance: float = DEFAULT_TOLERANCE,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> tuple[dict, pd.DataFrame]:
    """Return one taxon's doubt and the cells to review for it.

    The first result has the taxon's records (cells where the results have it), the
    cells in doubt, the expected false and missed sites, how many events to review,
    the doubt that is left afterwards, and whether that doubt is above the target but
    spread too thinly for any event to be worth reviewing (``spreadThin``). The second has one row per cell to review, in
    order, with its doubt per event (``priority``) and its unreviewed events.
    """
    events = chances.events
    rows, columns = uncertainty.cell_positions(events, effort, occasion_days)
    certainty = uncertainty.cell_certainty(chances, taxon, effort, occasion_days)
    chance = chances.probabilities[taxon] if taxon in chances.probabilities else pd.Series(0.0, index=events.index)
    cells = pd.DataFrame(
        {
            "row": rows,
            "column": columns,
            "eventID": events["eventID"],
            "reported": (events["used"] == taxon) | events["also"].map(lambda names: taxon in names),
            "open": events["truth"].isna() & ((events["used"] == taxon) | (chance > 0)),
        }
    )
    cells = cells[cells["row"] >= 0]
    by_cell = cells.groupby(["row", "column"]).agg(reported=("reported", "any"), events=("open", "sum"))
    by_cell["ids"] = cells[cells["open"]].groupby(["row", "column"])["eventID"].agg(list)
    values = certainty.to_numpy()
    truly = values[by_cell.index.get_level_values("row"), by_cell.index.get_level_values("column")]
    by_cell["doubt"] = np.where(by_cell["reported"], 1 - np.nan_to_num(truly), np.nan_to_num(truly))
    records = int(by_cell["reported"].sum())

    reported_cells = by_cell.index[by_cell["reported"]]
    history = np.zeros(effort.shape, dtype=int)
    history[reported_cells.get_level_values("row"), reported_cells.get_level_values("column")] = 1
    history = pd.DataFrame(history, index=effort.index, columns=effort.columns)
    false_sites, missed_sites = uncertainty.site_doubt(history, certainty)

    to_review = by_cell[(by_cell["doubt"] > 0) & (by_cell["events"] > 0)].copy()
    to_review["priority"] = to_review["doubt"] / to_review["events"]
    to_review = to_review[to_review["priority"] >= min_gain].sort_values("priority", ascending=False)
    total = float(by_cell["doubt"].sum())
    target = max(tolerance * records, MIN_DOUBT_LEFT)
    left_before = total - to_review["doubt"].cumsum().shift(fill_value=0.0)
    chosen = to_review[left_before > target]
    left = total - float(chosen["doubt"].sum())
    summary = {
        "taxon": taxon,
        "records": records,
        "inDoubt": round(total, 2),
        "expectedFalseSites": round(false_sites, 1),
        "expectedMissedSites": round(missed_sites, 1),
        "toReview": len({event for ids in chosen["ids"] for event in ids}),
        "leftInDoubt": round(left, 2),
        "spreadThin": bool(chosen.empty and left > target),
    }
    return summary, chosen.assign(taxon=taxon)[["taxon", "priority", "doubt", "ids"]]


def build(
    chances: uncertainty.LabelChances,
    effort: pd.DataFrame,
    occasion_days: int = ecology.DEFAULT_OCCASION_DAYS,
    taxa: list[str] | None = None,
    all_taxa: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the review plan: per taxon, its doubt and the events to review, and
    one row per event in the order to review them, with the taxa it matters for.

    By default the taxa are every species the results record, and every species the
    AI may have missed at ``MISSED_SITES_TO_LIST`` sites or more; ``all_taxa`` adds
    higher taxa, and ``taxa`` chooses taxa by name.
    """
    if taxa:
        candidates = list(taxa)
    else:
        candidates = sorted(name for name in chances.probabilities.columns if all_taxa or ecology.is_species(name))
        candidates = [name for name in candidates if name not in uncertainty.NON_TAXON_LABELS and name != "animal"]
    rows, plans = [], []
    recorded = set(chances.events["used"])
    for taxon in candidates:
        summary, cells = taxon_plan(chances, taxon, effort, occasion_days, tolerance, min_gain)
        if taxon in recorded or summary["expectedMissedSites"] >= MISSED_SITES_TO_LIST or taxa:
            rows.append(summary)
            plans.append(cells)
    species = pd.DataFrame(
        rows,
        columns=[
            "taxon", "records", "inDoubt", "expectedFalseSites", "expectedMissedSites", "toReview", "leftInDoubt",
            "spreadThin",
        ],
    ).set_index("taxon")
    species = species.sort_values(["toReview", "records"], ascending=[False, False])
    cells = pd.concat(plans, ignore_index=True) if plans else pd.DataFrame(columns=["taxon", "priority", "doubt", "ids"])
    cells = cells.sort_values("priority", ascending=False, kind="stable")
    order: dict[str, dict] = {}
    for cell in cells.itertuples():
        for event_id in cell.ids:
            entry = order.setdefault(event_id, {"eventID": event_id, "taxa": [], "doubt": 0.0})
            if cell.taxon not in entry["taxa"]:
                entry["taxa"].append(cell.taxon)
            entry["doubt"] += cell.priority
    details = chances.events.set_index("eventID")[["site", "ai", "confidence"]]
    events = pd.DataFrame(list(order.values()), columns=["eventID", "taxa", "doubt"])
    events["taxa"] = events["taxa"].map("; ".join)
    events["doubt"] = events["doubt"].round(3)
    events = events.join(details, on="eventID")
    events.insert(0, "order", range(1, len(events) + 1))
    return species, events[["order", "eventID", "site", "ai", "confidence", "taxa", "doubt"]]


def run(
    package_dir: Path,
    occasion_days: int = ecology.DEFAULT_OCCASION_DAYS,
    taxa: list[str] | None = None,
    all_taxa: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> tuple[pd.DataFrame, pd.DataFrame, uncertainty.LabelChances]:
    """Plan the review of the package, write ``review_plan.csv`` (the events in order)
    and ``review_plan_species.csv`` (each taxon's doubt) to it, and return both and
    the label chances they rest on."""
    deployments = pd.read_csv(package_dir / "deployments.csv", dtype=str)
    effort = ecology.effort_matrix(deployments, occasion_days)
    chances = uncertainty.load(package_dir, deployments)
    species, events = build(chances, effort, occasion_days, taxa, all_taxa, tolerance, min_gain)
    species.to_csv(package_dir / SPECIES_FILE)
    events.to_csv(package_dir / PLAN_FILE, index=False)
    return species, events, chances


def minutes_needed(package_dir: Path, events: int) -> float | None:
    """Return how many minutes reviewing ``events`` events should take at the
    reviewer's speed so far, or ``None`` if the speed is not known yet."""
    speed = review_speed(main_reviews(load_reviews(package_dir))["reviewedAt"])
    return events / speed["perMinute"] if speed["perMinute"] else None
