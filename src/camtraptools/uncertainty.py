"""How likely each event's label is to be wrong, learned from the reviews.

Reviewed events show how often the AI is right at each confidence level. A logistic
calibration, logit P(right) = a + b · logit(confidence) + u(label), is fitted to them
with a prior that takes the AI's confidence at face value (a = 0, b = 1, u = 0), so a
few reviews move it only a little. The offset u lets labels the AI gets wrong more
often than its confidence says (or less often) differ from the rest. Because it is
conditional on the label and the confidence, it stays fair when the reviewer chose
events by them, such as the least confident first or the review plan. What an animal really was is estimated from
the reviewers' decisions on events with the same AI label: when the label was right
(which species an ``animal`` or a family turned out to be), shrunk towards how often
each species was recorded, and when it was wrong, shrunk towards the corrections of
all labels.

Every event then has a probability for each possible true label: certain for reviewed
events, and otherwise the chance that its label is right plus a share of the chance
that it is wrong. Each cell of a taxon's detection history (a site and occasion) gets
the probability that the taxon was truly recorded there. The review planner (``plan``)
and the sites in doubt of ``ecology`` use these.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from camtraptools.crops import location_of
from camtraptools.detect import OBSERVATIONS_FILE
from camtraptools.ecology import is_species, sigmoid
from camtraptools.evaluate import (
    NON_TAXON_LABELS,
    RIGHT_OUTCOMES,
    ancestry_from_results,
    event_label,
    outcome,
    truth_from_reviews,
)
from camtraptools.reviews import order_events

CONFIDENCE_LIMITS = (0.005, 0.995)
PRIOR_SD = 1.0
LABEL_SD = 1.0
CORRECTION_PRIOR = 2.0
MIN_REVIEWS = 20


@dataclass
class Calibration:
    """How the AI's confidence relates to the chance its label is right: an intercept,
    a slope and an offset per label (0 for labels without reviews)."""

    intercept: float = 0.0
    slope: float = 1.0
    offsets: dict[str, float] = field(default_factory=dict)


@dataclass
class LabelChances:
    """The events of a package with the probability of each true label.

    ``events`` has one row per event with its site, day, the AI's label and confidence,
    the reviewer's label (missing if not reviewed) and the label the results use.
    ``probabilities`` has the same rows and one column per possible true label.
    ``calibration`` is fitted to the ``reviewed`` reviewed events.
    """

    events: pd.DataFrame
    probabilities: pd.DataFrame
    calibration: Calibration
    reviewed: int


def logit(probability):
    """Return the logit of ``probability``, kept within ``CONFIDENCE_LIMITS``."""
    probability = np.clip(np.asarray(probability, dtype=float), *CONFIDENCE_LIMITS)
    return np.log(probability / (1 - probability))


def event_table(package_dir: Path, deployments: pd.DataFrame) -> pd.DataFrame:
    """Return one row per event with its site, day, the AI's label and confidence,
    the reviewer's label (missing if the event was not reviewed) and the label the
    results use (the reviewer's where there is one)."""
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    machine = observations[observations["classificationMethod"] != "human"]
    events = order_events(machine[machine["observationLevel"] == "event"], machine, "time")
    table = pd.DataFrame(
        {
            "eventID": events["eventID"],
            "site": events["deploymentID"].map(location_of(deployments)),
            "day": pd.to_datetime(events["eventStart"], format="ISO8601", errors="coerce").dt.date,
            "ai": [event_label(row.observationType, row.scientificName) for row in events.itertuples()],
            "confidence": events["confidence"].astype(float),
        }
    )
    table["truth"] = table["eventID"].map(truth_from_reviews(package_dir))
    table["used"] = table["truth"].fillna(table["ai"])
    return table.dropna(subset=["day", "site"]).reset_index(drop=True)


def is_right(ai: str, truth: str, ancestry: dict[str, set[str]]) -> bool:
    """Return whether the AI's label holds for the reviewer's: the same label, a
    higher taxon of it, a more specific one, or ``animal`` for any animal."""
    if ai == "animal":
        return truth not in NON_TAXON_LABELS
    return outcome(truth, ai, ancestry) in RIGHT_OUTCOMES


def calibrate(confidence: np.ndarray, labels: np.ndarray, right: np.ndarray) -> Calibration:
    """Return the calibration logit P(right) = a + b · logit(confidence) + u(label)
    fitted to reviewed events, with normal priors around a = 0, b = 1 (sd
    ``PRIOR_SD``) and u = 0 (sd ``LABEL_SD``): penalized logistic regression, solved
    by Newton's method with step halving."""
    names = sorted(set(labels))
    design = np.column_stack(
        [np.ones(len(confidence)), logit(confidence), *(np.asarray(labels) == name for name in names)]
    ).astype(float)
    prior_mean = np.zeros(design.shape[1])
    prior_mean[1] = 1.0
    precision = np.diag([PRIOR_SD**-2] * 2 + [LABEL_SD**-2] * len(names))
    y = np.asarray(right, dtype=float)

    def objective(beta):
        z = design @ beta
        return float((y * z - np.logaddexp(0, z)).sum() - (beta - prior_mean) @ precision @ (beta - prior_mean) / 2)

    beta = prior_mean.copy()
    for _ in range(100):
        chance = sigmoid(design @ beta)
        gradient = design.T @ (y - chance) - precision @ (beta - prior_mean)
        hessian = design.T @ (design * (chance * (1 - chance))[:, None]) + precision
        step = np.linalg.solve(hessian, gradient)
        while objective(beta + step) < objective(beta) and np.max(np.abs(step)) > 1e-10:
            step = step / 2
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return Calibration(float(beta[0]), float(beta[1]), dict(zip(names, map(float, beta[2:]), strict=True)))


def chance_right(confidence, labels, calibration: Calibration) -> np.ndarray:
    """Return the calibrated chance that ``labels`` given with ``confidence`` are right."""
    offsets = np.array([calibration.offsets.get(label, 0.0) for label in labels])
    return sigmoid(calibration.intercept + calibration.slope * logit(confidence) + offsets)


def right_shares(label: str, species_counts: pd.Series, ancestry: dict[str, set[str]]) -> dict[str, float]:
    """Return what an event labelled ``label`` truly is when the label is right: the
    label itself for a species or a non-animal label, and for a higher taxon or
    ``animal`` the species within it, in proportion to how often each was recorded."""
    if label != "animal" and (is_species(label) or label in NON_TAXON_LABELS):
        return {label: 1.0}
    within = {
        name: count
        for name, count in species_counts.items()
        if label == "animal" or label in ancestry.get(name, set())
    }
    total = sum(within.values())
    return {name: count / total for name, count in within.items()} if total else {label: 1.0}


def blend(prior: dict[str, float], counts: pd.Series) -> dict[str, float]:
    """Return the shares in ``counts`` (reviewed labels) with ``prior`` added as
    ``CORRECTION_PRIOR`` events."""
    names = sorted(set(prior) | set(counts.index))
    shares = pd.Series(prior).reindex(names, fill_value=0.0) * CORRECTION_PRIOR + counts.reindex(names, fill_value=0)
    return (shares / shares.sum()).to_dict()


def wrong_shares(label: str, corrections: pd.DataFrame, overall: pd.Series) -> dict[str, float]:
    """Return what an event labelled ``label`` truly is when the label is wrong: the
    reviewers' corrections of that label, with the corrections of all labels added
    as ``CORRECTION_PRIOR`` events. Empty if nothing was ever corrected."""
    counts = corrections.loc[corrections["ai"] == label, "truth"].value_counts()
    shares = (counts.reindex(overall.index, fill_value=0) + CORRECTION_PRIOR * overall) / (counts.sum() + CORRECTION_PRIOR)
    shares = shares.drop(label, errors="ignore")
    return (shares / shares.sum()).to_dict() if shares.sum() > 0 else {}


def label_chances(table: pd.DataFrame, ancestry: dict[str, set[str]]) -> LabelChances:
    """Return the probability of each true label for every event in ``table`` (from
    ``event_table``), using the reviewed events to calibrate the AI."""
    reviewed = table[table["truth"].notna()]
    right = np.array([is_right(row.ai, row.truth, ancestry) for row in reviewed.itertuples()], dtype=bool)
    calibration = calibrate(reviewed["confidence"].to_numpy(), reviewed["ai"].to_numpy(), right)
    confirmed, corrections = reviewed[right], reviewed[~right]
    overall = corrections["truth"].value_counts(normalize=True)
    species_counts = table.loc[table["used"].map(is_species), "used"].value_counts()

    labels = sorted(table["ai"].unique())
    right_by_label = {
        label: blend(
            right_shares(label, species_counts, ancestry), confirmed.loc[confirmed["ai"] == label, "truth"].value_counts()
        )
        for label in labels
    }
    wrong_by_label = {label: wrong_shares(label, corrections, overall) for label in labels}
    columns = sorted(
        set(table["used"])
        | {name for shares in right_by_label.values() for name in shares}
        | {name for shares in wrong_by_label.values() for name in shares}
    )
    when_right = pd.DataFrame([right_by_label[label] for label in labels], index=labels, columns=columns).fillna(0.0)
    when_wrong = pd.DataFrame([wrong_by_label[label] for label in labels], index=labels, columns=columns).fillna(0.0)

    chance = chance_right(table["confidence"].to_numpy(), table["ai"], calibration)[:, None]
    probabilities = (
        when_right.loc[table["ai"]].to_numpy() * chance + when_wrong.loc[table["ai"]].to_numpy() * (1 - chance)
    )
    probabilities = pd.DataFrame(probabilities, index=table.index, columns=columns)
    known = table["truth"].notna()
    probabilities.loc[known] = 0.0
    for name, rows in table[known].groupby("truth").groups.items():
        probabilities.loc[rows, name] = 1.0
    return LabelChances(table, probabilities, calibration, len(reviewed))


def load(package_dir: Path, deployments: pd.DataFrame) -> LabelChances:
    """Return the label chances of the package's events, calibrated on its reviews."""
    return label_chances(event_table(package_dir, deployments), ancestry_from_results(package_dir))


def cell_positions(events: pd.DataFrame, effort: pd.DataFrame, occasion_days: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the row (site) and column (occasion) of each event in the detection
    histories, -1 for events outside them."""
    rows = effort.index.get_indexer(events["site"])
    first = date.fromisoformat(effort.columns[0])
    columns = np.array([(day - first).days // occasion_days for day in events["day"]])
    outside = (rows < 0) | (columns < 0) | (columns >= len(effort.columns))
    return np.where(outside, -1, rows), np.where(outside, -1, columns)


def cell_certainty(
    chances: LabelChances, taxon: str, effort: pd.DataFrame, occasion_days: int
) -> pd.DataFrame:
    """Return, for each site and occasion, the probability that ``taxon`` was truly
    recorded: one minus the product over the cell's events of the chance that each
    was not the taxon. Cells without events are 0 and cells without a running camera
    are missing, as in ``ecology.detection_history``."""
    rows, columns = cell_positions(chances.events, effort, occasion_days)
    inside = rows >= 0
    chance = chances.probabilities[taxon].to_numpy() if taxon in chances.probabilities else np.zeros(len(rows))
    with np.errstate(divide="ignore"):
        log_not = np.log1p(-np.clip(chance[inside], 0.0, 1.0))
    sums = np.zeros(effort.shape)
    np.add.at(sums, (rows[inside], columns[inside]), log_not)
    has_events = np.zeros(effort.shape, dtype=bool)
    has_events[rows[inside], columns[inside]] = True
    truly = -np.expm1(sums)
    certainty = np.where(effort.to_numpy() > 0, 0.0, np.nan)
    certainty = np.where(has_events & (truly > 0), truly, certainty)
    return pd.DataFrame(certainty, index=effort.index, columns=effort.columns)


def site_doubt(history: pd.DataFrame, certainty: pd.DataFrame) -> tuple[float, float]:
    """Return the expected number of sites where the reported results are wrong
    about ``history``'s taxon: recorded but not truly there (false), and not recorded
    although it truly was (missed)."""
    reported = history.eq(1).any(axis=1).to_numpy()
    with np.errstate(divide="ignore"):
        log_not = np.log1p(-np.nan_to_num(certainty.to_numpy(dtype=float)).clip(0, 1)).sum(axis=1)
    truly = -np.expm1(log_not)
    return float((1 - truly)[reported].sum()), float(truly[~reported].sum())
