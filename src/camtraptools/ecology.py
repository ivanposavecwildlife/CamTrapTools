"""Turn event-level observations into detection histories and naive occupancy.

A detection history has one row per camera site and one column per sampling occasion
(a block of days): 1 if the species was recorded there during the occasion, 0 if the
camera was running but did not record it, and empty if the camera was not running.
Naive occupancy is the share of surveyed sites where a species was recorded at least
once. It ignores that species are missed where they are present; the occupancy model
(MacKenzie et al. 2002) corrects for this by estimating the probability that a site is
occupied together with the probability of recording the species where it is.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from camtraptools import activity, richness
from camtraptools.crops import location_of
from camtraptools.detect import OBSERVATIONS_FILE
from camtraptools.reviews import prefer_human

DEFAULT_OCCASION_DAYS = 7
HISTORIES_FOLDER = "histories"
EFFORT_FILE = "effort.csv"
SUMMARY_FILE = "occupancy.csv"
RAI_FILE = "rai.csv"
LOGIT_LIMITS = (-15.0, 15.0)
MIN_SITES_FOR_MODEL = 3
AT_LIMIT = 0.99
WIDE_INTERVAL = 0.5
CHI_SQUARE_95 = 3.841459
SITES_IN_DOUBT_TO_FLAG = 1.0


def is_species(name) -> bool:
    """Return whether a scientific name is a species (two words) rather than a
    higher taxon such as ``Bovidae``."""
    return isinstance(name, str) and len(name.split()) == 2


def active_days(deployments: pd.DataFrame) -> dict[str, set[date]]:
    """Return the calendar days each site's cameras were running, by site. Partial
    first and last days count, and deployments at the same site are combined."""
    sites = location_of(deployments)
    days: dict[str, set[date]] = {}
    for row in deployments.itertuples():
        start = pd.to_datetime(row.deploymentStart, format="ISO8601")
        end = pd.to_datetime(row.deploymentEnd, format="ISO8601")
        site_days = days.setdefault(sites[row.deploymentID], set())
        if pd.isna(start) or pd.isna(end):
            continue
        site_days.update((start + timedelta(days=offset)).date() for offset in range((end.date() - start.date()).days + 1))
    return days


def occasion_starts(first: date, last: date, occasion_days: int) -> list[date]:
    """Return the first day of every occasion from ``first`` to ``last``."""
    count = (last - first).days // occasion_days + 1
    return [first + timedelta(days=occasion_days * number) for number in range(count)]


def effort_matrix(deployments: pd.DataFrame, occasion_days: int = DEFAULT_OCCASION_DAYS) -> pd.DataFrame:
    """Return the number of days each site's cameras were running in each occasion,
    with sites as rows and occasion start dates as columns. Occasions start on the
    first day any camera was running."""
    days = active_days(deployments)
    every_day = set().union(*days.values())
    if not every_day:
        raise ValueError("no deployment has both a start and an end")
    starts = occasion_starts(min(every_day), max(every_day), occasion_days)
    effort = pd.DataFrame(0, index=sorted(days), columns=[start.isoformat() for start in starts])
    for site, site_days in days.items():
        for day in site_days:
            number = (day - starts[0]).days // occasion_days
            effort.iloc[effort.index.get_loc(site), number] += 1
    effort.index.name = "site"
    return effort


def event_observations(package_dir: Path, use_reviews: bool = True) -> pd.DataFrame:
    """Return the package's event-level observations: the reviewer's decision wherever
    an event was reviewed, or with ``use_reviews`` off, the machine's for every event."""
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    if use_reviews:
        observations = prefer_human(observations)
    else:
        observations = observations[observations["classificationMethod"] != "human"]
    return observations[observations["observationLevel"] == "event"]


def animal_events(observations: pd.DataFrame, deployments: pd.DataFrame) -> pd.DataFrame:
    """Return every animal event among event-level ``observations`` with its site,
    day, start time and scientific name."""
    events = observations[observations["observationType"] == "animal"].copy()
    events["site"] = events["deploymentID"].map(location_of(deployments))
    events["day"] = pd.to_datetime(events["eventStart"], format="ISO8601", errors="coerce").dt.date
    return events.dropna(subset=["day"])[["eventID", "site", "day", "eventStart", "scientificName"]]


def detection_history(events: pd.DataFrame, effort: pd.DataFrame, occasion_days: int) -> pd.DataFrame:
    """Return one species' detection history: 1 where ``events`` recorded it, 0 where
    cameras ran without recording it, and missing where no camera was running. An
    occasion with a record counts as surveyed even if the recorded period missed it."""
    history = pd.DataFrame(0.0, index=effort.index, columns=effort.columns).where(effort.gt(0))
    first = date.fromisoformat(effort.columns[0])
    for event in events.itertuples():
        number = (event.day - first).days // occasion_days
        if event.site in history.index and 0 <= number < len(history.columns):
            history.iloc[history.index.get_loc(event.site), number] = 1
    return history.astype("Int64")


def summarize(histories: dict[str, pd.DataFrame], events: pd.DataFrame) -> pd.DataFrame:
    """Return, per taxon, its events, the sites it was recorded at out of the sites
    surveyed, naive occupancy, and how often it was recorded per occasion surveyed at
    those sites (a first hint of its detection probability)."""
    rows = []
    for name, history in histories.items():
        surveyed = history.notna().any(axis=1)
        detected = history.eq(1).any(axis=1)
        at_detected = history[detected]
        rows.append(
            {
                "taxon": name,
                "events": int((events["scientificName"] == name).sum()),
                "sitesDetected": int(detected.sum()),
                "sitesSurveyed": int(surveyed.sum()),
                "naiveOccupancy": round(detected.sum() / surveyed.sum(), 2) if surveyed.any() else None,
                "detectionRate": round(at_detected.eq(1).sum().sum() / at_detected.notna().sum().sum(), 2)
                if detected.any()
                else None,
            }
        )
    return pd.DataFrame(rows).set_index("taxon").sort_values(["naiveOccupancy", "events"], ascending=False)


def relative_abundance(events: pd.DataFrame, effort: pd.DataFrame, taxa: list[str]) -> pd.DataFrame:
    """Return the relative abundance index (RAI) of each taxon at each site: its
    independent events per 100 camera-days. Sites are rows and taxa columns, with a
    last row for all sites together. RAI compares how often species are recorded; it
    is not a density, because species differ in how easily cameras record them."""
    camera_days = effort.sum(axis=1)
    counts = pd.crosstab(events["site"], events["scientificName"]).reindex(index=effort.index, columns=taxa, fill_value=0)
    rai = counts.div(camera_days.where(camera_days > 0), axis=0) * 100
    rai.loc["all sites"] = counts.sum() / camera_days.sum() * 100 if camera_days.sum() else float("nan")
    rai.index.name = "site"
    return rai.round(2)


def sigmoid(value):
    """Return the inverse logit of ``value``."""
    return 1 / (1 + np.exp(-value))


def log_likelihood(parameters, detections: np.ndarray, days: np.ndarray) -> float:
    """Return the log-likelihood of the single-season occupancy model.

    ``parameters`` are the logits of occupancy (ψ) and of the daily detection
    probability (q). ``detections`` holds each site's history (1, 0, or NaN where no
    camera ran) and ``days`` the camera days of each occasion, so an occasion's
    detection probability is 1 − (1 − q)^days. A site with a record is occupied; a
    site without one is either occupied and missed every time, or unoccupied.
    """
    return site_log_likelihood(sigmoid(parameters[0]), sigmoid(parameters[1]), detections, days)


def site_log_likelihood(psi, daily, detections: np.ndarray, days: np.ndarray) -> float:
    """Return the occupancy model's log-likelihood for occupancy ``psi`` and daily
    detection probability ``daily``, either one value for all sites or one per site
    (see ``log_likelihood``)."""
    psi = np.asarray(psi, dtype=float)
    daily = np.asarray(daily, dtype=float)
    if daily.ndim == 1:
        daily = daily[:, None]
    surveyed = ~np.isnan(detections)
    recorded = np.nan_to_num(detections) == 1
    log_missed = np.where(surveyed, days * np.log1p(-daily), 0.0)
    log_found = np.where(recorded, np.log(-np.expm1(log_missed.clip(max=-1e-12))), 0.0)
    log_history = np.where(recorded, log_found, log_missed).sum(axis=1)
    ever = recorded.any(axis=1)
    return float(
        np.where(ever, np.log(psi) + log_history, np.logaddexp(np.log(psi) + log_history, np.log1p(-psi))).sum()
    )


def numerical_derivatives(function, point: np.ndarray, step: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Return the gradient and Hessian of ``function`` at ``point`` by central
    differences."""
    size = len(point)
    gradient, hessian = np.zeros(size), np.zeros((size, size))
    unit = np.eye(size) * step
    centre = function(point)
    for i in range(size):
        gradient[i] = (function(point + unit[i]) - function(point - unit[i])) / (2 * step)
        for j in range(size):
            if i == j:
                hessian[i, i] = (function(point + unit[i]) - 2 * centre + function(point - unit[i])) / step**2
            else:
                hessian[i, j] = (
                    function(point + unit[i] + unit[j]) - function(point + unit[i] - unit[j])
                    - function(point - unit[i] + unit[j]) + function(point - unit[i] - unit[j])
                ) / (4 * step**2)
    return gradient, hessian


def maximize(function, start: np.ndarray, iterations: int = 100) -> np.ndarray:
    """Return the point that maximizes a smooth ``function`` of a few parameters,
    found by Newton's method with step halving, within ``LOGIT_LIMITS``."""
    point, value = start.astype(float), function(start)
    for _ in range(iterations):
        gradient, hessian = numerical_derivatives(function, point)
        try:
            step = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = gradient
        if np.dot(step, gradient) <= 0:
            step = gradient
        for _ in range(30):
            candidate = np.clip(point + step, *LOGIT_LIMITS)
            candidate_value = function(candidate)
            if candidate_value >= value:
                break
            step = step / 2
        else:
            break
        if candidate_value - value < 1e-10 and np.max(np.abs(candidate - point)) < 1e-6:
            point, value = candidate, candidate_value
            break
        point, value = candidate, candidate_value
    return point


def profile_interval(function, estimate: np.ndarray) -> tuple[float, float]:
    """Return the 95% profile-likelihood interval of the first parameter: the values
    at which its log-likelihood, maximized over the other parameters, falls 1.92 below
    the maximum. Unlike an interval from the curvature at the estimate, it follows the
    likelihood's real shape, so it stays accurate when detection is rare and reaches
    the limit when the data cannot rule out ψ = 1."""
    best = function(estimate)
    target = best - CHI_SQUARE_95 / 2

    def profile(value: float) -> float:
        nuisance = maximize(lambda rest: function(np.array([value, *rest])), estimate[1:])
        return function(np.array([value, *nuisance]))

    def crossing(limit: float) -> float:
        if profile(limit) >= target:
            return limit
        inside, outside = float(estimate[0]), limit
        for _ in range(40):
            middle = (inside + outside) / 2
            if profile(middle) >= target:
                inside = middle
            else:
                outside = middle
        return (inside + outside) / 2

    return crossing(LOGIT_LIMITS[0]), crossing(LOGIT_LIMITS[1])


def fit_occupancy(history: pd.DataFrame, effort: pd.DataFrame, occasion_days: int) -> dict:
    """Fit the occupancy model to one species' detection history and the camera days
    of each occasion. Returns occupancy (ψ) with a 95% confidence interval, the
    probability of recording the species in a full occasion where it is present, and
    the probability of missing it over the whole survey at an occupied site with the
    average effort, and a warning when the estimate should not be trusted."""
    detections = history.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    days = effort.loc[history.index, history.columns].to_numpy(dtype=float)
    days = np.where(np.nan_to_num(detections) == 1, np.maximum(days, 1), days)
    surveyed = ~np.isnan(detections).all(axis=1)
    detections, days = detections[surveyed], days[surveyed]

    def function(parameters):
        return log_likelihood(parameters, detections, days)

    grid = [np.array([a, b]) for a in np.linspace(-4, 6, 11) for b in np.linspace(-8, 1, 10)]
    estimate = maximize(function, max(grid, key=function))
    psi, daily = sigmoid(estimate)
    mean_days = np.nansum(np.where(np.isnan(detections), 0, days), axis=1).mean()
    result = {
        "occupancy": round(float(psi), 2),
        "detectionPerOccasion": round(float(1 - (1 - daily) ** occasion_days), 2),
        "missedAtOccupiedSite": round(float((1 - daily) ** mean_days), 2),
    }
    lower, upper = profile_interval(function, estimate)
    result["occupancyLower"] = round(float(sigmoid(lower)), 2)
    result["occupancyUpper"] = round(float(sigmoid(upper)), 2)
    result["warning"] = occupancy_warning(result)
    return result


def occupancy_warning(result: dict) -> str | None:
    """Return why an occupancy estimate should not be trusted, if it should not: at
    ψ = 1 the species was recorded too rarely to tell how many sites it was missed at,
    and a wide interval means the data cannot pin occupancy down."""
    if result["occupancy"] >= AT_LIMIT:
        return "detection too rare to estimate"
    if result["occupancyUpper"] - result["occupancyLower"] > WIDE_INTERVAL:
        return "wide interval"
    return None


def occupancy_models(histories: dict[str, pd.DataFrame], effort: pd.DataFrame, occasion_days: int) -> pd.DataFrame:
    """Return the occupancy model's estimates for every taxon recorded at enough
    sites to estimate both occupancy and detection."""
    rows = []
    for name, history in histories.items():
        if int(history.eq(1).any(axis=1).sum()) < MIN_SITES_FOR_MODEL:
            continue
        rows.append({"taxon": name, **fit_occupancy(history, effort, occasion_days)})
    columns = [
        "occupancy", "occupancyLower", "occupancyUpper", "detectionPerOccasion", "missedAtOccupiedSite", "warning",
    ]
    return pd.DataFrame(rows, columns=["taxon", *columns]).set_index("taxon")


def site_doubts(histories: dict[str, pd.DataFrame], effort: pd.DataFrame, occasion_days: int, chances) -> pd.DataFrame:
    """Return, for every taxon, the expected number of sites where its records are
    false and where it was missed, from the AI's calibrated mistakes (``chances``, see
    ``uncertainty.load``)."""
    from camtraptools import uncertainty

    rows = []
    for name, history in histories.items():
        certainty = uncertainty.cell_certainty(chances, name, effort, occasion_days)
        false_sites, missed_sites = uncertainty.site_doubt(history, certainty)
        rows.append({"taxon": name, "expectedFalseSites": round(false_sites, 1), "expectedMissedSites": round(missed_sites, 1)})
    return pd.DataFrame(rows, columns=["taxon", "expectedFalseSites", "expectedMissedSites"]).set_index("taxon")


def file_name(taxon: str) -> str:
    """Return a file name for a taxon's history, e.g. ``Oryx_gazella.csv``."""
    return taxon.replace(" ", "_") + ".csv"


def analyse(
    events: pd.DataFrame,
    deployments: pd.DataFrame,
    out_dir: Path,
    occasion_days: int = DEFAULT_OCCASION_DAYS,
    taxa: list[str] | None = None,
    all_taxa: bool = False,
    min_activity_events: int = activity.DEFAULT_MIN_EVENTS,
    occupancy_covariates: list[str] | None = None,
    detection_covariates: list[str] | None = None,
    chances=None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Write a detection history per taxon in ``out_dir/histories/``, the sampling
    effort in ``effort.csv``, the relative abundance index per site in ``rai.csv`` and,
    in ``occupancy.csv``, each taxon's events and RAI, naive occupancy and the
    occupancy model's estimates for taxa recorded at three or more sites. Taxa with at
    least ``min_activity_events`` events also get an activity curve, a peak hour and
    their overlap with each other (see ``activity.run``). The species accumulation
    curve and the Chao2 estimate of richness are written too (see ``richness.run``)
    and returned in ``summary.attrs["richness"]``. With ``occupancy_covariates`` or
    ``detection_covariates`` (deployment columns), occupancy models with those site
    covariates are compared by AIC (see ``covariates.run``), with the result in
    ``summary.attrs["covariates"]``. With ``chances`` (the AI's calibrated mistakes,
    from ``uncertainty.load``), the summary also has the expected number of sites per
    taxon with a false record and where it was missed.

    ``events`` comes from ``animal_events``. By default every species is included;
    ``all_taxa`` adds higher taxa such as ``Bovidae``, and ``taxa`` chooses taxa by
    name. Returns the summary, the number of events left out by taxon, and the overlaps.
    """
    effort = effort_matrix(deployments, occasion_days)
    named = events.dropna(subset=["scientificName"])
    if taxa:
        chosen = [name for name in taxa if name in set(named["scientificName"])]
    else:
        chosen = sorted(name for name in named["scientificName"].unique() if all_taxa or is_species(name))
    left_out = events.loc[~events["scientificName"].isin(chosen), "scientificName"].fillna("animal (unidentified)")

    histories = {
        name: detection_history(named[named["scientificName"] == name], effort, occasion_days) for name in chosen
    }
    (out_dir / HISTORIES_FOLDER).mkdir(parents=True, exist_ok=True)
    for name, history in histories.items():
        history.to_csv(out_dir / HISTORIES_FOLDER / file_name(name))
    effort.to_csv(out_dir / EFFORT_FILE)
    summary = summarize(histories, events) if histories else pd.DataFrame()
    rai = relative_abundance(named[named["scientificName"].isin(chosen)], effort, chosen)
    rai.to_csv(out_dir / RAI_FILE)
    if histories:
        summary.insert(1, "rai", rai.loc["all sites"].reindex(summary.index))
        summary = summary.join(occupancy_models(histories, effort, occasion_days))
        if chances is not None:
            summary = summary.join(site_doubts(histories, effort, occasion_days, chances))
    species_events = named[named["scientificName"].isin(chosen)]
    peaks, overlaps = activity.run(species_events, out_dir, min_activity_events)
    estimate = richness.run(species_events, int(effort.to_numpy().sum()), out_dir)
    if histories:
        summary = summary.join(peaks[["peak"]].rename(columns={"peak": "activityPeak"}))
    summary.to_csv(out_dir / SUMMARY_FILE)
    summary.attrs["richness"] = estimate
    if occupancy_covariates or detection_covariates:
        from camtraptools import covariates

        summary.attrs["covariates"] = covariates.run(
            histories, effort, deployments, occupancy_covariates or [], detection_covariates or [], out_dir
        )
    return summary, left_out.value_counts(), overlaps


def run(
    package_dir: Path,
    out_dir: Path,
    occasion_days: int = DEFAULT_OCCASION_DAYS,
    taxa: list[str] | None = None,
    all_taxa: bool = False,
    min_activity_events: int = activity.DEFAULT_MIN_EVENTS,
    occupancy_covariates: list[str] | None = None,
    detection_covariates: list[str] | None = None,
    ai_errors: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Run ``analyse`` on the package's events, using reviewed decisions where there
    are any. With ``ai_errors``, also estimate the sites each taxon's results may have
    wrong because of the AI's mistakes, calibrated on the reviews."""
    deployments = pd.read_csv(package_dir / "deployments.csv", dtype=str)
    events = animal_events(event_observations(package_dir), deployments)
    chances = None
    if ai_errors:
        from camtraptools import uncertainty

        chances = uncertainty.load(package_dir, deployments)
        if chances.reviewed < uncertainty.MIN_REVIEWS:
            raise ValueError(
                f"estimating AI mistakes needs at least {uncertainty.MIN_REVIEWS} reviewed events to learn how "
                f"often the AI is wrong, and there are {chances.reviewed}; review some first, for example with "
                "`camtraptools review --order plan`"
            )
    return analyse(
        events, deployments, out_dir, occasion_days, taxa, all_taxa, min_activity_events,
        occupancy_covariates, detection_covariates, chances,
    )
