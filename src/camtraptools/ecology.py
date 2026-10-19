"""Turn event-level observations into detection histories and naive occupancy.

A detection history has one row per camera site and one column per sampling occasion
(a block of days): 1 if the species was recorded there during the occasion, 0 if the
camera was running but did not record it, and empty if the camera was not running.
Naive occupancy is the share of surveyed sites where a species was recorded at least
once. It ignores that species are missed where they are present, which occupancy
models correct for.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from camtraptools.crops import location_of
from camtraptools.detect import OBSERVATIONS_FILE
from camtraptools.reviews import prefer_human

DEFAULT_OCCASION_DAYS = 7
HISTORIES_FOLDER = "histories"
EFFORT_FILE = "effort.csv"
SUMMARY_FILE = "occupancy.csv"


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


def animal_events(package_dir: Path, deployments: pd.DataFrame) -> pd.DataFrame:
    """Return every animal event with its site, day and scientific name, using the
    reviewer's decision wherever an event was reviewed."""
    observations = prefer_human(pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str))
    events = observations[
        (observations["observationLevel"] == "event") & (observations["observationType"] == "animal")
    ].copy()
    events["site"] = events["deploymentID"].map(location_of(deployments))
    events["day"] = pd.to_datetime(events["eventStart"], format="ISO8601", errors="coerce").dt.date
    return events.dropna(subset=["day"])[["eventID", "site", "day", "scientificName"]]


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


def file_name(taxon: str) -> str:
    """Return a file name for a taxon's history, e.g. ``Oryx_gazella.csv``."""
    return taxon.replace(" ", "_") + ".csv"


def run(
    package_dir: Path,
    out_dir: Path,
    occasion_days: int = DEFAULT_OCCASION_DAYS,
    taxa: list[str] | None = None,
    all_taxa: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Write a detection history per taxon in ``out_dir/histories/``, the sampling
    effort in ``effort.csv`` and naive occupancy in ``occupancy.csv``.

    By default every species is included; ``all_taxa`` adds higher taxa such as
    ``Bovidae``, and ``taxa`` chooses taxa by name. Returns the summary and the number
    of events left out, by taxon.
    """
    deployments = pd.read_csv(package_dir / "deployments.csv", dtype=str)
    effort = effort_matrix(deployments, occasion_days)
    events = animal_events(package_dir, deployments)
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
    summary.to_csv(out_dir / SUMMARY_FILE)
    return summary, left_out.value_counts()
