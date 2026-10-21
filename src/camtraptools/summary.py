"""A one-page HTML summary of a survey's results, to hand back to a project.

It combines camera health (``check``), the species recorded with their relative
abundance index, occupancy and daily activity (``ecology``), species richness, and
how the labels were made (AI, adapted layer, human review), with a short account of
the methods and their limits.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup

from camtraptools import __version__, activity, check, ecology, richness
from camtraptools.detect import OBSERVATIONS_FILE
from camtraptools.evaluate import EVALUATION_FILE
from camtraptools.report import timeline_svg
from camtraptools.reviews import load_reviews

templates = Environment(loader=PackageLoader("camtraptools"), autoescape=select_autoescape())


def species_rows(summary: pd.DataFrame) -> list[dict]:
    """Return one row per species for the summary table."""
    rows = []
    for name, row in summary.iterrows():
        modelled = row.get("occupancy")
        rows.append(
            {
                "name": name,
                "events": int(row["events"]),
                "rai": f"{row['rai']:.2f}",
                "sites": f"{int(row['sitesDetected'])}/{int(row['sitesSurveyed'])}",
                "naive": f"{row['naiveOccupancy']:.2f}",
                "modelled": f"{modelled:.2f}" if pd.notna(modelled) else "–",
                "interval": (
                    f"{row['occupancyLower']:.2f}–{row['occupancyUpper']:.2f}" if pd.notna(modelled) else ""
                ),
                "warning": row["warning"] if isinstance(row.get("warning"), str) else "",
                "doubt": (
                    f"{row['expectedFalseSites']:g} / {row['expectedMissedSites']:g}"
                    if pd.notna(row.get("expectedFalseSites"))
                    else ""
                ),
                "doubtful": pd.notna(row.get("expectedFalseSites"))
                and row["expectedFalseSites"] + row["expectedMissedSites"] >= ecology.SITES_IN_DOUBT_TO_FLAG,
                "peak": row["activityPeak"] if isinstance(row.get("activityPeak"), str) else "–",
            }
        )
    return rows


def labelling(package_dir: Path) -> dict:
    """Return how the event labels were made: the classifiers named in the
    observations, how many events a person reviewed, and the accuracy from the last
    ``evaluate``, if any."""
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    events = observations[(observations["observationLevel"] == "event") & (observations["classificationMethod"] != "human")]
    media = observations[observations["observationLevel"] == "media"]
    reviewed = load_reviews(package_dir)
    result = {
        "classifiers": sorted(media["classifiedBy"].dropna().unique()),
        "events": len(events),
        "reviewed": int(reviewed["eventID"].isin(events["eventID"]).sum()),
        "accuracy": None,
    }
    evaluation_file = package_dir / EVALUATION_FILE
    if evaluation_file.exists():
        evaluation = pd.read_csv(evaluation_file)
        if len(evaluation):
            outcomes = evaluation["outcome"].value_counts()
            result["accuracy"] = {
                "events": len(evaluation),
                "correct": int(outcomes.get("correct", 0)),
                "higher": int(outcomes.get("not to species", 0)),
                "finer": int(outcomes.get("more specific", 0)),
                "wrong": int(outcomes.get("wrong", 0)),
            }
    return result


def build(
    package_dir: Path,
    title: str | None = None,
    occasion_days: int = ecology.DEFAULT_OCCASION_DAYS,
    min_activity_events: int = activity.DEFAULT_MIN_EVENTS,
    ai_errors: bool = False,
) -> str:
    """Return the HTML summary of the survey in ``package_dir``, with the sites per
    species in doubt because of AI mistakes if ``ai_errors``."""
    media, deployments = check.load_package(package_dir)
    issues = check.run(package_dir)
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder)
        summary, left_out, overlaps = ecology.run(
            package_dir, out, occasion_days, min_activity_events=min_activity_events, ai_errors=ai_errors
        )
        effort = pd.read_csv(out / ecology.EFFORT_FILE, index_col="site")
        activity_chart = (out / activity.PLOT_FILE).read_text(encoding="utf-8") if (out / activity.PLOT_FILE).exists() else ""
        richness_chart = (out / richness.PLOT_FILE).read_text(encoding="utf-8") if (out / richness.PLOT_FILE).exists() else ""
    starts = pd.to_datetime(deployments["deploymentStart"], format="ISO8601", errors="coerce")
    ends = pd.to_datetime(deployments["deploymentEnd"], format="ISO8601", errors="coerce")
    labels = labelling(package_dir)
    return templates.get_template("summary.html").render(
        title=title or package_dir.resolve().name,
        generated=f"{datetime.now():%Y-%m-%d %H:%M}",
        period=f"{starts.min():%d %b %Y} – {ends.max():%d %b %Y}" if starts.notna().any() and ends.notna().any() else "",
        cameras=len(effort),
        camera_days=int(effort.to_numpy().sum()),
        images=len(media),
        events=labels["events"],
        species=len(summary),
        issues={name: found for name, found in issues.items() if found},
        deployments=len(issues),
        timeline=Markup(timeline_svg(media, deployments, check.MAX_GAP_DAYS)),
        species_rows=species_rows(summary) if not summary.empty else [],
        left_out=left_out.head(8).to_dict(),
        occasion_days=occasion_days,
        activity_chart=Markup(activity_chart),
        overlaps=overlaps.head(10).to_dict("records"),
        min_activity_events=min_activity_events,
        richness=summary.attrs.get("richness"),
        richness_chart=Markup(richness_chart),
        labels=labels,
        ai_errors=ai_errors,
        version=__version__,
    )


def run(
    package_dir: Path,
    out_file: Path,
    title: str | None = None,
    occasion_days: int = ecology.DEFAULT_OCCASION_DAYS,
    min_activity_events: int = activity.DEFAULT_MIN_EVENTS,
    ai_errors: bool = False,
) -> None:
    """Write the HTML summary of the survey in ``package_dir`` to ``out_file``."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(build(package_dir, title, occasion_days, min_activity_events, ai_errors), encoding="utf-8")
