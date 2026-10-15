"""Data-quality checks for camera-trap deployments.

Each check receives the media table for a single deployment and returns a
list of human-readable issues (empty if none were found).
"""

from itertools import pairwise
from pathlib import Path

import pandas as pd


def check_missing_timestamps(media: pd.DataFrame) -> list[str]:
    """Flag images whose capture time could not be read."""
    missing = media["timestamp"].isna().sum()
    if missing:
        return [f"{missing} image(s) without a readable timestamp"]
    return []


EARLIEST_PLAUSIBLE = pd.Timestamp("2010-01-01")


def check_implausible_dates(
    media: pd.DataFrame, now: pd.Timestamp | None = None
) -> list[str]:
    """Flag images dated before 2010 (typical of a reset camera clock) or in the future."""
    if now is None:
        now = pd.Timestamp.now()
    issues = []
    too_early = (media["timestamp"] < EARLIEST_PLAUSIBLE).sum()
    if too_early:
        issues.append(
            f"{too_early} image(s) dated before {EARLIEST_PLAUSIBLE.year}; the camera clock was probably reset"
        )
    in_future = (media["timestamp"] > now).sum()
    if in_future:
        issues.append(
            f"{in_future} image(s) dated in the future; the camera clock was set wrongly"
        )
    return issues


def check_clock_jumps(media: pd.DataFrame) -> list[str]:
    """Flag places where time goes backwards between consecutive images.

    Cameras number their files in the order they take them, so a later file
    with an earlier time means the clock was reset or changed mid-deployment.
    """
    ordered = media.sort_values("filePath").copy()
    ordered["step"] = ordered["timestamp"].diff()
    backwards = ordered[ordered["step"] < pd.Timedelta(0)]
    issues = []
    for row in backwards.itertuples():
        issues.append(f"clock went back {-row.step} at {row.fileName}")
    return issues


MAX_GAP_DAYS = 21


def find_gaps(
    media: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_gap_days: float = MAX_GAP_DAYS,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Return the periods longer than ``max_gap_days`` without images as
    ``(from, to, where)``, where ``where`` is "start", "middle" or "end"."""
    times = media["timestamp"]
    times = times[(times >= start) & (times <= end)].sort_values()
    points = pd.concat([pd.Series([start]), times, pd.Series([end])], ignore_index=True)
    intervals = list(pairwise(points))
    gaps = []
    for position, (before, after) in enumerate(intervals):
        if after - before > pd.Timedelta(days=max_gap_days):
            if position == 0:
                where = "start"
            elif position == len(intervals) - 1:
                where = "end"
            else:
                where = "middle"
            gaps.append((before, after, where))
    return gaps


def check_gaps(
    media: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_gap_days: float = MAX_GAP_DAYS,
) -> list[str]:
    """Flag long periods without images, which may mean the camera stopped working."""
    if pd.isna(start) or pd.isna(end) or start < EARLIEST_PLAUSIBLE:
        return []
    issues = []
    for gap_start, gap_end, where in find_gaps(media, start, end, max_gap_days):
        days = (gap_end - gap_start) / pd.Timedelta(days=1)
        period = f"{gap_start:%Y-%m-%d} to {gap_end:%Y-%m-%d}"
        if where == "start":
            issues.append(
                f"no images in the first {days:.1f} days ({period}); was the camera armed, and is the start date right?"
            )
        elif where == "end":
            issues.append(
                f"no images in the last {days:.1f} days ({period}); the camera may have stopped (battery or full card)"
            )
        else:
            issues.append(
                f"no images for {days:.1f} days ({period}); check for a camera fault"
            )
    return issues


MAX_DAILY_IMAGES = 100


def check_trigger_rate(
    media: pd.DataFrame, max_daily_images: int = MAX_DAILY_IMAGES
) -> list[str]:
    """Flag days with an abnormally high number of images."""
    daily = media["timestamp"].dt.date.value_counts()
    busy = daily[daily > max_daily_images]
    if busy.empty:
        return []
    most = f"most: {busy.max()} on {busy.idxmax()}"
    return [
        f"{len(busy)} day(s) with more than {max_daily_images} images ({most}); "
        + "often vegetation moving in front of the lens or sun glare"
    ]


def is_night(timestamps: pd.Series) -> pd.Series:
    """Return True for times between 18:00 and 06:00."""
    return (timestamps.dt.hour < 6) | (timestamps.dt.hour >= 18)


MIN_IMAGES_FOR_DAY_NIGHT_CHECK = 50


def check_day_night(media: pd.DataFrame) -> list[str]:
    """Flag cameras that only take images by day or only by night."""
    timestamps = media["timestamp"].dropna()
    if len(timestamps) < MIN_IMAGES_FOR_DAY_NIGHT_CHECK:
        return []
    night = is_night(timestamps)
    if not night.any():
        return [
            f"none of {len(timestamps)} images were taken at night; check the infrared flash"
        ]
    if night.all():
        return [
            f"none of {len(timestamps)} images were taken during the day; check the camera settings"
        ]
    return []


def summarize_deployment(
    media: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict:
    """Return key numbers for one deployment: its length, image counts and longest gap."""
    days = (end - start) / pd.Timedelta(days=1)
    timestamps = media["timestamp"].dropna()
    intervals = find_gaps(media, start, end, max_gap_days=0)
    longest_gap = max(
        (after - before for before, after, _ in intervals), default=pd.Timedelta(0)
    )
    return {
        "days": round(days, 1),
        "images": len(media),
        "imagesPerDay": round(len(media) / days, 1) if days > 0 else None,
        "nightShare": round(is_night(timestamps).mean(), 2)
        if len(timestamps)
        else None,
        "longestGapDays": round(longest_gap / pd.Timedelta(days=1), 1),
    }


def check_deployment(
    media: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    max_gap_days: float = MAX_GAP_DAYS,
) -> list[str]:
    """Run every check on the images of one deployment. Without a recorded
    period, the first and last image define it."""
    if start is None:
        start = media["timestamp"].min()
    if end is None:
        end = media["timestamp"].max()
    issues = []
    issues += check_missing_timestamps(media)
    issues += check_implausible_dates(media)
    issues += check_clock_jumps(media)
    issues += check_gaps(media, start, end, max_gap_days)
    issues += check_trigger_rate(media)
    issues += check_day_night(media)
    return issues


def load_package(package_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read media.csv and deployments.csv from a package, with dates parsed."""
    media = pd.read_csv(package_dir / "media.csv", parse_dates=["timestamp"])
    deployments = pd.read_csv(
        package_dir / "deployments.csv",
        parse_dates=["deploymentStart", "deploymentEnd"],
    ).set_index("deploymentID")
    return media, deployments


def run(package_dir: Path, max_gap_days: float = MAX_GAP_DAYS) -> dict[str, list[str]]:
    """Check every deployment in ``package_dir`` and return its issues by deployment ID."""
    media, deployments = load_package(package_dir)
    results = {}
    for deployment_id, deployment_media in media.groupby("deploymentID"):
        start = deployments.loc[deployment_id, "deploymentStart"]
        end = deployments.loc[deployment_id, "deploymentEnd"]
        results[deployment_id] = check_deployment(
            deployment_media, start, end, max_gap_days
        )
    return results


def summarize(package_dir: Path) -> pd.DataFrame:
    """Return one row of key numbers per deployment in ``package_dir``."""
    media, deployments = load_package(package_dir)
    rows = {}
    for deployment_id, deployment_media in media.groupby("deploymentID"):
        start = deployments.loc[deployment_id, "deploymentStart"]
        end = deployments.loc[deployment_id, "deploymentEnd"]
        rows[deployment_id] = summarize_deployment(deployment_media, start, end)
    return pd.DataFrame.from_dict(rows, orient="index")
