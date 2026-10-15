"""Data-quality checks for camera-trap deployments.

Each check receives the media table for a single deployment and returns a
list of human-readable issues (empty if none were found).
"""

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


def check_deployment(media: pd.DataFrame) -> list[str]:
    """Run every check on the images of one deployment."""
    issues = []
    issues += check_missing_timestamps(media)
    issues += check_implausible_dates(media)
    issues += check_clock_jumps(media)
    return issues


def run(package_dir: Path) -> dict[str, list[str]]:
    """Check every deployment in ``package_dir`` and return its issues by deployment ID."""
    media = pd.read_csv(package_dir / "media.csv", parse_dates=["timestamp"])
    results = {}
    for deployment_id, deployment_media in media.groupby("deploymentID"):
        results[deployment_id] = check_deployment(deployment_media)
    return results
