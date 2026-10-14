"""Data-quality checks for camera-trap deployments.

Each check receives the media table for a single deployment and returns a
list of human-readable issues (empty if none were found).
"""

from pathlib import Path

import pandas as pd


def check_bad_timestamps(media: pd.DataFrame) -> list[str]:
    """Flag missing timestamps, dates before 2010 or in the future, and abrupt
    jumps that indicate a clock reset or daylight-saving shift."""
    raise NotImplementedError


def check_gaps(media: pd.DataFrame, max_gap_days: float = 7) -> list[str]:
    """Flag periods longer than ``max_gap_days`` without any capture, which may
    indicate battery failure, a full card or a dislodged camera."""
    raise NotImplementedError


def check_trigger_rate(media: pd.DataFrame) -> list[str]:
    """Report triggers per day and the day/night split; an abnormally high rate
    usually indicates vegetation or sun glare causing false triggers."""
    raise NotImplementedError


def run(package_dir: Path) -> None:
    """Run all checks on each deployment in ``package_dir`` and write a report."""
    raise NotImplementedError("`check` is not implemented yet.")
