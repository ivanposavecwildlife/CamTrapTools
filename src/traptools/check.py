"""Module 1b: health checks for each camera deployment.

Each check takes the media table for ONE deployment and returns a list of
problems found. Keeping checks small and separate makes them easy to test.
"""

from pathlib import Path

import pandas as pd


def check_bad_timestamps(media: pd.DataFrame) -> list[str]:
    """TODO (week 7): flag missing timestamps, dates before 2010, dates in the
    future, and sudden jumps (a clock reset or a daylight-saving change)."""
    raise NotImplementedError


def check_gaps(media: pd.DataFrame, max_gap_days: float = 7) -> list[str]:
    """TODO (week 7): flag long stretches with no photos. The camera may have
    died, the card may have filled up, or it may just be a quiet spot. How
    could you tell these apart?"""
    raise NotImplementedError


def check_trigger_rate(media: pd.DataFrame) -> list[str]:
    """TODO (week 8): triggers per day, plus the day/night split. A huge rate
    often means vegetation in front of the lens is setting the camera off."""
    raise NotImplementedError


def run(package_dir: Path) -> None:
    """Load media.csv, run every check for each deployment and write a report."""
    raise NotImplementedError("check isn't implemented yet. That's your week 7 task!")
