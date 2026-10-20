"""Export a package as a complete, publishable Camtrap DP 1.0.2 data package."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from camtraptools import __version__
from camtraptools.detect import OBSERVATION_COLUMNS, OBSERVATIONS_FILE
from camtraptools.reviews import prefer_human

CAMTRAP_DP_VERSION = "1.0.2"
CAMTRAP_DP_URL = f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{CAMTRAP_DP_VERSION}"
MEDIA_COLUMNS = [
    "mediaID", "deploymentID", "captureMethod", "timestamp", "filePath", "filePublic",
    "fileName", "fileMediatype", "exifData", "favorite", "mediaComments",
]
DEPLOYMENT_COLUMNS = [
    "deploymentID", "locationID", "locationName", "latitude", "longitude", "coordinateUncertainty",
    "deploymentStart", "deploymentEnd", "setupBy", "cameraID", "cameraModel", "cameraDelay",
    "cameraHeight", "cameraDepth", "cameraTilt", "cameraHeading", "detectionDistance",
    "timestampIssues", "baitUse", "featureType", "habitat", "deploymentGroups", "deploymentTags",
    "deploymentComments",
]
SAMPLING_DESIGNS = ["simpleRandom", "systematicRandom", "clusteredRandom", "experimental", "targeted", "opportunistic"]


def with_timezone(values: pd.Series, timezone: str) -> pd.Series:
    """Return local date-times as ISO 8601 text with the UTC offset of ``timezone``,
    e.g. ``2018-11-29T08:25:47+02:00``. Missing values stay missing."""
    times = pd.to_datetime(values, format="ISO8601").dt.tz_localize(
        ZoneInfo(timezone), ambiguous="NaT", nonexistent="NaT"
    )
    text = [time.isoformat() if pd.notna(time) else None for time in times]
    return pd.Series(text, index=values.index, dtype=object)


def export_deployments(deployments: pd.DataFrame, timezone: str) -> tuple[pd.DataFrame, list[str]]:
    """Return the deployments table with every Camtrap DP column, and the names of
    columns that are not part of Camtrap DP and were left out."""
    exported = deployments.reindex(columns=DEPLOYMENT_COLUMNS)
    missing = exported.loc[exported["latitude"].isna() | exported["longitude"].isna(), "deploymentID"]
    if not missing.empty:
        raise ValueError(f"deployments without coordinates: {', '.join(missing)}; add them with a field sheet")
    exported["deploymentStart"] = with_timezone(deployments["deploymentStart"], timezone)
    exported["deploymentEnd"] = with_timezone(deployments["deploymentEnd"], timezone)
    extra = [column for column in deployments.columns if column not in DEPLOYMENT_COLUMNS]
    return exported, extra


def export_media(media: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Return the media table with every Camtrap DP column. Images without a
    timestamp are left out, because Camtrap DP requires one."""
    exported = media.reindex(columns=MEDIA_COLUMNS)
    exported["timestamp"] = with_timezone(media["timestamp"], timezone)
    exported["filePublic"] = "false"
    return exported.dropna(subset=["timestamp"])


def export_observations(observations: pd.DataFrame, media_ids: set[str], timezone: str) -> pd.DataFrame:
    """Return the observations of the exported images, with time zones added.

    Classification times without a recorded offset are left out, because the
    time zone of the computer that made them is unknown, and counts are written as
    whole numbers (``1``, not ``1.0``) as Camtrap DP requires.
    """
    kept = observations[observations["mediaID"].isna() | observations["mediaID"].isin(media_ids)].copy()
    kept["eventStart"] = with_timezone(kept["eventStart"], timezone)
    kept["eventEnd"] = with_timezone(kept["eventEnd"], timezone)
    has_offset = kept["classificationTimestamp"].str.contains(r"[+-]\d\d:\d\d$|Z$", na=False)
    kept["classificationTimestamp"] = kept["classificationTimestamp"].where(has_offset)
    kept["count"] = pd.to_numeric(kept["count"], errors="coerce").round().astype("Int64")
    return kept[OBSERVATION_COLUMNS]


def human_media(observations: pd.DataFrame) -> set[str]:
    """Return the images that show people: those with a human observation, and every
    image of an event whose decision is human."""
    media_level = observations[observations["observationLevel"] == "media"]
    events = observations[observations["observationLevel"] == "event"]
    human_events = set(events.loc[events["observationType"] == "human", "eventID"])
    human = (media_level["observationType"] == "human") | media_level["eventID"].isin(human_events)
    return set(media_level.loc[human, "mediaID"].dropna())


METRES_PER_DEGREE = 111_320


def generalise_coordinates(deployments: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Return the deployments with coordinates rounded to ``decimals`` decimal places,
    to protect the sites of sensitive species, and ``coordinateUncertainty`` widened to
    the distance from the centre to the corner of the rounding grid cell."""
    generalised = deployments.copy()
    latitude = generalised["latitude"].astype(float)
    step = 10.0**-decimals
    half_height = step / 2 * METRES_PER_DEGREE
    half_width = half_height * np.cos(np.radians(latitude))
    corner = np.ceil(np.sqrt(half_height**2 + half_width**2)).astype(int)
    existing = pd.to_numeric(generalised["coordinateUncertainty"], errors="coerce")
    generalised["coordinateUncertainty"] = np.maximum(existing.fillna(0), corner).astype(int).astype(str)
    generalised["latitude"] = latitude.round(decimals).map(lambda value: f"{value:.{decimals}f}")
    generalised["longitude"] = (
        generalised["longitude"].astype(float).round(decimals).map(lambda value: f"{value:.{decimals}f}")
    )
    return generalised


def spatial_coverage(deployments: pd.DataFrame) -> dict:
    """Return the GeoJSON bounding box (or point) around all deployments."""
    west, east = deployments["longitude"].astype(float).agg(["min", "max"])
    south, north = deployments["latitude"].astype(float).agg(["min", "max"])
    if (west, south) == (east, north):
        return {"type": "Point", "coordinates": [west, south]}
    corners = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    return {"type": "Polygon", "coordinates": [corners]}


def descriptor(
    deployments: pd.DataFrame,
    observations: pd.DataFrame,
    title: str,
    contributors: list[str],
    sampling_design: str,
    data_license: str,
    media_license: str,
    coordinate_decimals: int | None = None,
) -> dict:
    """Return the ``datapackage.json`` contents for the exported tables."""
    names = sorted(observations["scientificName"].dropna().unique())
    starts = pd.to_datetime(deployments["deploymentStart"], format="ISO8601", utc=True)
    ends = pd.to_datetime(deployments["deploymentEnd"], format="ISO8601", utc=True)
    resources = [
        {
            "name": name,
            "path": f"{name}.csv",
            "profile": "tabular-data-resource",
            "format": "csv",
            "mediatype": "text/csv",
            "encoding": "utf-8",
            "schema": f"{CAMTRAP_DP_URL}/{name}-table-schema.json",
        }
        for name in ("deployments", "media", "observations")
    ]
    return {
        "profile": f"{CAMTRAP_DP_URL}/camtrap-dp-profile.json",
        "title": title,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "contributors": [
            {"title": name, "role": "contact" if index == 0 else "contributor"} for index, name in enumerate(contributors)
        ],
        "licenses": [{"name": data_license, "scope": "data"}, {"name": media_license, "scope": "media"}],
        "sources": [{"title": f"CamTrapTools {__version__}", "path": "https://github.com/ivanposavecwildlife/CamTrapTools"}],
        "project": {
            "title": title,
            "samplingDesign": sampling_design,
            "captureMethod": ["activityDetection"],
            "individualAnimals": False,
            "observationLevel": sorted(observations["observationLevel"].unique()),
        },
        **({"coordinatePrecision": 10.0**-coordinate_decimals} if coordinate_decimals is not None else {}),
        "spatial": spatial_coverage(deployments),
        "temporal": {"start": f"{starts.min():%Y-%m-%d}", "end": f"{ends.max():%Y-%m-%d}"},
        "taxonomic": [{"scientificName": name} for name in names],
        "resources": resources,
    }


def run(
    package_dir: Path,
    out_dir: Path,
    title: str,
    contributors: list[str],
    timezone: str,
    sampling_design: str = "opportunistic",
    data_license: str = "CC-BY-4.0",
    media_license: str = "CC-BY-4.0",
    exclude_humans: bool = False,
    coordinate_decimals: int | None = None,
) -> dict:
    """Write a Camtrap DP data package to ``out_dir`` and return what was left out.

    With ``exclude_humans``, images of people and their observations are left out;
    with ``coordinate_decimals``, coordinates are rounded (see
    ``generalise_coordinates``).
    """
    deployments, extra_columns = export_deployments(
        pd.read_csv(package_dir / "deployments.csv", dtype=str), timezone
    )
    if coordinate_decimals is not None:
        deployments = generalise_coordinates(deployments, coordinate_decimals)
    media = pd.read_csv(package_dir / "media.csv", dtype=str)
    all_observations = prefer_human(pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str))
    people = human_media(all_observations)
    if exclude_humans:
        media = media[~media["mediaID"].isin(people)]
        all_observations = all_observations[
            ~((all_observations["observationLevel"] == "event") & (all_observations["observationType"] == "human"))
        ]
    exported_media = export_media(media, timezone)
    observations = export_observations(all_observations, set(exported_media["mediaID"]), timezone)
    out_dir.mkdir(parents=True, exist_ok=True)
    deployments.to_csv(out_dir / "deployments.csv", index=False)
    exported_media.to_csv(out_dir / "media.csv", index=False)
    observations.to_csv(out_dir / "observations.csv", index=False)
    package = descriptor(
        deployments, observations, title, contributors, sampling_design, data_license, media_license,
        coordinate_decimals,
    )
    (out_dir / "datapackage.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    return {
        "media_without_timestamp": len(media) - len(exported_media),
        "extra_columns": extra_columns,
        "human_media": len(people),
        "humans_excluded": exclude_humans,
    }
