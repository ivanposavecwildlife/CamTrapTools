"""Export a package as a complete, publishable Camtrap DP 1.0.2 data package."""

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from camtraptools import __version__
from camtraptools.detect import OBSERVATION_COLUMNS, OBSERVATIONS_FILE

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
    time zone of the computer that made them is unknown.
    """
    kept = observations[observations["mediaID"].isna() | observations["mediaID"].isin(media_ids)].copy()
    kept["eventStart"] = with_timezone(kept["eventStart"], timezone)
    kept["eventEnd"] = with_timezone(kept["eventEnd"], timezone)
    has_offset = kept["classificationTimestamp"].str.contains(r"[+-]\d\d:\d\d$|Z$", na=False)
    kept["classificationTimestamp"] = kept["classificationTimestamp"].where(has_offset)
    return kept[OBSERVATION_COLUMNS]


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
) -> dict:
    """Write a Camtrap DP data package to ``out_dir`` and return what was left out."""
    deployments, extra_columns = export_deployments(
        pd.read_csv(package_dir / "deployments.csv", dtype=str), timezone
    )
    media = pd.read_csv(package_dir / "media.csv", dtype=str)
    exported_media = export_media(media, timezone)
    observations = export_observations(
        pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str), set(exported_media["mediaID"]), timezone
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    deployments.to_csv(out_dir / "deployments.csv", index=False)
    exported_media.to_csv(out_dir / "media.csv", index=False)
    observations.to_csv(out_dir / "observations.csv", index=False)
    package = descriptor(deployments, observations, title, contributors, sampling_design, data_license, media_license)
    (out_dir / "datapackage.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
    return {"media_without_timestamp": len(media) - len(exported_media), "extra_columns": extra_columns}
