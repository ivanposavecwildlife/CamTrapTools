"""Import existing image labels as a truth file for ``evaluate``, ``crops`` and ``compare``.

Labels can come from another Camtrap DP package (for example from Agouti or Trapper),
a Wildlife Insights ``images.csv`` export, a Timelapse CSV export or any spreadsheet
with an image path and a species column. Each labelled image is matched to an image
in the package, and common names are turned into scientific names where possible.
The result is a CSV with ``filePath``, ``scientificName`` (empty for blank images) and,
where known, ``genus``, ``family``, ``order`` and ``class``.
"""

import os
from pathlib import Path, PurePosixPath

import pandas as pd

from camtraptools.detect import parse_taxon
from camtraptools.importer import match_path

FORMATS = ["auto", "camtrap-dp", "wildlife-insights", "csv"]
RANKS = ["genus", "family", "order", "class"]
BLANK_VALUES = {"blank", "empty", "none", "nothing", "no animal", "false trigger", "misfire"}
HUMAN_NAMES = {"homo sapiens", "human", "person", "people"}
UNKNOWN_NAMES = {"no cv result", "unknown", "unidentified", "unidentifiable"}
LABEL_COLUMNS = ["path", "fileName", "deployment", "label", *RANKS]
TRUTH_COLUMNS = ["filePath", "scientificName", *RANKS]


def detect_format(source: Path) -> str:
    """Return the format of a labels file or folder: a Camtrap DP package (a folder or
    its ``datapackage.json``), a Wildlife Insights image export, or another CSV."""
    if source.is_dir() or source.name == "datapackage.json":
        return "camtrap-dp"
    columns = set(pd.read_csv(source, nrows=0).columns)
    if {"filename", "deployment_id", "is_blank", "genus", "species"} <= columns:
        return "wildlife-insights"
    return "csv"


def empty_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_COLUMNS)


def read_camtrap_dp(source: Path) -> pd.DataFrame:
    """Return one label per image from a Camtrap DP package. Human classifications are
    used when the package has any, otherwise all of them. Event-level observations
    label every image of their deployment between the event's start and end."""
    folder = source if source.is_dir() else source.parent
    media = pd.read_csv(folder / "media.csv", dtype=str)
    observations = pd.read_csv(folder / "observations.csv", dtype=str)
    if (observations["classificationMethod"] == "human").any():
        observations = observations[observations["classificationMethod"] == "human"]
    labels = observations.apply(lambda row: observation_label(row["observationType"], row["scientificName"]), axis=1)
    observations = observations.assign(label=labels).dropna(subset=["label"])

    media_level = observations.dropna(subset=["mediaID"]).merge(media, on="mediaID", suffixes=("", "Media"))
    rows = [media_level]
    events = observations[observations["mediaID"].isna() & observations["eventStart"].notna()]
    if not events.empty:
        times = media.assign(time=pd.to_datetime(media["timestamp"], format="ISO8601", utc=True, errors="coerce"))
        for event in events.itertuples():
            start = pd.to_datetime(event.eventStart, format="ISO8601", utc=True)
            end = pd.to_datetime(event.eventEnd, format="ISO8601", utc=True)
            inside = times[(times["deploymentID"] == event.deploymentID) & times["time"].between(start, end)]
            rows.append(inside.assign(label=event.label))
    found = pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        {
            "path": found["filePath"],
            "fileName": found["filePath"].map(lambda path: PurePosixPath(str(path)).name),
            "deployment": found["deploymentID"],
            "label": found["label"],
        }
    ).reindex(columns=LABEL_COLUMNS)


def observation_label(observation_type, scientific_name) -> str | None:
    """Return the label for a Camtrap DP observation: its scientific name for animals,
    ``blank`` or ``human`` for those types, and ``None`` for observations that do not
    say what is in the image (unknown, unclassified, or an animal without a name)."""
    if observation_type == "animal":
        return scientific_name if isinstance(scientific_name, str) and scientific_name else None
    if observation_type in ("blank", "human", "vehicle"):
        return observation_type
    return None


def read_wildlife_insights(source: Path) -> pd.DataFrame:
    """Return one label per row of a Wildlife Insights ``images.csv`` export, with the
    taxonomy it gives."""
    table = pd.read_csv(source, dtype=str)
    rows = []
    for row in table.to_dict("records"):
        ranks = {rank: clean(row.get(rank)) for rank in RANKS}
        genus, species = ranks["genus"], clean(row.get("species"))
        if str(row.get("is_blank")).strip() in ("1", "1.0", "True", "true"):
            label = "blank"
        elif genus and species:
            label = f"{genus} {species}"
        else:
            label = next((ranks[rank] for rank in RANKS if ranks[rank]), None)
        location = str(row.get("location") or "")
        path = location.split("://", 1)[-1].split("/", 1)[-1] if location else row["filename"]
        rows.append(
            {"path": path, "fileName": row["filename"], "deployment": row.get("deployment_id"), "label": label, **ranks}
        )
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def clean(value) -> str | None:
    """Return a taxonomy value, or ``None`` when it is empty or a placeholder such as
    ``No CV Result``."""
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in UNKNOWN_NAMES:
        return None
    return value.strip()


def read_csv(
    source: Path,
    species_column: str,
    path_column: str | None = None,
    file_column: str | None = None,
    folder_column: str | None = None,
) -> pd.DataFrame:
    """Return one label per row of a spreadsheet, such as a Timelapse export. The image
    is ``path_column``, or ``folder_column`` joined with ``file_column`` (Timelapse's
    ``RelativePath`` and ``File`` are used when no columns are given). Rows with an
    empty species are left out, as they are not labelled yet."""
    table = pd.read_csv(source, dtype=str)
    if species_column not in table.columns:
        raise ValueError(f"no column {species_column!r} in {source.name}; columns: {', '.join(table.columns)}")
    if path_column is None and file_column is None and "File" in table.columns:
        file_column = "File"
        folder_column = folder_column or ("RelativePath" if "RelativePath" in table.columns else None)
    if path_column is not None:
        paths = table[path_column].fillna("")
    elif file_column is not None:
        folders = table[folder_column].fillna("") if folder_column else ""
        paths = (folders + "/" + table[file_column].fillna("")) if folder_column else table[file_column].fillna("")
    else:
        raise ValueError("give the image column with --path-column, or --file-column (and --folder-column)")
    paths = paths.str.replace("\\", "/", regex=False).str.strip("/")
    labels = table[species_column].fillna("").str.strip()
    found = pd.DataFrame(
        {
            "path": paths,
            "fileName": paths.map(lambda path: PurePosixPath(path).name),
            "deployment": None,
            "label": labels.where(labels != ""),
        }
    )
    return found.dropna(subset=["label"]).reindex(columns=LABEL_COLUMNS)


def speciesnet_taxonomy() -> pd.DataFrame | None:
    """Return SpeciesNet's taxonomy (scientific name, common name and ranks of about
    3,500 taxa) if its model files have been downloaded by ``detect``, without
    downloading anything."""
    cache = Path(os.environ.get("KAGGLEHUB_CACHE", Path.home() / ".cache" / "kagglehub"))
    files = sorted(cache.glob("models/google/speciesnet/*/*/*/taxonomy_release*.txt"))
    if not files:
        return None
    rows = []
    for line in files[-1].read_text(encoding="utf-8").splitlines():
        parts = line.split(";")
        if len(parts) != 7:
            continue
        scientific_name, common_name = parse_taxon(line)
        if scientific_name:
            ranks = [part.capitalize() for part in parts[4:5]] + [part.capitalize() for part in parts[3:0:-1]]
            rows.append([scientific_name, common_name, *ranks])
    return pd.DataFrame(rows, columns=["scientificName", "commonName", *RANKS])


def resolve_names(
    labels: pd.Series, names_file: Path | None = None, taxonomy: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Return, for every distinct label, its scientific name and ranks, and the labels
    that could not be recognised.

    Blank and human labels are standardised. A ``names_file`` (columns ``label`` and
    ``scientificName``) maps labels first; then SpeciesNet's taxonomy recognises
    scientific names in any capitalisation and common names such as ``red fox``.
    Labels it does not know are kept as they are and reported.
    """
    mapping = {}
    if names_file is not None:
        names = pd.read_csv(names_file, dtype=str)
        mapping = {str(row.label).strip().lower(): row.scientificName for row in names.itertuples()}
    by_scientific = by_common = {}
    if taxonomy is not None:
        by_scientific = {name.lower(): row for name, row in taxonomy.set_index("scientificName", drop=False).iterrows()}
        by_common = {}
        for row in taxonomy.itertuples():
            by_common.setdefault(row.commonName.lower(), taxonomy.loc[row.Index])
    rows, unknown = [], []
    for label in sorted(set(labels.dropna())):
        key = label.strip().lower()
        target = mapping.get(key, label.strip())
        target_key = str(target).strip().lower()
        ranks = dict.fromkeys(RANKS)
        if target_key in BLANK_VALUES or (isinstance(target, float) and pd.isna(target)):
            name = ""
        elif target_key in HUMAN_NAMES:
            name = "human"
        elif target_key == "vehicle":
            name = "vehicle"
        elif target_key in by_scientific:
            match = by_scientific[target_key]
            name, ranks = match["scientificName"], {rank: match[rank] or None for rank in RANKS}
        elif target_key in by_common:
            match = by_common[target_key]
            name, ranks = match["scientificName"], {rank: match[rank] or None for rank in RANKS}
        else:
            name = str(target).strip()
            if taxonomy is not None and key not in mapping:
                unknown.append(label)
        rows.append({"label": label, "scientificName": name, **ranks})
    return pd.DataFrame(rows, columns=["label", "scientificName", *RANKS]), unknown


def match_images(labels: pd.DataFrame, media: pd.DataFrame) -> tuple[pd.Series, int, int]:
    """Return the package ``filePath`` for every label (``None`` when not found), and
    how many labels matched no image and more than one image.

    A label's path is matched as it is, then without its leading folders, so paths
    from another computer still match; otherwise its file name is matched when it is
    unique in the package, or unique within the label's deployment.
    """
    known = set(media["filePath"])
    by_name = media.groupby("fileName")["filePath"].agg(list).to_dict()
    deployment_of = dict(zip(media["filePath"], media["deploymentID"], strict=True))
    matched, missing, ambiguous = [], 0, 0
    for row in labels.itertuples():
        path = match_path(str(row.path), known) if isinstance(row.path, str) and row.path else None
        if path is None:
            candidates = by_name.get(row.fileName, [])
            if len(candidates) > 1 and isinstance(row.deployment, str):
                candidates = [candidate for candidate in candidates if deployment_of[candidate] == row.deployment]
            if len(candidates) == 1:
                path = candidates[0]
            elif candidates:
                ambiguous += 1
            else:
                missing += 1
        matched.append(path)
    return pd.Series(matched, index=labels.index, dtype=object), missing, ambiguous


def run(
    package_dir: Path,
    source: Path,
    out_file: Path,
    source_format: str = "auto",
    species_column: str | None = None,
    path_column: str | None = None,
    file_column: str | None = None,
    folder_column: str | None = None,
    names_file: Path | None = None,
) -> dict:
    """Write the labels in ``source`` as a truth file for the package's images and
    return a summary: the format, rows read, images matched, unmatched and ambiguous
    rows, images per label, and labels that were not recognised."""
    source_format = detect_format(source) if source_format == "auto" else source_format
    if source_format == "camtrap-dp":
        labels = read_camtrap_dp(source)
    elif source_format == "wildlife-insights":
        labels = read_wildlife_insights(source)
    else:
        if species_column is None:
            raise ValueError("give the column with the species or label with --species-column")
        labels = read_csv(source, species_column, path_column, file_column, folder_column)
    rows_read = len(labels)
    labels = labels.dropna(subset=["label"])

    media = pd.read_csv(package_dir / "media.csv", dtype=str)
    labels["filePath"], missing, ambiguous = match_images(labels, media)
    names, unknown = resolve_names(labels["label"], names_file, speciesnet_taxonomy())
    resolved = labels.drop(columns=RANKS).merge(names, on="label", how="left")
    given = labels.groupby("label")[RANKS].first()
    for rank in RANKS:
        resolved[rank] = resolved[rank].fillna(resolved["label"].map(given[rank]))
    truth = resolved.dropna(subset=["filePath"])
    truth = truth.sort_values("scientificName", key=lambda names: names.eq("")).drop_duplicates("filePath")
    truth = truth.sort_values("filePath")[TRUTH_COLUMNS]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    truth.to_csv(out_file, index=False)
    counts = truth["scientificName"].replace("", "blank").value_counts()
    return {
        "format": source_format,
        "rows": rows_read,
        "matched": len(truth),
        "missing": missing,
        "ambiguous": ambiguous,
        "labels": counts,
        "unknown": unknown,
        "packageImages": len(media),
    }
