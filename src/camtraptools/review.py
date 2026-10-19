"""Local web app for reviewing the AI's observations event by event."""

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from jinja2 import Environment, PackageLoader, select_autoescape

from camtraptools.detect import OBSERVATIONS_FILE, RESULTS_FILE, parse_taxon

templates = Environment(loader=PackageLoader("camtraptools"), autoescape=select_autoescape())


def load_review_data(package_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the package's media and observations, with event-level rows required."""
    media = pd.read_csv(package_dir / "media.csv", dtype=str).set_index("mediaID", drop=False)
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    if not (observations["observationLevel"] == "event").any():
        raise ValueError(f"no events in {package_dir / OBSERVATIONS_FILE}; run `camtraptools events` first")
    return media, observations


def common_names(package_dir: Path) -> dict[str, str]:
    """Return common names by scientific name, from SpeciesNet's labels if available."""
    results_file = package_dir / RESULTS_FILE
    if not results_file.exists():
        return {}
    names = {}
    for prediction in json.loads(results_file.read_text(encoding="utf-8"))["predictions"]:
        for detection in prediction.get("detections") or []:
            label = (detection.get("processed_classification") or {}).get("class")
            if label:
                scientific_name, common_name = parse_taxon(label)
                if scientific_name:
                    names[scientific_name] = common_name
    return names


def create_app(package_dir: Path, images_dir: Path) -> FastAPI:
    """Return the review web app for the package, serving images from ``images_dir``."""
    media, observations = load_review_data(package_dir)
    names = common_names(package_dir)
    events = observations[observations["observationLevel"] == "event"].sort_values(["eventStart", "eventID"])
    event_ids = list(events["eventID"])
    app = FastAPI(title="CamTrapTools review")

    @app.get("/")
    def first_event() -> RedirectResponse:
        return RedirectResponse(f"/events/{event_ids[0]}")

    @app.get("/events/{event_id}", response_class=HTMLResponse)
    def show_event(event_id: str) -> str:
        if event_id not in event_ids:
            raise HTTPException(status_code=404, detail=f"no event {event_id}")
        position = event_ids.index(event_id)
        event = events[events["eventID"] == event_id].iloc[0]
        detections = observations[(observations["eventID"] == event_id) & (observations["observationLevel"] == "media")]
        images = [
            {"media": media.loc[media_id], "detections": group.dropna(subset=["bboxX"]).to_dict("records")}
            for media_id, group in detections.groupby("mediaID", sort=False)
        ]
        return templates.get_template("event.html").render(
            event=event,
            images=images,
            names=names,
            position=position + 1,
            total=len(event_ids),
            previous_id=event_ids[position - 1] if position > 0 else None,
            next_id=event_ids[position + 1] if position + 1 < len(event_ids) else None,
        )

    @app.get("/media/{media_id:path}")
    def image(media_id: str) -> FileResponse:
        if media_id not in media.index:
            raise HTTPException(status_code=404, detail="no such image in the package")
        return FileResponse(images_dir / media.loc[media_id, "filePath"])

    return app
