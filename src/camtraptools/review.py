"""Local web app for reviewing the AI's observations event by event."""

import json
import secrets
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from jinja2 import Environment, PackageLoader, select_autoescape

from camtraptools import reviews
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


def create_app(package_dir: Path, images_dir: Path, reviewer: str = "reviewer") -> FastAPI:
    """Return the review web app for the package, serving images from ``images_dir``
    and recording decisions under the name ``reviewer``."""
    media, observations = load_review_data(package_dir)
    names = common_names(package_dir)
    machine = observations[observations["classificationMethod"] != "human"]
    events = machine[machine["observationLevel"] == "event"].sort_values(["eventStart", "eventID"])
    event_ids = list(events["eventID"])
    taxa = sorted(set(machine["scientificName"].dropna()) | set(names))
    token = secrets.token_urlsafe(16)
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
        existing = reviews.load_reviews(package_dir)
        existing = existing[existing["eventID"] == event_id]
        return templates.get_template("event.html").render(
            event=event,
            review=existing.iloc[0] if not existing.empty else None,
            review_types=reviews.REVIEW_TYPES,
            taxa=taxa,
            token=token,
            reviewer=reviewer,
            images=images,
            names=names,
            position=position + 1,
            total=len(event_ids),
            previous_id=event_ids[position - 1] if position > 0 else None,
            next_id=event_ids[position + 1] if position + 1 < len(event_ids) else None,
        )

    @app.post("/events/{event_id}/review")
    def save_review(
        event_id: str,
        form_token: Annotated[str, Form(alias="token")],
        observation_type: Annotated[str, Form()],
        scientific_name: Annotated[str, Form()] = "",
        count: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        if not secrets.compare_digest(form_token, token):
            raise HTTPException(status_code=403, detail="invalid form token; reload the page")
        if event_id not in event_ids:
            raise HTTPException(status_code=404, detail=f"no event {event_id}")
        try:
            reviews.save_review(
                package_dir,
                event_id,
                observation_type,
                scientific_name.strip() or None,
                int(count) if count.strip() else None,
                reviewer,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        position = event_ids.index(event_id)
        next_id = event_ids[position + 1] if position + 1 < len(event_ids) else event_id
        return RedirectResponse(f"/events/{next_id}", status_code=303)

    @app.get("/media/{media_id:path}")
    def image(media_id: str) -> FileResponse:
        if media_id not in media.index:
            raise HTTPException(status_code=404, detail="no such image in the package")
        return FileResponse(images_dir / media.loc[media_id, "filePath"])

    return app
