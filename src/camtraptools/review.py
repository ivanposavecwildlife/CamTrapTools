"""Local web app for reviewing the AI's observations event by event."""

import difflib
import json
import secrets
import socket
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from jinja2 import Environment, PackageLoader, select_autoescape

from camtraptools import reviews
from camtraptools.detect import OBSERVATIONS_FILE, RESULTS_FILE, parse_taxon
from camtraptools.labels import Taxonomy, normalise, speciesnet_taxonomy
from camtraptools.reviews import ORDERS, event_confidence, order_events  # noqa: F401

templates = Environment(loader=PackageLoader("camtraptools"), autoescape=select_autoescape())


def load_review_data(package_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the package's media and observations, with event-level rows required."""
    media = pd.read_csv(package_dir / "media.csv", dtype=str).set_index("mediaID", drop=False)
    observations = pd.read_csv(package_dir / OBSERVATIONS_FILE, dtype=str)
    if not (observations["observationLevel"] == "event").any():
        raise ValueError(f"no events in {package_dir / OBSERVATIONS_FILE}; run `camtraptools events` first")
    return media, observations


def video_frames(package_dir: Path) -> dict[str, str]:
    """Return the best frame saved by ``detect`` for each video, by file path."""
    results_file = package_dir / RESULTS_FILE
    if not results_file.exists():
        return {}
    predictions = json.loads(results_file.read_text(encoding="utf-8"))["predictions"]
    return {prediction["filepath"]: prediction["frame"] for prediction in predictions if prediction.get("frame")}


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


class Names:
    """The taxa a reviewer can choose: every name in the survey and, once ``detect``
    has downloaded SpeciesNet, every taxon in its taxonomy, with common names."""

    def __init__(self, survey_taxa: set[str], common: dict[str, str], taxonomy: pd.DataFrame | None):
        self.taxonomy = Taxonomy(taxonomy)
        self.common = dict(common)
        if taxonomy is not None:
            for row in taxonomy.itertuples():
                self.common.setdefault(row.scientificName, row.commonName)
        self.taxa = sorted(set(survey_taxa) | set(self.common))
        self.by_key = {normalise(name): name for name in self.taxa}

    def resolve(self, name: str) -> str:
        """Return the scientific name a reviewer meant by ``name``: a scientific or
        common name in any capitalisation (``aves``: Aves, ``meerkat``: Suricata
        suricatta, ``caracal``: Caracal caracal), or a new two-word scientific name,
        capitalised as one. Raises ValueError, with suggestions, for names that match
        nothing known or look like a misspelt one (``brown hyena``)."""
        match = self.taxonomy.find(name)
        if match is not None:
            return match["scientificName"]
        found = self.by_key.get(normalise(name))
        if found is not None:
            return found
        words = name.split()
        close = self.suggestions(name)
        if not close and (len(words) == 2 or (len(words) == 1 and not self.taxonomy.by_scientific)):
            return " ".join([words[0].capitalize(), *(word.lower() for word in words[1:])])
        hint = f"did you mean {', '.join(close)}?" if close else "type its scientific name, e.g. Suricata suricatta"
        raise ValueError(f"'{name}' is not a species or group SpeciesNet knows; {hint}")

    def suggestions(self, name: str, limit: int = 4) -> list[str]:
        """Return known names a reviewer may have meant: common names containing the
        typed word (``jackal``), then names spelt almost the same (``brown hyena``)."""
        key = normalise(name)
        labels = {normalise(scientific): scientific for scientific in self.taxa}
        labels |= {normalise(common): f"{common} ({scientific})" for scientific, common in self.common.items() if common}
        containing = [label for found, label in labels.items() if len(key) >= 3 and key in found.split()]
        close = [labels[found] for found in difflib.get_close_matches(key, labels, n=limit, cutoff=0.8)]
        return list(dict.fromkeys(close + sorted(containing)))[:limit]


def free_port(preferred: int) -> int:
    """Return ``preferred`` if it is free on this computer, otherwise any free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]


def decision_label(review) -> str:
    """Return a short text for one review decision, e.g. ``Oryx gazella (3)``."""
    label = review.scientificName if isinstance(review.scientificName, str) else review.observationType
    return f"{label} ({review.count})" if isinstance(review.count, str) else label


def planned_first(package_dir: Path, events: pd.DataFrame) -> pd.DataFrame:
    """Return the events with those of a fresh review plan first, in the plan's
    order, followed by the rest least confident first."""
    from camtraptools import plan

    _, planned, _ = plan.run(package_dir)
    position = {event_id: number for number, event_id in enumerate(planned["eventID"])}
    rank = events["eventID"].map(position).fillna(len(position))
    return events.assign(rank=rank).sort_values("rank", kind="stable").drop(columns="rank").reset_index(drop=True)


def create_app(package_dir: Path, images_dir: Path, reviewer: str = "reviewer", order: str = "uncertainty") -> FastAPI:
    """Return the review web app for the package, serving images from ``images_dir``,
    recording decisions under the name ``reviewer`` and showing events in ``order``."""
    media, observations = load_review_data(package_dir)
    names = common_names(package_dir)
    frames = video_frames(package_dir)
    machine = observations[observations["classificationMethod"] != "human"]
    events = order_events(machine[machine["observationLevel"] == "event"], machine, order)
    if order == "plan":
        events = planned_first(package_dir, events)
    event_ids = list(events["eventID"])
    known = Names(set(machine["scientificName"].dropna()), names, speciesnet_taxonomy())
    token = secrets.token_urlsafe(16)
    app = FastAPI(title="CamTrapTools review")

    def reviewed_ids() -> set[str]:
        return set(reviews.load_reviews(package_dir)["eventID"])

    def next_to_review(after: int = -1) -> str | None:
        done = reviewed_ids()
        for event_id in event_ids[after + 1 :] + event_ids[: after + 1]:
            if event_id not in done:
                return event_id
        return None

    @app.get("/")
    def start() -> RedirectResponse:
        event_id = next_to_review()
        return RedirectResponse(f"/events/{event_id}" if event_id else "/events")

    @app.get("/events", response_class=HTMLResponse)
    def list_events() -> str:
        done = {
            event_id: " + ".join(decision_label(row) for row in rows.itertuples())
            for event_id, rows in reviews.load_reviews(package_dir).groupby("eventID", sort=False)
        }
        return templates.get_template("events.html").render(
            events=events.to_dict("records"), done=done, names=names, order=order
        )

    @app.get("/events/{event_id}", response_class=HTMLResponse)
    def show_event(event_id: str) -> str:
        if event_id not in event_ids:
            raise HTTPException(status_code=404, detail=f"no event {event_id}")
        return render_event(event_id)

    def render_event(event_id: str, error: str | None = None) -> str:
        position = event_ids.index(event_id)
        event = events.iloc[position]
        detections = observations[(observations["eventID"] == event_id) & (observations["observationLevel"] == "media")]
        detections = detections[detections["classificationMethod"] != "human"]
        images = [
            {
                "media": media.loc[media_id],
                "detections": group.dropna(subset=["bboxX"]).to_dict("records"),
                "frame": frames.get(media.loc[media_id, "filePath"]),
                "video": str(media.loc[media_id].get("fileMediatype", "")).startswith("video/"),
            }
            for media_id, group in detections.groupby("mediaID", sort=False)
        ]
        existing = reviews.load_reviews(package_dir)
        reviewed = len(set(existing["eventID"]) & set(event_ids))
        existing = existing[existing["eventID"] == event_id]
        return templates.get_template("event.html").render(
            event=event,
            review=existing.iloc[0] if not existing.empty else None,
            others=existing.iloc[1:].to_dict("records"),
            error=error,
            review_types=reviews.REVIEW_TYPES,
            taxa=known.taxa,
            token=token,
            reviewer=reviewer,
            images=images,
            names=known.common,
            order=order,
            reviewed=reviewed,
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
        extra_name: Annotated[list[str] | None, Form()] = None,
        extra_count: Annotated[list[str] | None, Form()] = None,
    ) -> Response:
        if not secrets.compare_digest(form_token, token):
            raise HTTPException(status_code=403, detail="invalid form token; reload the page")
        if event_id not in event_ids:
            raise HTTPException(status_code=404, detail=f"no event {event_id}")
        try:
            name = known.resolve(scientific_name.strip()) if scientific_name.strip() else None
            others = [
                (known.resolve(other.strip()), int(number) if number.strip() else 1)
                for other, number in zip(extra_name or [], extra_count or [], strict=False)
                if other.strip()
            ]
            reviews.save_review(
                package_dir,
                event_id,
                observation_type,
                name,
                int(count) if count.strip() else None,
                reviewer,
                others,
            )
        except ValueError as error:
            return HTMLResponse(render_event(event_id, str(error)), status_code=400)
        next_id = next_to_review(event_ids.index(event_id))
        return RedirectResponse(f"/events/{next_id}" if next_id else "/events", status_code=303)

    @app.get("/media/{media_id:path}")
    def image(media_id: str) -> FileResponse:
        if media_id not in media.index:
            raise HTTPException(status_code=404, detail="no such image in the package")
        return FileResponse(images_dir / media.loc[media_id, "filePath"])

    @app.get("/frame/{media_id:path}")
    def frame(media_id: str) -> FileResponse:
        if media_id not in media.index or media.loc[media_id, "filePath"] not in frames:
            raise HTTPException(status_code=404, detail="no frame for this video")
        return FileResponse(package_dir / frames[media.loc[media_id, "filePath"]])

    return app
