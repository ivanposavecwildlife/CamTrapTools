"""Download part of Orinoquía Camera Traps from LILA BC for the second validation.

Orinoquía Camera Traps (University of Minnesota) has 112,000 images from about 50
cameras in two private reserves in Meta, Colombia, from January to July 2020, labelled
by experts in Wildlife Insights. The whole set is 72 GB, so this script downloads every
image taken in a window of days (two weeks from 11 March 2020 by default, about 6,400
images and 4.4 GB), from all cameras. Images labelled as people are left out.

Writes, under the output folder:

- ``metadata/orinoquia_camera_traps.json``: the labels in COCO Camera Traps format,
  for ``camtraptools labels``.
- ``sd_cards/<camera>/…``: the images, one folder per camera.

Usage (with the ``detect`` extra installed):

    uv run python examples/orinoquia/prepare.py orinoquia [--start 2020-03-11] [--days 14]

The data is published on LILA BC under the Community Data License Agreement
(permissive). Downloads resume: existing images are kept.
"""

import argparse
import io
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

LILA = "https://storage.googleapis.com/public-datasets-lila/orinoquia-camera-traps/"
METADATA = LILA + "orinoquia_camera_traps_metadata.zip"
IMAGES = LILA + "public/"
METADATA_FILE = "orinoquia_camera_traps.json"


def metadata(out: Path) -> dict:
    """Download (once) and return the COCO Camera Traps metadata."""
    path = out / "metadata" / METADATA_FILE
    if not path.exists():
        response = requests.get(METADATA, timeout=120)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            path.write_bytes(archive.read(METADATA_FILE))
    return json.loads(path.read_text(encoding="utf-8"))


def chosen_images(data: dict, start: str, days: int) -> list[str]:
    """Return the file names of the images taken in the window, without people."""
    images = pd.DataFrame(data["images"])
    annotations = pd.DataFrame(data["annotations"])
    names = {category["id"]: category["name"] for category in data["categories"]}
    people = set(annotations.loc[annotations["category_id"].map(names) == "human", "image_id"])
    taken = pd.to_datetime(images["datetime"]).dt.tz_localize(None)
    first = pd.Timestamp(start)
    inside = (taken >= first) & (taken < first + pd.Timedelta(days=days)) & ~images["id"].isin(people)
    return sorted(images.loc[inside, "file_name"])


sessions = threading.local()


def get(url: str) -> requests.Response | None:
    """Return the response for ``url``, reusing one connection per thread (opening a
    new one per image runs out of network ports on large downloads) and retrying
    network errors a few times; ``None`` if it keeps failing."""
    if not hasattr(sessions, "session"):
        sessions.session = requests.Session()
    for attempt in range(4):
        try:
            return sessions.session.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(2**attempt)
    return None


def download(file_name: str, sd_cards: Path) -> str:
    """Download one image unless it is already there; return what happened."""
    target = sd_cards / file_name
    if target.exists():
        return "already downloaded"
    response = get(IMAGES + file_name)
    if response is None:
        return "failed (network error); run the script again"
    if response.status_code != 200:
        return f"not available (HTTP {response.status_code})"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    partial.write_bytes(response.content)
    partial.rename(target)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="Folder to download into.")
    parser.add_argument("--start", default="2020-03-11", help="First day of the window (YYYY-MM-DD).")
    parser.add_argument("--days", type=int, default=14, help="Length of the window in days.")
    arguments = parser.parse_args()
    files = chosen_images(metadata(arguments.out), arguments.start, arguments.days)
    print(f"Downloading {len(files)} images from {arguments.start} for {arguments.days} days")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = pd.Series(list(pool.map(lambda name: download(name, arguments.out / "sd_cards"), files)), index=files)
    for name in results.index[results.str.startswith("failed")]:
        results[name] = download(name, arguments.out / "sd_cards")
    print(results.value_counts().to_string())


if __name__ == "__main__":
    main()
