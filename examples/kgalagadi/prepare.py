"""Download Snapshot Kgalagadi Season 1 from LILA BC for the regional case study.

Writes, under the output folder:

- ``sd_cards/<camera>/<roll>/<image>``: every image with an animal (about 2,300
  images, 2 GB), one folder per camera. Blank images and images of people are left out.
- ``truth.csv``: the volunteer label of each image (``filePath``, ``scientificName``
  and the taxon's genus, family, order and class), for ``crops`` and ``evaluate``.
- ``field_sheet.csv``: one row per camera for ``ingest --field-sheet``, with the first
  and last capture of the season (blank captures included) as the deployment period.
  LILA does not publish camera coordinates, so every camera gets the park's approximate
  position.

Usage (with the ``detect`` extra installed):

    uv run python examples/kgalagadi/prepare.py kgalagadi
    uv run camtraptools ingest kgalagadi/sd_cards --out kgalagadi/package \
        --field-sheet kgalagadi/field_sheet.csv

Snapshot Kgalagadi is published by Snapshot Safari on LILA BC under the Community
Data License Agreement (permissive). Downloads resume: existing images are kept.
"""

import io
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

LILA = "https://storage.googleapis.com/public-datasets-lila/snapshot-safari/KGA/"
METADATA = LILA + "KGA_S1.csv.zip"
IMAGES = LILA + "KGA_public/"

# LILA species code: (scientific name, genus, family, order, class), using SpeciesNet's
# names where taxonomies differ. Codes not listed (people, reptiles and amphibians as
# one group) are left out of the truth file.
TAXA = {
    "aardvarkantbear": ("Orycteropus afer", "Orycteropus", "Orycteropodidae", "Tubulidentata", "Mammalia"),
    "birdofprey": ("Accipitriformes", "", "", "", "Aves"),
    "birdother": ("Aves", "", "", "", ""),
    "bustardkori": ("Ardeotis kori", "Ardeotis", "Otididae", "Otidiformes", "Aves"),
    "caracal": ("Caracal caracal", "Caracal", "Felidae", "Carnivora", "Mammalia"),
    "catafricanwild": ("Felis silvestris", "Felis", "Felidae", "Carnivora", "Mammalia"),
    "cheetah": ("Acinonyx jubatus", "Acinonyx", "Felidae", "Carnivora", "Mammalia"),
    "duikercommongrey": ("Sylvicapra grimmia", "Sylvicapra", "Bovidae", "Artiodactyla", "Mammalia"),
    "eland": ("Tragelaphus oryx", "Tragelaphus", "Bovidae", "Artiodactyla", "Mammalia"),
    "foxbateared": ("Otocyon megalotis", "Otocyon", "Canidae", "Carnivora", "Mammalia"),
    "foxcape": ("Vulpes chama", "Vulpes", "Canidae", "Carnivora", "Mammalia"),
    "gemsbokoryx": ("Oryx gazella", "Oryx", "Bovidae", "Artiodactyla", "Mammalia"),
    "harecape": ("Lepus capensis", "Lepus", "Leporidae", "Lagomorpha", "Mammalia"),
    "hartebeestred": ("Alcelaphus buselaphus", "Alcelaphus", "Bovidae", "Artiodactyla", "Mammalia"),
    "honeybadger": ("Mellivora capensis", "Mellivora", "Mustelidae", "Carnivora", "Mammalia"),
    "hyenabrown": ("Parahyaena brunnea", "Parahyaena", "Hyaenidae", "Carnivora", "Mammalia"),
    "hyenaspotted": ("Crocuta crocuta", "Crocuta", "Hyaenidae", "Carnivora", "Mammalia"),
    "jackalblackbacked": ("Canis mesomelas", "Canis", "Canidae", "Carnivora", "Mammalia"),
    "kudu": ("Tragelaphus strepsiceros", "Tragelaphus", "Bovidae", "Artiodactyla", "Mammalia"),
    "leopard": ("Panthera pardus", "Panthera", "Felidae", "Carnivora", "Mammalia"),
    "lionfemale": ("Panthera leo", "Panthera", "Felidae", "Carnivora", "Mammalia"),
    "meerkatsuricate": ("Suricata suricatta", "Suricata", "Herpestidae", "Carnivora", "Mammalia"),
    "ostrich": ("Struthio camelus", "Struthio", "Struthionidae", "Struthioniformes", "Aves"),
    "porcupine": ("Hystrix africaeaustralis", "Hystrix", "Hystricidae", "Rodentia", "Mammalia"),
    "rodent": ("Rodentia", "", "", "", "Mammalia"),
    "secretarybird": ("Sagittarius serpentarius", "Sagittarius", "Sagittariidae", "Accipitriformes", "Aves"),
    "steenbok": ("Raphicerus campestris", "Raphicerus", "Bovidae", "Artiodactyla", "Mammalia"),
    "wildebeestblue": ("Connochaetes taurinus", "Connochaetes", "Bovidae", "Artiodactyla", "Mammalia"),
}
SKIPPED_IMAGES = {"blank", "human"}
PARK_LATITUDE, PARK_LONGITUDE = -25.5, 20.5


def image_labels() -> pd.DataFrame:
    """Return one row per image and species from LILA's Season 1 metadata."""
    response = requests.get(METADATA, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        captures = pd.read_csv(archive.open("KGA_S1_report_lila.csv"))
        images = pd.read_csv(archive.open("KGA_S1_report_lila_image_inventory.csv"))
    return images.merge(captures, on="capture_id")


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


def download(path: str, sd_cards: Path) -> str:
    """Download one image unless it is already there; return what happened."""
    target = sd_cards / path.removeprefix("KGA_S1/")
    if target.exists():
        return "already downloaded"
    response = get(IMAGES + path)
    if response is None:
        return "failed (network error); run the script again"
    if response.status_code != 200:
        return f"not available (HTTP {response.status_code})"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    partial.write_bytes(response.content)
    partial.rename(target)
    return "downloaded"


def main(out: Path) -> None:
    labels = image_labels()
    animals = labels[~labels["question__species"].isin(SKIPPED_IMAGES)]
    paths = sorted(set(animals["image_path_rel"]))
    print(f"Downloading {len(paths)} images with animals to {out / 'sd_cards'}")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = pd.Series(list(pool.map(lambda path: download(path, out / "sd_cards"), paths)), index=paths)
    for path in results.index[results.str.startswith("failed")]:
        results[path] = download(path, out / "sd_cards")
    print(results.value_counts().to_string())

    truth = animals[animals["question__species"].isin(TAXA)].copy()
    truth["filePath"] = truth["image_path_rel"].str.removeprefix("KGA_S1/")
    columns = ["scientificName", "genus", "family", "order", "class"]
    truth[columns] = pd.DataFrame(truth["question__species"].map(TAXA).tolist(), index=truth.index)
    truth[["filePath", *columns]].to_csv(out / "truth.csv", index=False)
    print(f"Wrote {out / 'truth.csv'} with {len(truth)} labels")

    labels["capturedAt"] = labels["capture_date_local"] + "T" + labels["capture_time_local"]
    sheet = labels.groupby("site")["capturedAt"].agg(deploymentStart="min", deploymentEnd="max").reset_index()
    sheet = sheet.rename(columns={"site": "deploymentID"})
    sheet["locationName"] = "Kgalagadi " + sheet["deploymentID"]
    sheet["latitude"], sheet["longitude"] = PARK_LATITUDE, PARK_LONGITUDE
    sheet["deploymentComments"] = "Period from first and last capture; approximate park-level coordinates"
    columns = ["deploymentID", "locationName", "latitude", "longitude", "deploymentStart", "deploymentEnd"]
    sheet[[*columns, "deploymentComments"]].to_csv(out / "field_sheet.csv", index=False)
    print(f"Wrote {out / 'field_sheet.csv'} for {len(sheet)} cameras")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python prepare.py OUTPUT_FOLDER")
    main(Path(sys.argv[1]))
