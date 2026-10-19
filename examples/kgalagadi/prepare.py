"""Download Snapshot Kgalagadi Season 1 from LILA BC for the regional case study.

Writes, under the output folder:

- ``sd_cards/<camera>/<roll>/<image>``: every image with an animal (about 2,300
  images, 2 GB), one folder per camera. Blank images and images of people are left out.
- ``truth.csv``: the volunteer label of each image (``filePath``, ``scientificName``
  and the taxon's genus, family, order and class), for ``crops`` and ``evaluate``.

Usage (with the ``detect`` extra installed):

    uv run python examples/kgalagadi/prepare.py kgalagadi

Snapshot Kgalagadi is published by Snapshot Safari on LILA BC under the Community
Data License Agreement (permissive). Downloads resume: existing images are kept.
"""

import io
import sys
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


def image_labels() -> pd.DataFrame:
    """Return one row per image and species from LILA's Season 1 metadata."""
    response = requests.get(METADATA, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        captures = pd.read_csv(archive.open("KGA_S1_report_lila.csv"))
        images = pd.read_csv(archive.open("KGA_S1_report_lila_image_inventory.csv"))
    return images.merge(captures, on="capture_id")


def download(path: str, sd_cards: Path) -> str:
    """Download one image unless it is already there; return what happened."""
    target = sd_cards / path.removeprefix("KGA_S1/")
    if target.exists():
        return "already downloaded"
    response = requests.get(IMAGES + path, timeout=60)
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
        results = pd.Series(list(pool.map(lambda path: download(path, out / "sd_cards"), paths)))
    print(results.value_counts().to_string())

    truth = animals[animals["question__species"].isin(TAXA)].copy()
    truth["filePath"] = truth["image_path_rel"].str.removeprefix("KGA_S1/")
    columns = ["scientificName", "genus", "family", "order", "class"]
    truth[columns] = pd.DataFrame(truth["question__species"].map(TAXA).tolist(), index=truth.index)
    truth[["filePath", *columns]].to_csv(out / "truth.csv", index=False)
    print(f"Wrote {out / 'truth.csv'} with {len(truth)} labels")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python prepare.py OUTPUT_FOLDER")
    main(Path(sys.argv[1]))
