# 📷🛠️ CamTrapTools

**Open-source toolkit for camera-trap projects: from SD card to ecological answers.**

[![CI](https://github.com/ivanposavecwildlife/CamTrapTools/actions/workflows/ci.yml/badge.svg)](https://github.com/ivanposavecwildlife/CamTrapTools/actions/workflows/ci.yml)
![status: alpha](https://img.shields.io/badge/status-alpha-orange)
![python: 3.11–3.14](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)
[![PyPI](https://img.shields.io/pypi/v/camtraptools)](https://pypi.org/project/camtraptools/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23500635.svg)](https://doi.org/10.5281/zenodo.23500635)

Camera traps are one of the main ways conservationists monitor wildlife, but a single
project can produce millions of images. Field teams routinely lose months of data to
cameras that died, fell over or had their clock reset, and nobody notices until the
analysis is already wrong.

CamTrapTools turns a folder of SD-card images into a standard
[Camtrap DP](https://camtrap-dp.tdwg.org) data package, checks every camera for problems,
finds and identifies animals with [SpeciesNet](https://github.com/google/cameratrapai),
counts independent visits, lets a person review the AI's least confident decisions
first, adapts the AI to the region, estimates occupancy and daily activity, and exports
a validated package ready to publish on [GBIF](https://www.gbif.org).

```
SD cards ─► ingest ─► check / report ─► detect (or import) ─► events ─► review ─► ecology ─► export ─► GBIF
                                             ▲                            │
                                             └── evaluate / crops / adapt ┘
```

![Camera health report showing a timeline per camera with highlighted gaps](https://raw.githubusercontent.com/ivanposavecwildlife/CamTrapTools/main/docs/report.png)

*Health report for eight simulated cameras: a dead battery (`cam_dead`), a camera knocked
over for a month (`cam_gap`), wind-triggered images (`cam_vegetation`), a broken infrared
flash (`cam_noflash`) and clock errors.*

## Features

- **`ingest`**: scans SD-card folders (one per camera), reads capture times from EXIF and
  writes `media.csv` and `deployments.csv` in Camtrap DP format. With `--videos`, MP4,
  MOV and AVI clips are included too, with recording times read from the files, and
  `detect` identifies animals in frames sampled from them. An optional CSV or Excel
  **field sheet** adds coordinates, camera details and the recorded setup and pickup times.
- **Field-sheet validation**: duplicate IDs, impossible coordinates, unreadable dates and
  camera names that don't match their folders, with suggestions such as
  *"did you mean 'camera-B'?"*
- **`check`**: reports camera problems for each deployment:

  | Problem | How it is detected |
  |---|---|
  | Clock reset or set wrongly | images dated before 2010 or in the future; time going backwards between consecutive files |
  | Dead battery or full card | no images in the last 21+ days before pickup |
  | Camera knocked over or covered | no images for 21+ days mid-deployment (adjustable with `--max-gap-days`) |
  | Vegetation or sun glare | more than 100 images in one day |
  | Broken infrared flash | no night images out of 50 or more |
  | Wrong setup date on the field sheet | no images in the first 21+ days (`ingest` also reports images taken before the recorded start) |

- **`report`**: a single self-contained HTML page with headline numbers, an interactive
  timeline, a summary table and all issues. It works offline, supports dark mode and can
  be emailed as one file.
- **`detect`**: runs SpeciesNet, which finds animals, people and vehicles with
  MegaDetector v5a and identifies each animal, with geofencing by country
  (`--country ZAF`). Observations record the species (or a higher taxon such as *Bovidae*
  when the model is unsure), bounding box, confidence and model version. Changing
  `--threshold` reuses existing results instantly.
- **`events`**: groups images into independent visits (30-minute rule, adjustable with
  `--gap`), chooses each visit's species by a confidence-weighted vote across its images,
  counts the most animals seen in any single image, and prints visits per species and
  camera.
- **`import`**: reads results from MegaDetector, AddaxAI, Timelapse or SpeciesNet, even
  when their image paths are relative to a different folder, so teams can adopt
  CamTrapTools mid-project without re-running the AI.
- **`review`**: a local web app for checking the AI event by event, **least confident
  first**. Each page shows the images with bounding boxes, the AI's decision and
  confidence, and a form pre-filled with that decision: press Enter to accept, `B` for
  blank, or correct the species and count. Decisions are saved as human observations
  next to the machine's, and `review-stats` reports agreement with the AI, corrections
  by confidence and review speed.

  ![Review app showing a camera event with the AI's boxes and a human correction](https://raw.githubusercontent.com/ivanposavecwildlife/CamTrapTools/main/docs/review.jpg)

- **`labels`**: imports labels you already have, from Timelapse, Wildlife Insights,
  another Camtrap DP package or any spreadsheet, as a truth file for `evaluate`, `crops`
  and `compare`. It matches images even when paths differ and turns common names into
  scientific names.
- **`evaluate`**: measures the AI against your reviews or a file of true labels:
  precision and recall per species, the confidence above which each species' answers
  can be trusted, and an HTML report of every mistake with its images.
- **`crops`**: cuts every labelled animal detection into a training dataset for
  fine-tuning a classifier (`train/`, `val/`, `test/`, one folder per species). Whole
  camera locations go to one split, so the model is tested on backgrounds it has never
  seen, and the command warns about species that are missing from training.
- **`adapt`**: adapts SpeciesNet to your region. SpeciesNet's camera-trap backbone stays
  frozen and a new final layer is trained on your crops, only for the species in them,
  starting from SpeciesNet's own weights for species it already knows. It runs on a
  laptop (using an Apple or NVIDIA GPU when available), keeps the epoch with the lowest
  validation loss, and compares the adapted layer with SpeciesNet and with SpeciesNet
  limited to the local species list, overall and per species. `--balanced` favours rare
  species over overall accuracy. It also picks a confidence threshold on the validation
  cameras: the lowest at which the adapted layer is as precise as SpeciesNet.
  `detect --adapted my_model` then uses the adapted answer for every animal above that
  threshold, unless SpeciesNet named a species the layer was never trained on.
- **`ecology`**: turns events into a detection history per species (camera sites by
  sampling occasions of `--occasion-days`, default 7: recorded, not recorded, or camera
  not running), the sampling effort per site and occasion, the relative abundance
  index (events per 100 camera-days) per species and site, and naive occupancy: the
  share of surveyed sites where each species was recorded, with its weekly detection
  rate. For species recorded at three or more sites, an occupancy model (MacKenzie et
  al. 2002) corrects for animals being missed where they are present: it estimates
  occupancy with a 95% profile-likelihood interval and the chance of recording the
  species per occasion, using each occasion's camera days, and warns when detections are
  too rare for a reliable estimate. Site covariates from the field sheet (habitat,
  camera height…) can be added to occupancy and detection, with models compared by
  AIC. For species with 20 or more events it also draws
  daily activity curves (circular kernel densities, as in the R package `overlap`),
  reports each species' busiest time of day and the overlap coefficient between every
  pair of species with a 95% bootstrap interval, and saves the curves as a chart. A
  species accumulation curve and the Chao2 estimate show how many species the survey
  likely missed.
  Reviewed decisions replace the AI's, and higher taxa are left out unless asked for
  (`--all-taxa`, `--taxon`).
- **`summary`**: one HTML page to hand back to a project: key numbers, camera
  problems, a species table with abundance index, occupancy and activity, the activity
  and species-richness charts, how the species were identified, and the methods and
  their limits in plain language.
- **`compare`**: runs the ecology analyses on several labellings of the same survey
  (the AI's labels alone, with human reviews, ground truth from `--truth`, and other
  packages such as one with an adapted layer, `--other adapted=FOLDER`) and reports how
  far each is from the reference: species and species-at-site records missed or wrongly
  added, occupancy error, and how much activity patterns and overlaps change.
- **`export`**: writes a complete Camtrap DP 1.0.2 package (`datapackage.json` and three
  tables) with time zones, coverage, contributors and licenses, using the reviewer's
  decision wherever an event was reviewed. For privacy, `--exclude-humans` leaves out
  images of people and `--coordinate-decimals` rounds the cameras' coordinates.
  Exported packages pass
  [Frictionless](https://framework.frictionlessdata.io/) validation against the official
  Camtrap DP profile.

## Quick start

New to CamTrapTools? The [user guide](https://github.com/ivanposavecwildlife/CamTrapTools/blob/main/docs/user-guide.md) walks through a whole survey
step by step, with the output of every command.

Install from [PyPI](https://pypi.org/project/camtraptools/) with
[uv](https://docs.astral.sh/uv/) (recommended) or pip. The `detect` extra installs
SpeciesNet and PyTorch (about 1 GB); leave it out if you only need the quality checks.

```bash
uv tool install "camtraptools[detect]"
```

```bash
pip install "camtraptools[detect]"
```

```bash
camtraptools ingest sd_cards --out my_project --field-sheet field_sheet.xlsx
camtraptools check my_project
camtraptools report my_project --out report.html
camtraptools detect my_project --images sd_cards --country ZAF
camtraptools events my_project
camtraptools review my_project --images sd_cards --reviewer "Your Name"
camtraptools review-stats my_project
camtraptools evaluate my_project --report mistakes.html --images sd_cards
camtraptools crops my_project --images sd_cards --out my_crops
camtraptools adapt my_crops --out my_model
camtraptools detect my_project --images sd_cards --adapted my_model
camtraptools ecology my_project --out my_ecology
camtraptools summary my_project --out summary.html --title "My survey"
camtraptools compare my_project --truth labels.csv --out my_comparison
camtraptools export my_project --out publish --title "My survey" \
    --contributor "Your Name" --timezone Africa/Johannesburg
```

Already have results from another tool? Replace `detect` with
`camtraptools import my_project results.json`.

### Input layout

One folder per camera deployment; images can be nested at any depth (e.g. `DCIM/100MEDIA`).

```
sd_cards/
├── camera_A/DCIM/100MEDIA/IMG_0001.JPG …
└── camera_B/DCIM/100MEDIA/IMG_0001.JPG …
```

### Field sheet (optional)

A CSV or Excel file with one row per deployment. Column names follow Camtrap DP.

| Column | Required | Example |
|---|---|---|
| `deploymentID` | yes, matches the folder name | `camera_A` |
| `latitude`, `longitude` | yes | `52.7031`, `23.8452` |
| `deploymentStart`, `deploymentEnd` | recommended | `2024-05-01T09:15`, `2024-06-30` |
| `cameraTime`, `actualTime` | optional, to correct a wrong clock | `2010-01-01T00:04`, `2024-05-01T09:19` |
| any other Camtrap DP column | no | `locationName`, `cameraModel`, `cameraHeight`, `deploymentComments` |

Recorded start and end times take priority over the first and last image, which usually
underestimate how long a camera was running. Coordinates are required for `export`.

## Validated on real data

Tested first on four deployments from
[Snapshot Kgalagadi](https://lila.science/datasets/snapshot-kgalagadi/) (LILA BC):

- EXIF timestamps matched the published capture times **76 of 76** times.
- Without field-sheet dates, deployment periods inferred from images were **10–16% too
  short**, because setup photos had been removed for privacy. This is why recorded periods
  take priority.
- The default thresholds raise **no false alarms** on these desert cameras, where
  10–19 days without an animal is normal.
- At the default 0.2 threshold, the detector found all **50 of 50** captures with
  animals and all **26 of 26** empty captures were recognised as blank.
- SpeciesNet named **38 of 50** animal captures correctly to species, **10** correctly
  at a higher level (e.g. *Bovidae*, *Hyaenidae*), and got **2** wrong (a grey duiker
  as steenbok, a Cape fox as *Bovidae*). A close-up of jackal fur was called a vervet
  monkey with 94% confidence, a reminder that confidence is not accuracy.
- **194 images became 60 independent events**; at one camera, 64 gemsbok images were
  11 separate visits.
- Ordering events by AI confidence put **all 12 events the AI did not get exactly right
  among the first 13 of 60**; the remaining 47 were all correct. Reviewing the least
  confident 22% of events would have caught every mistake.
- Results imported from LILA's published MegaDetector run matched the toolkit's own
  detections, and every exported package passes Camtrap DP validation.

### Regional adaptation

On all of Snapshot Kgalagadi Season 1 (2,336 animal images from 20 cameras), adapting
SpeciesNet with `adapt` and using it in `detect` raised **correct independent events on
seven held-out cameras from 72 to 92 of 104**, and wrong events fell from 16 to 9. The
gain came mostly from overexposed night images, where correct events rose from 27 to 40
of 45, and none of the detections it changed on those cameras went from right to wrong.
The full method, per-species results and limitations are in the
[case study](https://github.com/ivanposavecwildlife/CamTrapTools/blob/main/docs/case-study-kgalagadi.md).

### Ecological consequences

From SpeciesNet's labels alone, 7 of 25 species and 41 of 113 species-at-site records
were missing compared with the volunteer labels: brown hyena, Cape fox and African
wildcat looked absent from the park, and steenbok looked more diurnal than it is.
**Reviewing the least confident 20% of events recovered all but one species and cut
missed records to 18**; reviewing half recovered all of them. A cross-fitted adapted
layer cut missed records to 32, and to 25 with 10% of events reviewed.

![Test-camera detections that the adapted layer identified correctly](https://raw.githubusercontent.com/ivanposavecwildlife/CamTrapTools/main/docs/adapted-examples.jpg)

## Roadmap

```
SD card ─► ingest/check ─► detect/events ─► review ─► ecology ─► GBIF
                                ▲              │
                                └─── adapt ◄───┘
```

| Module | Commands | Purpose | Status |
|---|---|---|---|
| Ingest & QA | `ingest`, `check`, `report` | Build the data package, find camera problems, report them | ✅ v0.1.0 |
| Detection | `detect`, `events`, `import`, `export` | SpeciesNet detection and species identification, independent events, importing other tools' results, validated Camtrap DP export | ✅ v0.2.0 |
| Review | `review`, `review-stats` | Local web app for checking AI labels, least confident first; review statistics | ✅ v0.3.0 |
| Evaluation | `evaluate`, `crops`, `adapt` | Per-species accuracy and confidence thresholds; training data and regional fine-tuning | ✅ v0.4.0 |
| Ecology | `ecology`, `compare` | Occupancy and activity analysis, and how sensitive results are to AI errors | ✅ v0.5.0 |

## Development

```bash
git clone https://github.com/ivanposavecwildlife/CamTrapTools
cd CamTrapTools
uv sync --extra detect
uv run ruff check src tests
uv run pytest
```

CI runs the linter and tests on Python 3.11–3.14 (Linux) and on Windows, without the
`detect` extra: tests use stand-ins for SpeciesNet, so CI never downloads the models.
The full suite, including the tests that need PyTorch, runs on macOS before each
release.

## License

If you use CamTrapTools in your work, please cite it using the DOI
[10.5281/zenodo.23500635](https://doi.org/10.5281/zenodo.23500635) or GitHub's "Cite this
repository" button.

[MIT](https://github.com/ivanposavecwildlife/CamTrapTools/blob/main/LICENSE). The optional `detect` extra installs
[SpeciesNet](https://github.com/google/cameratrapai) (Apache-2.0), whose MegaDetector
inference code is GPL-licensed. Snapshot Kgalagadi data used for validation is published by
[Snapshot Safari](https://lila.science/datasets/snapshot-kgalagadi/) under the
Community Data License Agreement (permissive); apart from the example images in `docs/`,
it is not included in this repository.
