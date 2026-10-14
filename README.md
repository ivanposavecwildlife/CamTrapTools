# 📷🛠️ CamTrapTools

**Open-source toolkit for camera-trap projects: from SD card to ecological answers.**

![status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)
![python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

## Why

Camera traps are one of the main ways conservationists monitor wildlife, but a
single project can produce millions of images. Field teams routinely:

- lose months of data to broken, moved or misconfigured cameras that nobody noticed
- spend most of their review time on empty images
- have no efficient way to check AI predictions
- can't tell whether AI models actually work on *their* cameras and species
- struggle to turn detections into ecological results and share the data openly

CamTrapTools covers the whole workflow in one place. Every module reads and writes a
[Camtrap DP](https://camtrap-dp.tdwg.org) package, the open standard for camera-trap
data, so results can be published directly to [GBIF](https://www.gbif.org) and used
with other tools.

## Workflow

```
SD card ─► ingest/check ─► detect/events ─► review ─► ecology ─► GBIF
                                ▲              │
                                └─── adapt ◄───┘
```

| Module | Commands | Purpose | Status |
|---|---|---|---|
| Ingest & QA | `ingest`, `check` | Build the data package; detect clock errors, outages and false-trigger problems | In development |
| Detection | `detect`, `events`, `import` | MegaDetector and SpeciesNet inference, grouping photos into independent events, importing existing outputs | Planned |
| Review | `review` | Local web app for checking AI labels, least confident first | Planned |
| Evaluation | `evaluate`, `adapt` | Per-species accuracy and confidence thresholds; regional fine-tuning | Planned |
| Ecology | `ecology` | Occupancy and activity analysis, and how sensitive results are to AI errors | Planned |

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ivanposavecwildlife/CamTrapTools
cd CamTrapTools
uv sync
uv run camtraptools --help
```

## Usage

```bash
uv run camtraptools ingest path/to/images --out my_project
uv run camtraptools check my_project
```

## Development

```bash
uv run pytest
```

## License

[MIT](LICENSE)
