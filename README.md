# traptools

**An open-source toolkit for camera-trap projects: from SD card to ecological answers.**

> 🚧 Early development. Built in the open while I learn conservation technology.
> See the [roadmap](#roadmap).

## The problem

Camera traps are one of the main ways conservationists monitor wildlife, but a
single project can produce millions of images. Field teams struggle to:

- notice broken, moved or misconfigured cameras before months of data are lost
- sort through huge numbers of empty images
- check AI predictions efficiently
- know whether AI models actually work on *their* cameras and species
- turn detections into ecological results and share the data openly

`traptools` handles each of these steps, with every module reading and writing the open
[Camtrap DP](https://camtrap-dp.tdwg.org) standard, so its output can be published to
GBIF and used with other tools.

## Roadmap

| Module | Command | Status |
|---|---|---|
| 1 | `ingest`, `check`: build the data package and check camera health | 🛠 in progress |
| 2 | `detect`, `events`, `import`: AI labels with MegaDetector and SpeciesNet, grouped into animal visits | planned |
| 3 | `review`: human review, least confident first | planned |
| 4 | `evaluate`, `adapt`: measure the AI and fine-tune it for your region | planned |
| 5 | `ecology`: occupancy and activity, and how much AI errors change them | planned |

## Install (development)

```bash
git clone https://github.com/ivanposavecwildlife/traptools
cd traptools
uv sync
uv run traptools --help
```

## Usage

```bash
uv run traptools ingest path/to/images --out my_project
uv run traptools check my_project
```

## License

MIT
