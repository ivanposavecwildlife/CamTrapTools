# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-19

AI detection and species identification, independent events, importing other tools'
results and validated Camtrap DP export.

### Added

- `detect` command: runs SpeciesNet on every ingested image. SpeciesNet uses
  MegaDetector v5a to find animals, people and vehicles, classifies each animal and
  applies geofencing (`--country`, `--admin1-region`) and taxonomic rollup. Results are
  saved as `speciesnet.json`, and the command prints the images per category and per
  species for each deployment. Requires the optional `detect` extra; SpeciesNet is
  Apache-2.0 and its MegaDetector inference code is GPL-licensed.
- `detect` writes `observations.csv` with Camtrap DP media-level observations: one per
  detection at or above `--threshold` (default 0.2) with bounding box, `scientificName`
  (species, or a higher taxon such as *Bovidae* when SpeciesNet rolls up), classifier
  probability and model version; `blank` for empty images, `unknown` when SpeciesNet
  cannot tell, and `unclassified` for images it could not process.
- `detect` reuses existing results for the same model and country, so trying another
  `--threshold` is instant; `--rerun` runs SpeciesNet again, and `--images` is only
  needed when it runs.
- `events` command: groups consecutive images at each deployment into independent
  events (a new event after more than `--gap` minutes without images, default 30),
  fills in `eventID` on media-level observations and adds one event-level observation
  per event. The event's species is chosen by summed classifier probability across its
  images, its count is the largest number of individuals in any single image, and the
  command prints independent animal events per taxon and deployment. Rerunning
  replaces earlier event-level observations.
- `import` command: reads results in the MegaDetector batch format (MegaDetector,
  AddaxAI, Timelapse, run_md_and_speciesnet) or SpeciesNet's format, matches their
  image paths to the package even when they are relative to another folder, and writes
  them as `speciesnet.json` and `observations.csv`, so `events` and later steps work
  unchanged. Classification taxonomy is kept when it is in SpeciesNet's format; the
  command reports how many images matched, were ignored or have no results, and
  images without results are recorded as `unclassified`.
- `export` command: writes a complete Camtrap DP 1.0.2 data package (`datapackage.json`,
  `deployments.csv`, `media.csv`, `observations.csv`) to a separate folder, with every
  standard column, time zones on all timestamps (`--timezone`, e.g.
  `Africa/Johannesburg`, including daylight saving), spatial, temporal and taxonomic
  coverage, contributors, project details (`--sampling-design`) and data and media
  licenses. Images without a timestamp and non-standard deployment columns are left out
  and reported. Exported packages pass Frictionless validation against the Camtrap DP
  profile, which is checked in the test suite.
- `detect` records when SpeciesNet ran with the computer's UTC offset.

## [0.1.0] - 2026-10-15

First release: ingest, quality checks and an HTML health report.

### Added

- `ingest` command: scans one folder per camera deployment, reads capture times from
  EXIF and writes `media.csv` and `deployments.csv` in Camtrap DP format.
- Optional CSV or Excel field sheets with coordinates, camera details and recorded
  deployment periods, which take priority over image dates.
- Field-sheet validation: duplicate IDs, impossible coordinates, unreadable or reversed
  dates, images outside the recorded period, and deployment IDs that do not match image
  folders, with suggestions for near matches.
- `check` command: missing timestamps, reset or wrongly set clocks, time going
  backwards, long gaps (`--max-gap-days`), days with excessive triggers and cameras
  without night or day images, with key numbers for each deployment.
- `report` command: a self-contained HTML report with an interactive timeline, a
  summary table and all issues, in light and dark mode.
- Continuous integration on Python 3.11–3.14 (Linux) and on Windows and macOS.

[0.2.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.2.0
[0.1.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.1.0
