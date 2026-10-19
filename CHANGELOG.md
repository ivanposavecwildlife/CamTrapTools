# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `detect` command: runs MegaDetector (MDv5a by default, `--model` to choose another)
  on every ingested image, saves the results as `megadetector.json` in the standard
  MegaDetector batch format, and summarizes images with animals, people, vehicles or
  nothing per deployment (`--threshold`, default 0.2). Requires the optional `detect`
  extra; MegaDetector's inference code is GPL-licensed.
- `detect` writes `observations.csv` with Camtrap DP media-level observations: one per
  detection at or above the threshold (with bounding box, confidence and the model
  version as `classifiedBy`), `blank` for images without detections and `unclassified`
  for images MegaDetector could not read.
- `detect` reuses existing results for the same model, so trying another `--threshold`
  is instant; `--rerun` runs MegaDetector again and `--images` is only needed when it runs.

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

[0.1.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.1.0
