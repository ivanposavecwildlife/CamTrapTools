# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `evaluate` command: compares the AI's event-level decisions with ground truth, either
  the package's reviews or a CSV of image labels (`--truth`, with `filePath` and
  `scientificName` and optional `genus`, `family`, `order` and `class` columns). Each
  event is correct, correct only above species level (verified against the true
  taxon's ancestry), unverified, or wrong. The command prints per-label precision and
  recall and every event that was not correct, and writes `evaluation.csv`.
- Confidence thresholds: `evaluate` prints, for each label the AI used, the lowest event
  confidence at which its answers reach a target precision (`--target-precision`,
  default 0.95) and how many answers that keeps. Labels with fewer than ten answers are
  flagged as unreliable, and `evaluation.csv` includes each event's confidence.
- Failure report: `evaluate --report report.html --images DIR` writes a self-contained
  HTML page with the outcomes, precision and recall, thresholds and every event the AI
  did not get exactly right, with its images and bounding boxes.
- `crops` command: saves a JPEG crop of every animal detection with a known true label
  (from `--truth` or the reviews) in `train/`, `val/` and `test/` folders with one
  subfolder per label, and lists them in `crops.csv`. Each camera location (its
  `locationID`, `locationName` or deployment) goes to a single split, so evaluation uses
  unseen backgrounds; target shares are set with `--val` and `--test`. Boxes are padded
  (`--padding`), crops under 32 pixels and detections whose true label is blank are
  skipped, and the command warns about labels with no or few training crops.
- `adapt` command: trains a new final layer for SpeciesNet on the crops from `crops`,
  for only the labels in the training split, on frozen SpeciesNet features. Labels
  SpeciesNet knows start from its own weights; classes are weighted by their number of
  crops, and the layer from the epoch with the lowest validation loss is kept
  (`--epochs`, `--learning-rate`, `--seed`, `--device`). The command saves the layer as
  `adapted.pt` with every crop's prediction in `predictions.csv`, and prints accuracy
  per split for SpeciesNet and the adapted layer and the held-out labels it cannot name.
  Requires the optional `detect` extra.

### Changed

- Per-label precision counts answers that are correct above species level (such as
  *Bovidae* for a gemsbok) as right; recall still requires the exact taxon.

## [0.3.0] - 2026-10-19

Human review: a local web app that shows the AI's least confident decisions first and
records corrections as human observations.

### Added

- `review` command: a local web app (on 127.0.0.1 only) that shows each event's images
  with the AI's bounding boxes, labels and confidences, the event-level decision with its
  vote share and common name, and previous/next navigation with the arrow keys. Only
  images listed in the package can be served.
- Review decisions: each event page has a form, pre-filled with the AI's decision, to
  accept or correct the type, scientific name and count. Decisions are stored in
  `reviews.csv` with the reviewer's name (`--reviewer`) and time, and written to
  `observations.csv` as event-level observations with `classificationMethod` `human`,
  next to the machine's. Saving moves to the next event. `events` re-applies reviews,
  `export` uses the human decision for reviewed events, and forms are protected by a
  per-session token.
- The review app shows events least confident first (`--order uncertainty`, the default,
  or `--order time`), scoring each event by its vote share and best classifier
  probability, with animals not identified to species capped at 0.5. It starts at the
  least confident unreviewed event and, after saving, moves to the next unreviewed one.
- Review progress on every page, an "all events" page with confidence and review status,
  and keyboard shortcuts: Enter or A to save, B for blank, 1–9 for the count, S for the
  species field and the arrow keys to move between events.
- `review` uses another port when the requested one is in use, and says so.
- `review-stats` command: reviewed events, agreement with the AI, each correction (AI →
  reviewer), changes per AI confidence band and review speed (pauses over ten minutes
  count as breaks).

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

[0.3.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.3.0
[0.2.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.2.0
[0.1.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.1.0
