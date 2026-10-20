# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.7.1] - 2026-10-20

Tested on a second survey (Orinoquía, Colombia): COCO Camera Traps labels, smarter name
matching, a "more specific" outcome in `evaluate`, fixes for Intel Macs and resumed
runs, and macOS back in CI.

### Added

- `evaluate` has a fifth outcome, **more specific**: the AI names a taxon within the
  reference's label (a nine-banded armadillo where the reference says only
  Dasypodidae). It was counted as wrong; it now counts as right for precision, and the
  survey summary reports it.
- Second validation (`docs/validation-orinoquia.md`) on Orinoquía Camera Traps
  (Colombia), with `examples/orinoquia/prepare.py` to download the subset and
  `names.csv` for labels that need a mapping.
- `labels` reads COCO Camera Traps JSON files, the format of LILA BC datasets and
  MegaDetector tools (`--format coco`, detected automatically for `.json` files), with
  category names such as `collared_peccary` turned into words.

### Changed

- `labels` matches names more flexibly: punctuation is ignored (`salvins curassow`
  matches Salvin's curassow), `unknown X` is tried as X, `X family` and the family
  `X` + `ae` (`unknown cervid` becomes Cervidae), a label matches the one taxonomy name
  it ends (`cattle`: domestic cattle) or that starts it (`red brocket deer`: red
  brocket), and a species is preferred when a name is also a genus (`puma`: *Puma
  concolor*). A plain `unknown` label counts as not labelled.
- When `detect` continues an interrupted run, it counts only images SpeciesNet has
  finished; its saved progress also lists images not reached yet, which are processed
  again.
- CI tests macOS on Intel (`macos-15-intel`) for every push, and Apple Silicon
  (`macos-latest`) weekly, for every release and on demand: GitHub's Apple Silicon
  runners were often unavailable, so jobs were cancelled before they started. macOS
  jobs time out after 30 to 60 minutes instead of waiting indefinitely.

### Fixed

- The example download scripts reuse one connection per thread, retry network errors
  and fetch failed images one at a time at the end; opening a new connection per image
  ran out of network ports on large downloads.
- Activity and richness calculations work with NumPy 1.26, which Intel Macs install;
  they failed there because `numpy.trapezoid` needs NumPy 2.

## [0.7.0] - 2026-10-20

Deliverables for partners: a one-page survey summary, species richness, occupancy
with site covariates, privacy options for publishing, and videos.

### Added

- Videos: `ingest --videos` includes MP4, MOV and AVI files as media, with the
  recording time read from the file (the `mvhd` header of MP4 and MOV, wherever it is
  in the file, or the `IDIT` chunk of AVI) or else the file date, noted in
  `mediaComments`. `detect` samples frames from each video (`--frame-interval`,
  default one per second, `--max-frames`, default 10) into `frames/`, runs SpeciesNet on
  them and uses the frame with the most confident animal as the video's result,
  recorded with its path. The review app shows a video's best frame with the AI's boxes
  and a player for the clip.
- `summary` command: a self-contained HTML page of a survey's results to share with a
  project, with the survey's key numbers, camera problems and timeline, a species table
  (events, relative abundance index, sites, naive and modelled occupancy with
  warnings, busiest time of day), the daily activity chart and overlaps, species
  richness with the accumulation curve, how the labels were made (classifiers,
  reviewed events, accuracy from `evaluate`), and methods and limitations. It works
  offline, in light and dark mode and on narrow screens.
- Occupancy with site covariates in `ecology`: `--occupancy-covariate` and
  `--detection-covariate` name deployment columns (from the field sheet) that may
  affect occupancy or daily detection, through logistic regression with numeric
  covariates standardised and categorical ones coded per level. For each species
  recorded at three or more sites, models without covariates, with covariates on
  occupancy, on detection and on both are compared by AIC; the best model's effects
  with 95% Wald intervals and occupancy per level are printed, species at fewer than
  10 sites are flagged as exploratory, and every model is saved in
  `occupancy_covariates.csv`.
- Species richness in `ecology`: the species accumulation curve over camera-days by
  exact rarefaction (`richness.csv`, chart in `richness.svg`) and the Chao2 estimate of
  total species richness with a 95% log-normal interval and survey completeness, from
  species recorded on only one or two camera-days.
- Privacy in `export`: `--exclude-humans` leaves out images of people (with a human
  observation, or in an event decided as human) and their observations, and `export`
  warns when people are present without it; `--coordinate-decimals` rounds every
  deployment's coordinates, widens `coordinateUncertainty` to the rounding grid's
  half-diagonal and records `coordinatePrecision` in `datapackage.json`.

### Fixed

- `export` writes observation counts as whole numbers, as Camtrap DP requires, even
  when `observations.csv` has them as decimals (for example after editing in a
  spreadsheet).

### Changed

- Installation from PyPI (`uv tool install "camtraptools[detect]"` or `pip install
  "camtraptools[detect]"`) in the README and user guide, with PyPI and DOI badges and a
  citation note; `CITATION.cff` includes the Zenodo DOI.

## [0.6.0] - 2026-10-20

Ready for real surveys: large surveys and interrupted runs, other folder layouts,
camera clock correction, importing existing labels, the relative abundance index, a
user guide, and packaging for PyPI.

### Added

- `labels` command: imports existing image labels from a Camtrap DP package (human
  classifications first; event-level observations label every image in the event), a
  Wildlife Insights `images.csv` export (with its taxonomy), a Timelapse CSV export or
  any CSV (`--species-column`, `--path-column`, `--file-column`, `--folder-column`), and
  writes a truth file for `evaluate`, `crops` and `compare`. Images are matched by path,
  without leading folders, then by unique file name; common names are turned into
  scientific names with SpeciesNet's taxonomy when its model is downloaded, or with a
  `--names` mapping; blank labels are recognised, and unmatched, ambiguous and
  unrecognised labels are reported.
- Clock correction: optional `cameraTime` and `actualTime` field-sheet columns (what a
  camera's clock showed and the real time at that moment) make `ingest` shift all of
  that deployment's image times by the difference before deployment periods are
  derived, keep each image's original camera time in `mediaComments`, and report the
  correction. `check` suggests these columns when it finds a wrong clock.
- Relative abundance index in `ecology`: independent events per 100 camera-days for
  each species, in `occupancy.csv` and the printed table, and per site and species in
  `rai.csv`.
- `detect` saves SpeciesNet's progress every 10 minutes and continues an interrupted
  run when the same command is run again, discarding saved progress from a run with
  other settings; it shows SpeciesNet's progress bars.
- `ingest --folder-depth` reads deployments from deeper folder levels, such as
  `site/camera/` (deployment `site_camera`).
- `ingest` reports the files it skipped by type, such as videos, and JPEG images
  outside any deployment folder.
- User guide (`docs/user-guide.md`): installation, preparing images and field sheets,
  every step from `ingest` to `export` with example output and how to read it, choosing
  a workflow, the files in a package, troubleshooting and a command reference.

- Packaging for PyPI: project links, keywords and classifiers, a workflow that
  publishes to PyPI with trusted publishing when a GitHub release is published, and a
  `CITATION.cff` file for citing the toolkit.

### Changed

- The README's images and links are absolute, so they also work on PyPI.
- The message for a missing `detect` extra gives the `pip` command as well.
- `events` summarises all events at once: 100,000 images now take about a second
  instead of more than ten minutes, with identical results.
- Activity curves of species with many events are computed from one-minute bins with
  a fast Fourier transform, so `ecology` handles thousands of events per species in
  seconds; smaller samples are computed exactly as before.

### Fixed

- An event whose images all had a classifier probability of 0 got no confidence; its
  species is now chosen by the number of images.

## [0.5.0] - 2026-10-20

Ecology: detection histories, occupancy corrected for imperfect detection, daily
activity and overlap between species, and a comparison of how AI mistakes, review and
adaptation change these results.

### Added

- `ecology` command: writes a detection history per species (sites by sampling
  occasions of `--occasion-days`, default 7, with 1 for recorded, 0 for not recorded and
  empty when no camera was running), the camera days per site and occasion
  (`effort.csv`) and naive occupancy per species (`occupancy.csv`): events, sites
  recorded out of sites surveyed, and the share of surveyed occasions with a record at
  those sites. Sites are camera locations, reviewed decisions replace the AI's, and
  events identified only to a higher taxon are left out and listed unless `--all-taxa`
  or `--taxon` is given.
- Occupancy model in `ecology`: for each species recorded at three or more sites, the
  single-season occupancy model (MacKenzie et al. 2002) estimates the probability that
  a site is occupied, with a 95% profile-likelihood interval, from a daily detection
  probability applied to each occasion's camera days. `occupancy.csv` and the printed
  table add the chance of recording the species in a full occasion and of missing it
  over the whole survey at an occupied site, with a warning when detection is too rare
  to estimate occupancy or the interval is wider than 0.5.
- Activity patterns in `ecology`: for each species with at least
  `--min-activity-events` events (default 20), a daily activity curve from the start
  times of its independent events, as a von Mises kernel density with the bandwidth of
  Taylor (2008) estimated from trigonometric moments up to order 3, and its busiest
  time of day. The overlap coefficient between every pair of species (Δ₁, or Δ₄ when
  both have 75 or more events; Ridout & Linkie 2009) comes with a 95% percentile
  bootstrap interval from 999 resamples. Writes `activity.csv`, `overlap.csv` and an
  `activity.svg` chart of the six most recorded species.
- `compare` command: runs the ecology analyses on every labelling of a survey (the
  machine's labels alone, with reviews, ground truth from `--truth`, and other packages
  with `--other NAME=FOLDER`) and compares them with the reference (truth if given,
  otherwise the reviews): species and species-at-site records missed or added, mean
  naive and modelled occupancy error, the difference between each species' activity
  curves (1 − Δ) and the error in overlaps between species. Writes each source's
  ecology results, `scores.csv`, `comparison.csv` and `overlap_comparison.csv`.
- Ecological consequences in the case study, with `examples/kgalagadi/sensitivity.py`,
  which builds labellings with cross-fitted adapted layers and with simulated review of
  the least confident 5% to 50% of events, and compares them with the volunteer labels.
- `examples/kgalagadi/prepare.py` also writes a field sheet with each camera's season,
  from its first and last capture including blank ones.

## [0.4.0] - 2026-10-19

Evaluation and regional adaptation: measure the AI against ground truth, build crop
datasets split by camera, adapt SpeciesNet to a region and use the adapted layer in
`detect`, with a case study on Snapshot Kgalagadi.

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
  skipped, the padding is recorded, and the command warns about labels with no or few
  training crops.
- `adapt` command: trains a new final layer for SpeciesNet on the crops from `crops`,
  for only the labels in the training split, on frozen SpeciesNet features. Labels
  SpeciesNet knows start from its own weights, the layer from the epoch with the lowest
  validation loss is kept (`--epochs`, `--learning-rate`, `--seed`, `--device`), and
  `--balanced` weights classes by their number of crops. The command saves the layer as `adapted.pt`, every crop's prediction
  in `predictions.csv` and the crops' SpeciesNet features in `features.npz`, which later
  runs reuse. It prints accuracy per split and per label for SpeciesNet as run by
  `detect`, SpeciesNet limited to the training labels and the adapted layer, and the
  held-out labels it cannot name. Requires the optional `detect` extra.
- `adapt` chooses a confidence threshold on the validation crops: the lowest at which
  the adapted layer is as precise as SpeciesNet's species-level answers. It is saved in
  `adapted.pt` with the crop padding.
- `detect --adapted MODEL_DIR` classifies every animal detection with an adapted layer,
  cropped like the training crops, and uses its answer at or above the layer's threshold
  (or `--adapted-threshold`), except where SpeciesNet named a species the layer was not
  trained on. Answers are saved in `adapted.csv`, observations record the adapted layer
  in `classifiedBy`, and the command prints how many of SpeciesNet's answers changed.
- `evaluate --deployment` compares only events from the given deployments, such as
  cameras held out from training.
- Case study (`docs/case-study-kgalagadi.md`): adapting SpeciesNet on Snapshot
  Kgalagadi Season 1 raised correct events on seven held-out cameras from 72 to 92 of
  104. `examples/kgalagadi/prepare.py` downloads the data and writes the truth file to
  reproduce it.

### Changed

- Per-label precision counts answers that are correct above species level (such as
  *Bovidae* for a gemsbok) as right; recall still requires the exact taxon.

### Fixed

- The review app screenshot in the README was missing from the repository.

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

[0.7.1]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.7.1
[0.7.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.7.0
[0.6.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.6.0
[0.5.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.5.0
[0.4.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.4.0
[0.3.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.3.0
[0.2.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.2.0
[0.1.0]: https://github.com/ivanposavecwildlife/CamTrapTools/releases/tag/v0.1.0
