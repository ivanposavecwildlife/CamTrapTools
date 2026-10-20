# Validation: Orinoquía, Colombia

A second test of the whole workflow, on a survey unlike the Kgalagadi in almost every
way: Neotropical savanna and gallery forest instead of desert, different cameras, a far
richer fauna, and labels from experts instead of volunteers.

## Data

[Orinoquía Camera Traps](https://lila.science/datasets/orinoquia-camera-traps/)
(University of Minnesota, published on LILA BC under the Community Data License
Agreement, permissive): 50 cameras 1 km apart in two private reserves in Meta,
Colombia, January to July 2020, labelled by experts in Wildlife Insights.

This test used every image from **11 to 24 March 2020**: **6,416 images from 48
cameras**, 587 camera-days. Images of people are not published and were left out.
`examples/orinoquia/prepare.py` downloads the same subset.

SpeciesNet's training data may include this dataset, so its accuracy here may be
higher than on a survey it has never seen.

## What the test found and fixed

Running a real survey through the toolkit showed five problems, now fixed:

- **Labels in COCO Camera Traps format.** The expert labels come as a COCO Camera
  Traps JSON file, the format of LILA BC datasets and MegaDetector tools, which
  `labels` could not read. It now can.
- **Names that did not match.** Labels such as `salvins_curassow`, `unknown_cervid`,
  `cattle` and `red_brocket_deer` did not match SpeciesNet's names (*Salvin's
  curassow*, *Cervidae*, *domestic cattle*, *red brocket*). `labels` now ignores
  punctuation, tries families for "unknown X", and accepts a unique match at the end or
  start of a name. Seven truly ambiguous labels (for example "unknown howler monkey")
  are mapped in `examples/orinoquia/names.csv`.
- **The AI more specific than the experts.** Experts often labelled an animal only to
  family ("armadillo family", "opossum family", "bird"), and SpeciesNet named the
  species (nine-banded armadillo, common opossum, grey-fronted dove). `evaluate`
  counted this as wrong. It is now its own outcome, **more specific**: consistent with
  the reference, which can neither confirm nor contradict the species.
- **A misleading resume count.** After the computer stopped mid-run, `detect` reported
  every image as done, because SpeciesNet's saved progress also lists images it has not
  reached. It now counts only finished images.
- **Downloads that ran out of network ports.** The download scripts opened a new
  connection per image; after about 6,300 images the computer ran out of ports. They
  now reuse connections, retry, and fetch failures one at a time at the end.

## Results

`ingest` read every image (EXIF times matched the published times exactly), and `check`
flagged four cameras with days of more than 100 images. `labels` matched 6,349 of the
6,416 images to an expert label.

### Identification

`evaluate` on the 761 events with an expert label:

| Outcome | Events |
|---|---|
| Correct | 385 (51%) |
| Correct only above species level | 104 (14%) |
| More specific than the experts | 162 (21%) |
| Wrong | 110 (14%) |

For comparison, on the Kgalagadi sample 80% were correct and 3% wrong. The Neotropical
fauna is harder: many similar birds, rodents and armadillos, and dense vegetation. The
most common real mistakes were close relatives and empty images:

- Salvin's curassow called razor-billed or wattled curassow (12 events), and collared
  peccary called white-lipped peccary (6);
- empty images with a false detection, recorded as "animal" or "unknown" (19);
- animals the AI only recognised as "animal", such as rodents, birds and a tapir.

Per label, the common mammals were identified with high precision: collared peccary
0.97, paca 1.00, black agouti 0.94, coati 0.96, tayra 0.92, ocelot 1.00.

### Ecology

`compare` between the AI's labels and the experts':

- **Common mammals were barely affected.** Black agouti was recorded at 26 sites (29
  by the experts), collared peccary at 22 (24), paca at 17 (18), coati at 11 (12) and
  tapir at 6 (7). Modelled occupancy differed by 0.02 to 0.09: collared peccary 0.59
  against 0.68, agouti 0.60 against 0.65, paca 0.43 against 0.45; giant anteater and
  tapir were identical.
- **Some species were missed.** Salvin's curassow (6 sites) was always called another
  curassow, blue-crowned motmot (7 sites) was never named, and southern tamandua was
  recorded at 6 of 13 sites.
- **"Added" species were mostly more specific names.** Nine-banded armadillo (25 sites)
  and common opossum (13) appear only at family level in the expert labels, so
  `compare` counts them as added.
- **Species richness:** 31 species in 587 camera-days, with a Chao2 estimate of 44.5
  (33.9–93.8): two weeks found about 70% of the species likely present.

## Takeaways

- The toolkit ran on a new country, new cameras and a new label format with the fixes
  above, and the ecological results for common mammals matched the experts' closely.
- In species-rich tropical sites, review should focus on birds, small rodents and
  curassows, where SpeciesNet confused close relatives; the least-confident-first
  order of `review` puts these early.
- Comparing the AI with reference labels needs care when the reference is coarser than
  the AI; the new "more specific" outcome makes that visible instead of counting it as
  an error.

## Reproduce

```bash
uv run python examples/orinoquia/prepare.py orinoquia
uv run camtraptools ingest orinoquia/sd_cards --out orinoquia/package
uv run camtraptools detect orinoquia/package --images orinoquia/sd_cards --country COL
uv run camtraptools events orinoquia/package
uv run camtraptools labels orinoquia/package orinoquia/metadata/orinoquia_camera_traps.json \
    --names examples/orinoquia/names.csv --out orinoquia/truth.csv
uv run camtraptools evaluate orinoquia/package --truth orinoquia/truth.csv
uv run camtraptools compare orinoquia/package --truth orinoquia/truth.csv --out orinoquia/comparison
uv run camtraptools summary orinoquia/package --out orinoquia/summary.html
```
