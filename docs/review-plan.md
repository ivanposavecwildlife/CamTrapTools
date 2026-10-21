# Planning the review

Reviewing every AI decision is slow, and reviewing a fixed share of them (say the least
confident 20%) cannot say whether that was enough. `plan` answers two questions for
each species: **how much of its result is still in doubt because of the AI's
mistakes**, and **which events to review, and how many, until it is not**.

```bash
camtraptools plan my_survey
camtraptools review my_survey --images sd_cards --order plan
camtraptools ecology my_survey --out ecology --ai-errors
```

## How it works

1. **Learn how often the AI is wrong.** Reviewed events show how often the AI's label
   held at each confidence level. A logistic calibration with an offset per label is
   fitted to them, starting from the AI's confidence taken at face value, so a handful
   of reviews moves it only a little. It is conditional on the label and the
   confidence, so it stays fair when reviews were chosen by them (least confident
   first, or by the plan).
2. **Learn what the animal really was.** When the AI was wrong, the reviewers'
   corrections of the same label show what it usually was (which species its "animal"
   or "Bovidae" events turned out to be, or which species it confuses), shrunk towards
   the corrections of all labels.
3. **Put a probability on every record.** Each unreviewed event gets a chance of being
   each species; reviewed events are certain. For each species, every cell of its
   detection history (a camera site in a 7-day occasion) gets the chance that the
   species was truly recorded there. Where that disagrees with the results (a record
   that is probably false, or a probable record that is missing), the cell is in
   doubt.
4. **Plan.** Reviewing a cell's events settles it, so cells are planned in order of
   doubt removed per event reviewed, until a species' doubt is at most 5% of its
   records (`--tolerance`), or a quarter of a cell for rare species. Events with less
   than a 1 in 20 chance of correcting a record are not worth the time and are left
   out (`--min-gain`).

The plan is saved in the package as `review_plan.csv` (events in order, with the
species each matters for) and `review_plan_species.csv`. `review --order plan` makes a
fresh plan and shows those events first. Run `plan` again after a review session: what
you corrected changes what the AI is likely to have got wrong elsewhere.

`ecology --ai-errors` and `summary --ai-errors` add, for each species, the expected
number of sites with a false record and where it was missed, and flag species whose
results may change with more review. They need at least 20 reviewed events.

## Does it work?

Tested on the two surveys with reference labels for every event: Snapshot Kgalagadi
Season 1 (466 events, volunteer labels; see the [case study](case-study-kgalagadi.md))
and Orinoquía (765 events, expert labels; see the [validation](validation-orinoquia.md)).
The reviewer was simulated with the reference labels. Two strategies were compared at
the same review budgets: the least confident events first, and the plan (after the
least confident 2%, to calibrate), recomputed after every quarter of the budget. A
"wrong site" is a species recorded at a site where the reference has none, or missed
at a site where the reference has it.

**Snapshot Kgalagadi** (14 species with an occupancy estimate in the reference)

| Events reviewed | Wrong sites, least confident first | Wrong sites, planned | Expected by `ecology --ai-errors` (planned) | Occupancy intervals containing the reference (planned) |
|---|---|---|---|---|
| none | 44 | 44 | – | 6 of 14 |
| 5% | 44 | 30 | 58 | 9 of 14 |
| 10% | 36 | 23 | 45 | 10 of 14 |
| 20% | 21 | 11 | 37 | 12 of 14 |
| 30% | 13 | 7 | 18 | 14 of 14 |
| 72%, plan followed until every species was stable | | 0 | 3 | 14 of 14 |

**Orinoquía** (16 species with an occupancy estimate in the reference)

| Events reviewed | Wrong sites, least confident first | Wrong sites, planned | Expected by `ecology --ai-errors` (planned) | Occupancy intervals containing the reference (planned) |
|---|---|---|---|---|
| none | 119 | 119 | – | 12 of 16 |
| 5% | 110 | 104 | 117 | 13 of 16 |
| 10% | 99 | 96 | 85 | 13 of 16 |
| 20% | 86 | 64 | 62 | 14 of 16 |
| 30% | 73 | 41 | 33 | 15 of 16 |
| 59%, plan followed until every species was stable | | 9 | 8 | 16 of 16 |

- **The plan fixes more with the same effort.** For the same number of events, it
  left about half as many wrong sites once 20% or more were reviewed: on Kgalagadi 11
  instead of 21 at 20%, on Orinoquía 41 instead of 73 at 30%. Least confident first
  can spend reviews on events whose correction changes no record, such as one more
  visit of a species already confirmed at that site that week; the plan goes where a
  correction changes the results, including events labelled "blank" or "animal" at
  sites where a species was probably missed.
- **The toolkit knows roughly how much is still wrong.** On Orinoquía the expected
  number of wrong sites followed the real one closely at every budget. On Kgalagadi it
  was too cautious while few events were reviewed (37 expected, 11 found at 20%), and
  close once the plan was followed: 3 expected and 0 found, against 8 and 9 on
  Orinoquía.
- **Following the plan to the end** meant reviewing 72% of Kgalagadi events and 59% of
  Orinoquía events. Both surveys are hard for SpeciesNet (it missed several species in
  the Kgalagadi entirely, and confused close relatives in Colombia); on a survey where
  the AI is more often right, the plan asks for less. It always shows what is left in
  doubt, so a project can stop earlier knowing what that costs.

Occupancy for common species with reliable estimates was close to the reference
without any review (mean difference 0.06 on Kgalagadi, 0.02 on Orinoquía): AI mistakes
mostly affect rare species, and that is where the plan sends the reviewer.

**Tested and dropped.** An occupancy model that weights each record by its chance of
being true (a scaled-likelihood version of the occupancy model) was tested on both
surveys. Its estimates were no closer to the reference than those from the labels as
they are, and its intervals did not contain the reference more often, so it is not
part of the toolkit. Reviewing the right events did more for the results than
modelling the mistakes.

## Reproduce

```bash
uv run python examples/planned_review.py kgalagadi/package kgalagadi/truth.csv kgalagadi/planned
uv run python examples/planned_review.py orinoquia/package orinoquia/truth.csv orinoquia/planned
```

Each package comes from the steps in its case study (`detect` and `events`, without
reviews). The script writes `scores.csv` and per-species occupancy in `species.csv`.
