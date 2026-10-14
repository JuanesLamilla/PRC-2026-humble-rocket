# Taxi-out time prediction: PRC Data Challenge 2026

> **Manuscript (draft, not yet submitted, not peer reviewed).** Lamilla Cuellar, J. E.,
> Urbiztondo, M., Parbhakar, A., and Ancheta, C. *Recording-Aware Ensembles for Taxi-Out
> Time Estimation at Ten European Airports.* Draft prepared for the Journal of Open Aviation
> Science: [`manuscript/main.pdf`](manuscript/main.pdf) (LaTeX source in [`manuscript/`](manuscript/)).
> The authors contributed equally. The draft may change before submission and has not been
> reviewed, so please do not cite it as a published result.

Team **humble-rocket**'s solution to the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/):
predict the taxi-out time of 670,790 departures at ten European airports in
January, February, June and July 2026, trained on all 2025 movements. The
metric is RMSE in seconds; February and June were never scored by the daily
leaderboard.

| | RMSE (s) |
|---|---:|
| Daily leaderboard (January/July half of this file) | 221.68 |
| Final ranking | `<FINAL>` |
| 2025 validation, month-held, ex-rollover (2.09 M departures) | 191.09 |
| 2025 validation, month-held, all rows | 227.40 |

Licensed GPL-3.0 (`LICENSE`).

## The model

A feature matrix built from the challenge movement tables, ADS-B ground tracks
(ADSB.lol) and the EUROCONTROL daily ATFM table feeds a set of members built
around an explicit model of the off-block *stamp*: two-stage GBMs that predict
the stamp's bounded offset to the flight plan and the probability that it is a
schedule-default artifact, distributional transformers whose output histogram
is masked to the provable label window, and a class-structured combiner that
merges every member's conditional components per recording class (with the
team stack of `docs/02-method.md` as one of its experts). A per-lane simplex
blend, fitted month-held on 2025, combines the members; two mechanism-based
corrections (feeder-arrival lateness, a 2026 EOBT-filing drift at Paris-CDG)
and a set of label-free rules (window clip, rollover identity, twin cap)
finish the file. Months are predicted by the member generation whose
training-time and test-time inputs are consistent; the write-up explains why
that distinction is worth several seconds of RMSE.

Full description and evidence: [`docs/06-final-model.md`](docs/06-final-model.md).

## Repository layout

| Path | Content |
|---|---|
| `src/prc/`, `experiments/`, `scripts/` | the team package: data preparation, feature blocks, the member stack whose prediction the class model consumes as an expert; documented in `docs/01`-`docs/05` |
| `pipeline/` | the final members: feature matrix (`fm_build.py`, `fm_v2.py`), two-stage GBMs (`gbm_nested.py`), distributional transformers (`ftdist.py`), the class-structured combiner (`class_mix.py`); `RUNS.md` records every training run |
| `final/` | `blend.py` (month-held per-lane simplex), `assemble.py` (corrections + label-free rules + submission writer), `verify.py` |
| `jobs/slurm/`, `pipeline/jobs/` | Slurm templates the runs used |
| `assets/` | the hand-compiled strike/disruption calendar (public news sources) |
| `docs/` | the write-up as an MkDocs site (`mkdocs.yml`): problem and data, method, validation protocol, reproduction, the final submission, code reference, data sources. Build it locally with `uv run --group docs mkdocs serve` |
| `manuscript/` | the manuscript (LaTeX source, figures, provenance of every quoted number) |
| `data/` | inputs; `data/README.md` says where every file comes from (only that file is committed) |

## Reproducing the submission

1. Download the data: [`data/README.md`](data/README.md) (challenge tables,
   ADS-B ground-track features, the EUROCONTROL ATFM workbook) and
   [`docs/data-sources.md`](docs/data-sources.md) (all external sources and
   licences).
2. Train the members per [`pipeline/RUNS.md`](pipeline/RUNS.md): matrix, then
   each member for folds 0-12 (GPU; the team-stack expert of the class model
   reproduces per [`docs/05-reproduction.md`](docs/05-reproduction.md)).
3. Blend, assemble, verify:

```bash
python final/blend.py
python final/assemble.py
python final/verify.py     # template check, per-month stats, sha256
```

The blend and assembly are deterministic: given the member files, they
reproduce the submitted parquet byte for byte. GPU member retraining
reproduces predictions only up to hardware nondeterminism; every training run
is recorded in `pipeline/RUNS.md` with its tag, flags and seeds.

## Validation

Every member writes month-held out-of-fold predictions for all twelve 2025
months from the same code that predicts 2026; nothing is ever scored in-fold.
Blend weights scored on a month are fitted on the other eleven, so the
February and June rows of the 2025 read are leave-that-month-out estimates,
the honest stand-in for the months no leaderboard ever scored. Changes were
accepted on day-block bootstrap intervals and months-improved counts, under
rules fixed in advance (`docs/03-validation.md`). We use no
leaderboard-derived values.

## External data

All external sources are open and declared; links, snapshots and licences are
in [`data/README.md`](data/README.md) and [`docs/data-sources.md`](docs/data-sources.md):

- ADSB.lol global history. *Contains data from ADSB.lol, made available under
  the Open Database License (ODbL) 1.0.* Runway thresholds from OurAirports
  (public domain).
- EUROCONTROL Performance Review Unit, daily airport arrival ATFM delay.
- METAR (Iowa Environmental Mesonet), Open-Meteo, OpenStreetMap aeroways and
  X-Plane Gateway stand data (used by the team feature blocks).
- A hand-compiled strike and disruption calendar from public news reports
  (`assets/strike_calendar.csv`).
