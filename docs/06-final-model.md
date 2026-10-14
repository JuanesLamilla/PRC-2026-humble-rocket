# The final submission

This document describes the model behind `humble-rocket_final.parquet`:
predictions for the 670,790 departures of `final_ranking.parquet` (January,
February, June and July 2026). It sits on top of the stack documented in
`docs/01`-`docs/05`: the members here are trained on a compact matrix of
raw-derived features plus the ADS-B ground-track columns, and the class model
additionally consumes the team stack's prediction ([`docs/02-method.md`](02-method.md))
as one of its experts. `pipeline/RUNS.md` records every training run;
`final/` holds the blend and assembly code.

## Problem

Predict the taxi-out time (take-off minus off-block, seconds) of each 2026
departure at ten European airports, trained on twelve labelled months of 2025.
Scoring is RMSE. February and June were never on the daily leaderboard; the
arrivals of all four months are labelled, the departures are not.

## What the data taught us

Three properties of the data shape the design.

**The join window.** The organizers attached an NM flight plan to a departure
only when the airport's off-block stamp lay within 3,606 s of the NM system's
last off-block time (LOBT). We verified this on all 2,062,577 NM departures of
2025: max |BLOCK − LOBT| is exactly 3,606 s. Two consequences follow.

1. On every NM row the label provably lies in [MVT − LOBT − 3606, MVT − LOBT + 3606].
   Projecting a prediction onto that interval can only reduce its error, so the
   window clip is the last step of the pipeline, and the strongest members
   build the bound into their architecture.
2. "No flight plan" is partly a label-selected event: the join failed because
   the stamp was far from LOBT. The no-NM lane (about 1 % of rows, roughly a
   third of the squared error) has a different label distribution and gets its
   own specialists and blend cells.

**Schedule defaults.** A recording artifact sets the off-block stamp equal to
the scheduled time on a few percent of rows, which makes the label exactly
C = MVT − SCHED. Whether a row is a default is driven by operator recording
practice, which arrivals reveal without touching any departure label. Most
members model P(default) explicitly and predict a mixture.

**Lane structure.** Error concentrates in a few row populations: no-NM rows,
NM rows with long delays, and LIRF. Members are combined per lane rather than
globally.

## Members

Trained by the scripts in `pipeline/`, thirteen folds each: folds 1-12 hold
out one 2025 month (the member's honest out-of-fold prediction of it), fold 0
trains on all of 2025 and predicts the evaluation rows.

- **Two-stage GBM** (`gbm_nested.py`): stage 1 predicts properties of the
  off-block stamp (the bounded offset d = BLOCK − LOBT and its quantiles,
  P(default), P(long)) with inner month-held folds, so no stage-2 training row
  sees an in-fold stage-1 fit. Stage 2 fits the taxi time, including the
  mixture y_mix = p_def · C + (1 − p_def) · normal-model. Variants: a CatBoost
  mixture, a per-airport mixture, and a four-class stamp specialist (normal /
  default / near-default / long) that sees the other members' disagreement as
  features and combines class-specific value models by expectation.
- **Distributional FT-transformer** (`ftdist.py`): a feature-token transformer
  whose head is a softmax over taxi-time bins masked to the NM join window,
  plus a point mass at y = C masked where a default is infeasible. The
  prediction is the distribution mean. Variants: base (7 seeds), stage-1
  tokens, strike-calendar tokens, EOBT-free (18 seeds), no-NM rows upweighted.
- **Class-structured combiner** (`class_mix.py`), the strongest single member:
  y = p_def · C + p_near · (C − offset) + p_long · v_long + p_normal · Σ g_j n_j,
  where the class probabilities, the gate g over the members' conditional
  normal means n_j, and small corrections come from one MLP that sees every
  member's class signals (including the team stack's prediction). A second
  instance is trained on the no-NM rows only.

## Two member generations

The challenge released its evaluation data twice: `ranking.parquet` (the
January/July rows scored by the daily leaderboard) first, and
`final_ranking.parquet` (all four months; the same January/July flights with
the take-off stamp dithered by a few seconds) close to the deadline. That
timing leaves two member generations:

- the **primary** members above, whose fold 0 predicts the January/July rows.
  Each produced its thirteen folds in one nested run, so its out-of-fold
  behaviour is a faithful proxy for its fold-0 behaviour;
- **fold-0 retrains** on the final matrix for the months the primary members
  never predicted (February/June).

The retrains are one-off fold-0 fits, and that costs them something the 2025
validation cannot see: a stacked model's test-time inputs must come from the
same model instances that generated its training inputs. Two retrained members
break that symmetry on the final rows: the anomaly specialist's
member-deviation features exist only for the January/July rows (its February/
June means collapse by ~500 s against every sibling member), and the class
model consumed it as an expert (a 35-65 s depression against its siblings on
exactly those two months, within ±10 s elsewhere). Both are therefore excluded
from the February/June member lists; everything else about the two generations
is identical. January/July rows are predicted by the primary members (their
prediction maps to the re-released rows by adding the published take-off-stamp
shift, which moves the label by exactly that amount).

## Blend

Per lane (NM C < 1 h, NM C ≥ 1 h, no-NM LIRF, no-NM elsewhere): non-negative
least squares on the simplex, lightly shrunk toward equal weights, fitted on
the members' month-held 2025 out-of-fold predictions (`final/blend.py`).
Weights scored on month k are fitted on the other eleven, so the February and
June rows of the 2025 read are leave-that-month-out estimates, the honest
stand-in for the never-evaluated months. On the NM lanes the class combiner
carries most of the weight (0.98 / 0.82); the no-NM lanes are led by the
transformer variants, and LIRF no-NM by the no-NM class specialist (0.75).

## Corrections (`final/assemble.py`)

- **Feeder lateness.** When the stand's feeding arrival went in-block more
  than 90 min behind schedule, the blend overshoots slightly (the aircraft
  tends to leave quickly once ready). Offsets per C band are fitted on the
  2025 out-of-fold blend with a heavy smoothing prior and validated
  month-held (better in 8 of 12 months, interval below zero). Arrival block
  times are published for the submission months, so the trigger is computable
  on every 2026 row.
- **LFPG EOBT drift.** In June and July 2026 the filed EOBT at LFPG sits about
  four minutes later relative to LOBT than in any 2025 month; no other
  airport-month moves comparably. This is a filing-practice change no
  2025-trained model has seen, and EOBT-anchored features mis-read exactly
  those rows. The EOBT-free transformer is immune by construction, so the mean
  gap between it and the base transformer over the affected cell (LFPG, NM,
  C < 1 h) estimates the induced bias and is added back there: +17 s in July,
  +16 s in June. Label-free: the drift is visible in published feature
  columns, and the magnitude comes from comparing two of our own models.

## Rules (label-free, proven on 2025)

1. window clip (a projection, so it can only help);
2. rollover identity: when MVT − LOBT > 20 h, LOBT kept the previous day's
   date and the label equals MVT − LOBT itself;
3. twin cap: a no-NM departure whose arrival "twin" (same callsign and route,
   landing at another challenge airport within six hours) carries an NM plan
   must have fallen outside the join window, so y ≤ MVT − twin LOBT − 3606.
   On 2025, labels respect this bound on 810 of 812 twin rows with margin
   ≥ 300 s;
4. floor at zero, round, Int32.

## Validation and results

Everything is selected on 2025 with months held out; candidate changes were
accepted on day-block bootstrap intervals of the paired RMSE delta together
with counts of months improved, never on a single pooled number (the protocol
of [`docs/03-validation.md`](03-validation.md)). Rollover rows are excluded from member training
and reported separately, because three or four rows a year would otherwise
dominate any comparison.

2025 month-held reads of the delivered construction, ex-rollover
(`results/blend_report.txt`):

| read | RMSE (s) |
|---|---:|
| all twelve months | 191.09 |
| lane NM C < 1 h | 149.37 |
| lane NM C ≥ 1 h | 310.58 |
| lane no-NM other | 588.53 |
| lane no-NM LIRF | 2538.10 |
| February (leave-out, retrain member list) | 200.87 |
| June (leave-out, retrain member list) | 201.78 |

The daily leaderboard (January/July 2026 only) scored this file's
January/July half at **221.68**. The leaderboard was used as a deployment
sanity check of completed submissions; the pipeline incorporates no
leaderboard-derived values, and its one 2026-specific term (the LFPG
adjustment) is estimated label-free as described above.
