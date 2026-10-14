# 2. Method

## 2.1 Overview

```text
challenge data + external archives
        │
        ▼
feature blocks (content-addressed cache)                      §2.2
        │
        ▼
members: CatBoost, RealMLP, GRU, set-attention, TabM          §2.3
  each with a recording-pattern head:  Σ p_k·C_k + (1 − Σ p_k)·ŷ
        │
        ├── routing: no-ADS-B expert on rows without a ground track   §2.4
        ▼
lane head on no-flight-plan rows                              §2.5
        ▼
stamp mixture (six recording classes, all departures)         §2.6
        ▼
cell stacker: non-negative member weights per cell            §2.7
        ▼
gate on C = MVT − SCHED: final stack below 1 h, earlier stack above   §2.8
        ▼
submission file
```

Every stage after the members is a post-processor. It reads a finished run's out-of-fold and test
predictions and writes a new run in the same format, so every stage is scored with the same
paired comparison ([`docs/03-validation.md`](03-validation.md)).

## 2.2 Features

A feature block is a function of the full movement table (2025 and 2026, departures and arrivals)
that returns one row per movement. No block reads the label or the off-block time of the row it
describes. Labels enter only through fold-scoped target encodings. Blocks are cached under a hash
of their source code, their input and any external archive they read.

Blocks used by every final member:

| Block | Content | Source |
|---|---|---|
| `calendar_v2`, `cats` | time of day, weekday, month; airport, runway, stand, operator, aircraft type as categoricals | challenge |
| `fpl_deltas`, `fpl_updates` | take-off minus each flight-plan time (EOBT, AOBT, …) and their pairwise differences | challenge |
| `regime` | real NM off-block, derived constant, or no NM record; which derived constant | challenge |
| `throughput`, `sched_demand`, `runway_gaps_v2` | rolling take-off and landing counts, scheduled demand, gaps between runway movements | challenge |
| `surface_queue`, `discharge_v2`, `arrival_occupancy_v2` | aircraft on the surface at pushback, take-offs and landings during the taxi window, queue position, measured taxi of neighbouring departures | challenge |
| `arrival_taxi_in` | rolling taxi-in statistics of recent arrivals | challenge |
| `stand_bounds_v2` | time since the stand's previous departure and arrival: physical bounds on the off-block time | challenge |
| `source_mismatch`, `identity_freq`, `feed_health` | disagreements between the airport and NM records, carrier and callsign identity, window share of rows without a flight plan | challenge |
| `metar_visibility`, `winter_ops` | visibility, ceiling, low-visibility procedures, frozen precipitation, de-icing likelihood | IEM METAR |
| `daylight_curfew` | solar elevation, local hour, minutes to and from night curfew | computed |
| `adsb_ground` | from the open ADS-B track: time from first ground movement to take-off, path length, stop time, final hold, ground speed, coverage flags | ADSB.lol |
| `adsb_ground_v2` | where the track starts relative to the stand, stand-to-threshold distance, pushback signature | ADSB.lol |

Added by some members: `taxi_distance` (taxiway-graph distance from stand to threshold;
OpenStreetMap), `stand_xplane` (stand size class, type and heading; X-Plane Scenery Gateway),
`context_sets` (the index used by the sequence and set models to gather nearby movements).

Five target encodings (airport × runway, × stand, × stand × runway, × operator, × aircraft type;
mean, p10, p50, count) are fitted inside each fold, on clean labels only, with shrinkage.

The ranking set is post-operational: every take-off time is known. Queue and window features
therefore use other movements' take-off times on both sides of the row. They never use another
departure's off-block time or label.

## 2.3 Members and the recording-pattern head

Every member predicts on a square-root target and carries the same head. For each recording
pattern `k` with a value `C_k` that is computable on a test row, a fold-scoped XGBoost classifier
estimates `p_k`. The prediction is the conditional mean:

```text
ŷ = Σ_k p_k · C_k + (1 − Σ_k p_k) · ŷ_normal
```

| Pattern `k` | Training label (from the target, training rows only) | Value `C_k` |
|---|---|---|
| schedule default | `|BLOCK − SCHED| ≤ 10 s` | `MVT − SCHED`, exactly the label |
| day rollover | `|y − 86,400| < 3,600 s` | 86,400 s + ŷ_normal (15 positives; falls back to the base rate) |
| early stamp | `y > 1 h` with an NM record, not one of the above | ŷ_normal + 1,495 s (fold-fitted mean offset) |

The normal regressor is trained on all rows with sample weight `1 − Σ p_k`, where `p` comes from
heads nested inside the training fold.

Final members:

| Member | Model | Notes |
|---|---|---|
| `exp0250` | CatBoost (depth 10, lr 0.02, leaf ≥ 100) | + `taxi_distance`, `stand_xplane`; tail-prone blocks masked on no-flight-plan rows |
| `exp0235` | CatBoost | + the out-of-fold prediction of a GRU member (`exp0223`), built with three inner groups per fold |
| `exp0225` | RealMLP | + `taxi_distance` |
| `exp0227` / `exp0268` | RealMLP | no masking; `exp0268` uses hyper-parameters from a TPE search (§3.6) |
| `exp0232` / `exp0271` | GRU over the airport's movement sequence around the departure | `exp0271` tuned |
| `exp0233` | set-attention over nearby movements | |
| `exp0234` | TabM (BatchEnsemble MLP) | |

The neural members are implemented in plain PyTorch in `src/prc/models/` (RealMLP after
Holzmüller et al. 2024, TabM after Gorishniy et al. 2025). Each member is trained with three
seeds; its test prediction comes from a fit on all twelve 2025 months.

## 2.4 Routing on ADS-B coverage

Roughly 40 % of departures have no ADS-B ground track: all of LTFM, most of LFPG, and LIRF in July
2026 ([`docs/01-problem-and-data.md`](01-problem-and-data.md) §1.4). A member trained with ADS-B features predicts these rows
from a missing-value pattern it has seen rarely at some airports.

For four members a twin is trained without the 21 ADS-B ground columns (`exp0259`, `exp0269`,
`exp0262`, `exp0263`). `scripts/route_no_ground.py` takes the twin's prediction on rows without a
ground track and the base member's elsewhere. The routing flag is label-free.

## 2.5 Lane head for departures without a flight plan

On no-flight-plan rows every `_flt` column is null and the member's head loses its strongest
inputs. `scripts/no_nm_lane_model.py` fits a two-branch estimator on these rows only:

```text
ŷ = p̂ · C + (1 − p̂) · ŷ_lane,        C = MVT − SCHED
```

with LightGBM for `p̂` and `ŷ_lane`, leave-one-month-out. The airports where it replaces the
member are chosen inside each fold from the other months only.

## 2.6 Stamp mixture

`scripts/mixture_v2.py` models the recorded label of every departure as a mixture of six classes.
The classes are training targets derived from the off-block stamp; they are never inputs.

| Class | Definition | Value |
|---|---|---|
| default | `|BLOCK − SCHED| ≤ 10 s` | `C` |
| far | off-block more than 1 h before schedule | `max(C, 0) + 3600·e^a` |
| near | `C > 30 min` and `10 s < |BLOCK − SCHED| ≤ 15 min` | `C − 900·tanh(a)` |
| early | NM record, label > 1 h | `(MVT − AOBT_3) + Δ_E(x)` |
| block ≪ flight plan | NM record, `AOBT_3 − BLOCK > 40 min`, NM taxi in [0, 1 h] | `(MVT − AOBT_3) + Δ_B(x)` |
| normal | the rest | `ŷ_member · e^a` |

A PyTorch network outputs the class probabilities and bounded per-row value heads. The loss is the
squared error of `Σ p_k v_k` on the raw scale plus a class log-loss. Classes that cannot occur on a
row, by test-observable facts, are masked out of the softmax. Folds are nested leave-one-month-out;
the test side is the mean of the thirteen fold models (`--test-foldavg`), averaged over two seeds.
The result is blended with the member's prediction at a fold-selected weight.

## 2.7 Cell stacker

`scripts/stack_l2.py --family cell` combines the seven members with non-negative weights that sum
to one, per cell of airport × flight-plan record present × C band (< 30 min, 30-60 min, 1-2 h,
> 2 h). Cell weights are shrunk toward their parent (flight-plan record × band, then global); the
shrinkage strength is chosen by inner leave-one-month-out. Rollover rows are excluded from every
fit. The test-side combiner is fitted on all twelve months.

Global weights of the final stack:

| Member (after lane head and mixture) | Weight |
|---|---:|
| routed GRU (`exp0271` / `exp0262`) | 0.445 |
| routed RealMLP (`exp0268` / `exp0269`) | 0.309 |
| routed set-attention (`exp0233` / `exp0263`) | 0.148 |
| routed CatBoost (`exp0250` / `exp0259`) | 0.098 |
| CatBoost + GRU stack (`exp0235`) | 0.000 |
| RealMLP + taxi distance (`exp0225`) | 0.000 |
| TabM (`exp0234`) | 0.000 |

Members with zero global weight still carry weight in some cells.

## 2.8 Gate on C

`scripts/gate_runs.py` takes the final stack on rows with `C < 1 h` (87.5 % of test rows) and an
earlier stack on the rest. The earlier stack has the same structure with the unrouted members
(`exp0250`, `exp0235`, `exp0225`, `exp0227`, `exp0232`, `exp0233`, `exp0234`) and one mixture fit
each.

The reason: on 2025 validation, rows with `C ≥ 1 h` (rollovers excluded) score 398.0 with the
earlier stack and 400.3-400.8 with the later ones. In 2026 those rows include a population with
no 2025 counterpart: LIRF in July 2026, delayed, without ADS-B ground coverage. Changes that moved
these rows could not be validated, so the final model leaves them on the earlier stack.

