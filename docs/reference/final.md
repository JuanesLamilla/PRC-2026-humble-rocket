# Blend and assembly (`final/`)

`final/` is the deterministic last layer: it reads the trained member outputs
(`work/<tag>/outer_NN.parquet`, written by the [member pipeline](pipeline.md))
and produces the submitted file. Given the member files, re-running it
reproduces `submissions/humble-rocket_final.parquet` byte for byte.

```bash
python final/blend.py       # fit the per-lane weights, blend OOF and test rows
python final/assemble.py    # corrections + label-free rules -> the submission
python final/verify.py      # template check, per-month stats, sha256
```

All three run from the repository root and resolve paths from the environment:

| Variable | Default | Used by |
|---|---|---|
| `WORK_DIR` | `work` | member outputs and blend intermediates |
| `RAW_DIR` | `data/raw` | challenge movement tables |
| `BLEND_DIR` | `WORK_DIR` | where `assemble.py` finds the blend files |
| `RESULTS_DIR` | `results` | the blend report |
| `SUBMISSIONS_DIR` | `submissions` | the output parquet |

## `blend.py`: per-lane month-held simplex blend

Members are combined per **lane**, four row populations with very different
error structure:

| Lane | Rows |
|---|---|
| `nm_lt1h` | NM rows with C < 1 h (the bulk) |
| `nm_ge1h` | NM rows with C ≥ 1 h (long delays, heavy tail) |
| `nonm_lirf` | LIRF rows without an NM plan (smallest cell, largest per-row error) |
| `nonm_other` | no-NM rows elsewhere |

Per lane, the weights solve non-negative least squares on the simplex
(weights ≥ 0, summing to 1, lightly shrunk toward equal weights). They are
fitted on the members' month-held 2025 out-of-fold predictions: the weights
scored on month k are fitted on the other eleven months, so every 2025 month
gets an honest read, and the rows of the never-evaluated submission months are
leave-that-month-out estimates. Nothing in this layer sees 2026 labels.

Two member generations cover the submission rows (see
[the final submission](../06-final-model.md)): the primary members, whose
fold 0 predicts the January/July rows, and the fold-0 retrains for the months
the primary members never predicted. `blend.py` derives the retrain months
from member coverage, fits one weight set per lane for each generation, and
blends each submission row with the generation that covers its month. The
class and anomaly models are excluded from the retrain-month member lists for
the stacking-consistency reason the final-model page documents.

Outputs: `work/blend_oof.parquet`, `work/blend_test.parquet`,
`work/blend_weights.json` and `results/blend_report.txt` (lane RMSEs, the
fitted weights, the January+July read and the leave-February / leave-June
reads).

## `assemble.py`: corrections, rules, writer

Steps, in order:

1. read the blended submission rows;
2. **feeder-lateness correction**: when the stand's feeding arrival went
   in-block more than 90 min behind schedule, the blend overshoots slightly.
   Offsets per C band are fitted on the 2025 out-of-fold blend with a heavy
   smoothing prior and validated month-held (negative in 8 of 12 months,
   interval below zero). Arrival block times are published for the submission
   months, so the trigger is computable on every 2026 row;
3. **LFPG EOBT-drift adjustment**: in June and July 2026 the filed EOBT at
   LFPG sits about four minutes later relative to LOBT than in any 2025
   month, a filing-practice change no 2025-trained model has seen.
   EOBT-anchored features mis-read exactly those rows; the EOBT-free
   transformer is immune by construction, so the mean gap between it and the
   base transformer over the affected cell estimates the bias, label-free.
   The gap is taken from the transformer pair of the generation that feeds
   each month's blend, and a diagnostic prints the same gap for every
   airport-month;
4. **label-free rules**, proven on 2025: the window clip (on NM rows the
   label provably lies within ±3,620 s of MVT − LOBT), the rollover identity
   (when MVT − LOBT exceeds 20 h, LOBT kept the previous day's date and the
   label equals MVT − LOBT itself), and the twin cap (a no-NM departure whose
   matched arrival "twin" carries an NM plan must have fallen outside the
   join window, so y ≤ MVT − twin LOBT − 3606, applied when the bound is at
   least 300 s);
5. floor at zero, round to Int32, write
   `submissions/humble-rocket_final.parquet` and validate it against the
   organizers' template.

## `verify.py`: submission checks

Validates the written file against the organizers' template: row count, the
two columns (`MVT_ID_mvt`, `TAXITIME_SEC_mvt`), Int32 type, no nulls, no
negatives, the exact id set. It then prints per-month count, mean, median and
p90, and the file's sha256.
