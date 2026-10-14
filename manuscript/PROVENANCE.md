# Sources of the numbers in `main.tex`

Every number in the manuscript comes from one of four kinds of source:

- **S** = recomputed for the paper by `scripts/make_figures_tables.py`. Outputs are in `scripts/out/`.
  Inputs: `data/raw/training_2025-*.parquet`, `data/raw/final_ranking.parquet` (2026 inputs only, no
  labels used) and `data/internal/fullstack_work/blend_oof.parquet`.
- **B** = `data/internal/fullstack_work/blend_report.txt`, written by `final/blend.py`. S reproduces
  the overall, lane and month RMSE values in B exactly.
- **F** = `data/internal/claude/FINAL_MODEL.md`, the internal run record, and `pipeline/RUNS.md`.
- **D** = `docs/01-problem-and-data.md`, `docs/02-method.md`, `docs/03-validation.md`, `docs/05-reproduction.md`.
- **C** = code: `pipeline/*.py`, `final/*.py`, `src/prc/**`.

Literature claims cite keys in `reference.bib`. Each key was verified, and its supporting text recorded, in
`data/internal/literature_review/` (see `evidence/*_notes.md` there).

Bootstrap intervals: airport-day blocks resampled within airport-month strata, 2,000 replicates, seed
20261012, partitions sorted. Re-running the script gives identical intervals.

| Claim (section) | Value | Source |
|---|---|---|
| Movements, departures, arrivals 2025 | 4,167,797; 2,085,047; 2,082,750 | S |
| Departures with / without flight plan | 2,062,577; 22,470 (1.08%) | S |
| Label median, mean, negatives, >1 h | 912 s, 991 s, 369 (min −12 s), 4,126 | S |
| Table 1 per-airport values | see `scripts/out/airport_stats.csv` | S |
| Max \|off-block − LOBT\| on matched rows | 3,606 s | S (also F §2) |
| Schedule-coincident share | 4.80% (100,048 rows) | S |
| LIRF coincidence vs ±60 s | 18.1% vs 2.5% | S (`schedule_coincidence.csv`) |
| Excess over neighbouring minutes | ≈32,300 rows (1.6%), ≈25,100 at LIRF | S |
| Coincidence infeasible (\|SCHED − LOBT\| > 3,616 s) | 3.2% of matched rows | S |
| Near-schedule class | 139,055 (6.7%), median label 1,497 s | S (`recording_classes.csv`); class definition F §2 |
| Long class; matched, C < 1 h, label > 1 h | 1,250; 115 (31 in the long class, stamp 16–115 min before schedule; 84 near-schedule) | S |
| Rollovers | 15 (13 LIRF) | S |
| Table 2 error decomposition | see `scripts/out/error_share.csv` | S |
| Overall RMSE, rollovers excluded / all | 191.09 s / 227.40 s | S = B |
| Overall 95% interval | 183.5–201.7 s | S |
| Lane RMSE and intervals | 149.37, 310.58, 588.53, 2,538.10 s | S = B; intervals S |
| Month RMSE and intervals; Feb/Jun leave-out | see `oof_results.json` | S = B; intervals S |
| Airport RMSE and intervals | see `oof_results.json` | S |
| 16.2-h LFPG record: share of SSE, RMSE without it, January share | 4.3%; 186.9 s (Jan 176.8 s, LFPG 213.0 s, unmatched outside LIRF 436.5 s); about 40% | S |
| LIRF unmatched labels > 20,000 s | 59 | S |
| Table 6 member RMSE and blend weights | — | B |
| Blend vs combiner alone | 191.09 vs 191.12; −0.02 s [−0.68, +0.65] (1,000 airport-day replicates, unstratified, `final/blend.py`) | B |
| Reduced build (75 columns, 9 members) | 206.80 s (+15.7); lanes +9.3, +21.3, +27, +580 | F §9 |
| Seed curves | combiner 1→17 seeds ≈ −1.6 s; EOBT-free transformer curve flattens after ≈10 seeds | F §9, §4.3 |
| Feeder offsets | −4.27, −6.12, −3.43 s; pseudo-count 2,000; 90 min trigger | F §7 |
| Feeder month-held effect (earlier blend) | −0.006 s [−0.010, −0.002], 8/12 months | F §7; `data/internal/claude/EXPLORATION.md` item 32 |
| Negative results | de-icing offsets 0/5 winter months; global delay-band offsets worse | EXPLORATION items 31, 28; F §9 |
| Rollovers with a flight plan | 1 of 15 (LIRF, y ≈ m) | S |
| Feb/Jun 2025 rows use the reduced member list | — | C `final/blend.py` (`RETRAIN_LANES`), F §6 |
| Twin cap | of 812 twin rows with bound ≥ 300 s, label respected it on 810; 10 min–6 h | F §2.5, `docs/06` |
| Rollover identity threshold | m > 72,000 s (20 h) | F §2.3 |
| Window clip margin | ±3,620 s | F §2.1, C `final/assemble.py` |
| Consistency defects | four-class member Feb/Jun means 539/505 vs 1,089/1,004 Jan/Jul; combiner −35 to −65 s | F §5 |
| A/A calibration | 37% / 35% (point rule), 7% / 5% (interval rule); 60 and 20 null pairs | D 03 §3.4 |
| Winner's-curse example | −3.76 s [−6.95, −1.00] → +2.32 s [−0.37, +5.20] | D 03 §3.5 |
| Seed spread vs bootstrap | factor 10 to 100 | D 03 §3.2 |
| ADS-B archive rebuild window | 2025-01-01 to 2025-04-20 | D 03 §3.7 |
| Stack member weights (Table 4) | 0.445, 0.309, 0.148, 0.098, 0, 0, 0 | D 02 §2.7 |
| Stack C ≥ 1 h gate rationale | 398.0 vs 400.3–400.8 s | D 02 §2.8 |
| EOBT−LOBT change, 2026 vs same 2025 month | LFPG Jun +240 s, Jul +288 s; other 38 cells −89 to +50 s | S (`eobt_drift.csv`) |
| LFPG drift correction | +17.2 s (Jul), +16.0 s (Jun); clamp ±60 s | F §7 |
| LIRF ADS-B coverage | 0.956 (Jul 2025) → 0.002 (Jul 2026); 0.80 (Jan 2026) | D 01 §1.4 |
| EHAM unmatched share | 1.6% (2025) → 5.0% (Jan 2026) | S (2026 inputs) and D 01 |
| Hyper-parameters (Appendix A) | — | F §4, C `pipeline/ftdist.py`, `class_mix.py`, `gbm_nested.py`, `final/blend.py` |
| HL-Gauss σ | 0.75 × bin width: 15 s (20 s bins), 90 s (120 s bins) | C `pipeline/ftdist.py` |
| Hardware | H200 (members), L40S (stack) | F/`pipeline/RUNS.md`, D 05 |
