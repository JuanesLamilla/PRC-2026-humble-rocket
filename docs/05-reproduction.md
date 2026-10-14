# 5. Reproduction

This document rebuilds the final submission from the raw challenge data. Every command runs
from the repository root. GPU steps are given twice where it matters: as a SLURM job
(`jobs/slurm/`) and as a plain command for a machine with one GPU.

## 5.1 The chain

```text
challenge data ─ prepare ─┐
external archives ────────┼─ feature cache (ADS-B state H) ─ H members ─┐
                          │                                             │
                          ├─ routing flag (state H)                     │
                          └─ traces rebuild of 2025-01-01..04-20 (state T) ─ T members
                                                                        │
      routing ── lane heads ── mixture v2 ── two cell stacks ── gate ── candidate file
```

The final prediction is a gate of two stacks on C = MVT − SCHED (label-free, known for every
row):

| rows | stack | members (each: lane head, then mixture v2) |
|---|---|---|
| C < 1 h | `stack_l2_cell_ng4h2G` | `route_ng`, `exp0235`, `exp0225`, `route_ngRh2`, `route_ngGh`, `route_ngS`, `exp0234`; mixture averaged over two fold-averaged fits (`_mx2faA`) |
| C ≥ 1 h | `stack_l2_cell_mx2_xp` | `exp0250`, `exp0235`, `exp0225`, `exp0227`, `exp0232`, `exp0233`, `exp0234`; one mixture fit each (`_mx2`) |

A routed member `route_<tag>` takes a no-ADS-B expert's prediction on rows without an ADS-B
ground track and the base member's prediction elsewhere:

| routed member | base | no-ADS-B expert |
|---|---|---|
| `route_ng` | `exp0250_r7d_cat_sqrt_taxi_distance_xplane` (H) | `exp0259_r7d_cat_sqrt_taxi_distance_xplane_noadsb` (T) |
| `route_ngRh2` | `exp0268_r7c_realmlp_nomask_adsbtr_hp1` (T) | `exp0269_r7c_realmlp_nomask_noadsb_hp1` (T) |
| `route_ngGh` | `exp0271_r7c_seqenc_slog_nomask_adsbtr_hp1` (T) | `exp0262_r7c_seqenc_slog_nomask_noadsb` (T) |
| `route_ngS` | `exp0233_r7c_setenc_slog_nomask` (H) | `exp0263_r7c_setenc_slog_nomask_noadsb` (T) |

`exp0235_r7d_cat_stack_gru` is a CatBoost member with one extra input: the out-of-fold
prediction of the GRU member `exp0223_r7c_seqenc_slog`. It needs 39 inner GRU runs (§5.8).

**Two ADS-B archive states.** The ADSB.lol heatmap files of 2025-01-01 to 2025-04-20 drop
ground tracks at EGLL, EDDF and EHAM. Those 110 days are rebuilt from the full traces. Members
marked H were trained on the heatmap archive, members marked T on the rebuilt one. Only those
110 days differ; every other 2025 day and every 2026 day is identical in both states. An exact
rebuild therefore trains the H members, writes the routing flag, rebuilds the 110 days, and
then trains the T members.

## 5.2 Requirements

GPU steps ran on NVIDIA L40S cards with 48 GB of memory, one card per array task. The SLURM
templates request the CPU cores and RAM below; set the time limit for your cluster.

| step | resource |
|---|---|
| ADS-B heatmap build (§5.5) | CPU; internet access, about 550 GB streamed; 9 GB transient disk |
| member cells, H and T (§5.8, §5.11) | 1 CUDA GPU with 48 GB per seed; 8 CPU cores, 80 GB RAM |
| exp0223 inner runs (§5.8) | 1 CUDA GPU with 48 GB per task, 39 tasks; 16 CPU cores, 160 GB RAM |
| stack table (§5.8) | CPU; 4 cores, 16 GB RAM |
| traces rebuild (§5.10) | 11 CPU tasks; internet access; 12 GB RAM and up to 9 GB transient disk per task |
| mixture v2 fits (§5.13) | 1 CUDA GPU with 48 GB per fit, 21 fits; 8 CPU cores, 64 GB RAM |
| routing, lane heads, applies, stacks, gate, candidate | CPU; up to 32 GB RAM |

Disk: raw and interim challenge data 0.6 GB, feature cache 1.8 GB, external
archives 3.1 GB, about 0.1 GB per run directory.

## 5.3 Environment

```bash
git clone <repository URL> && cd prc-data-challenge-2026-humble-rocket
uv sync --group nn          # Python 3.12 and the locked packages, PyTorch included
```

[uv](https://docs.astral.sh/uv/) installs Python 3.12 if it is missing. `uv.lock` pins every
package (PyTorch 2.14, XGBoost 3.4.1, CatBoost 1.2.10). The GPU driver must support the CUDA
version of the locked PyTorch wheels. Every GPU job script checks for a visible GPU before it
starts.

On a SLURM cluster, enable the `##SBATCH --account`, `--partition` and `--time` lines of the
templates in `jobs/slurm/` for your site and create the log directory once: `mkdir -p jobs/logs`.

## 5.4 Challenge data

The organizers issue access keys to registered teams. Keys never go into a file in the
repository.

```bash
mc alias set prc https://s3.opensky-network.org <access-key> <secret-key>   # MinIO client, once
cp .env.example .env              # set PRC_TEAM_NAME; PRC_DATA_BUCKET is the organizers' bucket
uv run prc data pull              # mc mirror -> data/raw/
uv run prc data prepare           # -> data/interim/movements.parquet
```

`jobs/slurm/stage-data.sbatch` runs the same steps plus the feature build of §5.6.
`data/raw/manifest.json` records the sha256 of every challenge file, so a reissued file is
visible.

## 5.5 External archives (state H)

Sources, licences and attributions: [`docs/data-sources.md`](data-sources.md). Run these on a CPU node with
internet access.

```bash
uv run python scripts/fetch_external.py ourairports   # first: the ADS-B build reads runway thresholds
uv run python scripts/fetch_external.py metar --years 2025 2026
uv run python scripts/fetch_external.py osm           # -> data/external/osm_<date>/
uv run python scripts/fetch_external.py xplane        # -> data/external/xplane_gateway_<date>/
uv run python scripts/fetch_external.py openmeteo     # read by the mixture v2 frame
uv run python scripts/adsb_build.py run --stage 1 --workers 3   # 2025-01, 2025-07, 2026-01, 2026-07
uv run python scripts/adsb_build.py run --stage 2 --workers 3   # the other ten 2025 months
uv run python scripts/adsb_build.py status                      # expect 426 of 427 days (2025-12-31 is absent)
uv run python scripts/adsb_v2.py archive                        # ground points -> adsblol_ground_v2/ and the stand table
```

`adsb_build.py` reads the heatmap files and falls back to the full traces for a day without
them. It resumes where it stopped. Keep exactly one `data/external/osm_*` directory: the
`taxi_distance` block reads the lexically last one.

## 5.6 Feature cache (state H)

```bash
uv run prc features build --all
```

Run it after §5.5. The cache is content-addressed
(`data/features/<block>__<source hash>__<input hash>[__x<external hash>].parquet`), so the H
and T caches of the ADS-B blocks live side by side.

## 5.7 Running one member cell

A member is one experiment YAML in `experiments/`, run with three seeds. Each seed is one
cell: every fold of the CV scheme, plus (with `test_fit: full` in the YAML) one fit on all of
2025 that writes the test prediction.

With SLURM, one array task per seed:

```bash
cell() {  # cell <experiment> <scheme> <run root> [seeds]
  local seeds="${4:-1,2,3}"
  CONFIG="experiments/$1.yaml" SCHEME="$2" OUT_ROOT="outputs/runs/$3" SEEDS="$seeds" \
    sbatch --parsable --array=0-2 --export=ALL,CONFIG,SCHEME,OUT_ROOT,SEEDS \
    jobs/slurm/member-cell.sbatch
}
cell exp0225_r7c_realmlp_taxi month_loo loo-exp0225_r7c_realmlp_taxi
```

Without SLURM, on one GPU, the same cells one after another:

```bash
e=exp0225_r7c_realmlp_taxi
for s in 1 2 3; do
  uv run python scripts/run_cell.py --config "experiments/$e.yaml" --seed "$s" \
    --scheme month_loo --out "outputs/runs/loo-$e"
done
```

Each cell writes `outputs/runs/loo-<exp>/<exp>__s<seed>__month_loo/`. A cell whose
`metrics.json` exists is skipped, so a rerun resumes.

**Check every cell before merging.** A cell can finish with a NaN fold; that fold's rows are
then missing from its out-of-fold file.

```bash
for s in outputs/runs/loo-*/*/status.json; do
  uv run python -c "import json, math, sys; d = json.load(open(sys.argv[1])); \
bad = [i for i, x in enumerate(d['fit_rmse']) if x is None or math.isnan(x)]; \
print(sys.argv[1], 'NaN folds', bad) if bad else None" "$s"
done
```

Rerun a member with a NaN fold with the next unused seed (`SEEDS=4`), drop the failed cell's
directory from the run root, and merge.

**Merge the seeds** into one run directory per member:

```bash
uv run python scripts/merge_seeds.py merge-seeds outputs/runs/loo-exp0225_r7c_realmlp_taxi
# -> outputs/runs/exp0225_r7c_realmlp_taxi__s1-2-3__month_loo/
```

The merge joins the cells on the movement id, refuses cells of different configs or rows, and
recomputes every metric. The merged name carries the seeds (`__s2-3-4__` for seeds 2, 3, 4).

## 5.8 H members

Train on the heatmap archive (state H), before §5.10.

```bash
for e in exp0250_r7d_cat_sqrt_taxi_distance_xplane exp0225_r7c_realmlp_taxi \
         exp0227_r7c_realmlp_nomask exp0232_r7c_seqenc_slog_nomask \
         exp0233_r7c_setenc_slog_nomask exp0234_r7c_tabm_slog_nomask exp0223_r7c_seqenc_slog; do
  cell "$e" month_loo "loo-$e"
done
cell exp0223_r7c_seqenc_slog tail_nov_dec fwd-exp0223_r7c_seqenc_slog
```

Then the stacked member `exp0235`. Its stack table needs 39 inner runs of `exp0223` (3 seeds
× the 13 outer training-month sets of `month_loo` and `tail_nov_dec`) and the two `exp0223`
run roots above:

```bash
MEMBER=exp0223_r7c_seqenc_slog OUT_ROOT=outputs/runs/inner-exp0223_r7c_seqenc_slog \
  sbatch --array=0-38%8 --export=ALL,MEMBER,OUT_ROOT jobs/slurm/stack-inner.sbatch
sbatch jobs/slurm/stack-build.sbatch           # after the inner runs and both exp0223 roots
cell exp0235_r7d_cat_stack_gru month_loo loo-exp0235_r7d_cat_stack_gru
```

Without SLURM: `uv run python scripts/stack_inner.py inner --member exp0223_r7c_seqenc_slog
--out outputs/runs/inner-exp0223_r7c_seqenc_slog` runs all 39 inner runs in sequence;
`scripts/stack_inner.py build` (arguments in the script's docstring and in
`jobs/slurm/stack-build.sbatch`) writes the table to `data/stack/exp0223_r7c_seqenc_slog__g3/`.

Check the cells (§5.7), then merge the seven member roots:

```bash
for r in outputs/runs/loo-exp02{25,27,32,33,34,35,50}_*; do
  uv run python scripts/merge_seeds.py merge-seeds "$r"
done
```

## 5.9 Routing flag (state H)

Write the flag before the traces rebuild, while `data/external/` holds the heatmap archive:

```bash
uv run python scripts/write_route_flag.py exp0250_r7d_cat_sqrt_taxi_distance_xplane__s1-2-3__month_loo
# -> outputs/route/adsb_has_gnd_heatmap.parquet
```

Our flag: 2,429,888 rows; 1,444,508 with a ground track, 980,665 without, 4,715 null (null
is routed as "no ground track").

## 5.10 Traces rebuild (state T)

```bash
P=$(MODE=prep sbatch --parsable --export=ALL,MODE jobs/slurm/adsb-traces.sbatch)
Q=$(MODE=pull sbatch --parsable --dependency=afterok:$P --array=0-10%8 --export=ALL,MODE jobs/slurm/adsb-traces.sbatch)
MODE=v2 sbatch --dependency=afterok:$Q --export=ALL,MODE jobs/slurm/adsb-traces.sbatch
```

`prep` keeps a backup of the 110 heatmap days. `pull` rebuilds 10 days per task from the full
traces. `v2` checks that all 110 days were rebuilt, rebuilds their v2 archive and the
`adsb_ground` / `adsb_ground_v2` feature cache. Without SLURM, run the three modes in order with
`bash`, setting `SLURM_ARRAY_TASK_ID=0` to `10` for `pull`.

## 5.11 T members

Train after §5.10. Our `exp0269` seed-1 cell diverged, so that member uses seeds 2, 3 and 4.

```bash
for e in exp0259_r7d_cat_sqrt_taxi_distance_xplane_noadsb exp0262_r7c_seqenc_slog_nomask_noadsb \
         exp0263_r7c_setenc_slog_nomask_noadsb exp0268_r7c_realmlp_nomask_adsbtr_hp1 \
         exp0271_r7c_seqenc_slog_nomask_adsbtr_hp1; do
  cell "$e" month_loo "loo-$e"
done
cell exp0269_r7c_realmlp_nomask_noadsb_hp1 month_loo loo-exp0269_r7c_realmlp_nomask_noadsb_hp1 2,3,4
```

Check the cells (§5.7), then merge:

```bash
for r in outputs/runs/loo-exp02{59,62,63,68,69,71}_*; do
  uv run python scripts/merge_seeds.py merge-seeds "$r"
done
```

## 5.12 Routing and lane heads

CPU only.

```bash
F=outputs/route/adsb_has_gnd_heatmap.parquet
L=__s1-2-3__month_loo
uv run python scripts/route_no_ground.py exp0250_r7d_cat_sqrt_taxi_distance_xplane$L \
  exp0259_r7d_cat_sqrt_taxi_distance_xplane_noadsb$L route_ng__month_loo --flag $F
uv run python scripts/route_no_ground.py exp0268_r7c_realmlp_nomask_adsbtr_hp1$L \
  exp0269_r7c_realmlp_nomask_noadsb_hp1__s2-3-4__month_loo route_ngRh2__month_loo --flag $F
uv run python scripts/route_no_ground.py exp0271_r7c_seqenc_slog_nomask_adsbtr_hp1$L \
  exp0262_r7c_seqenc_slog_nomask_noadsb$L route_ngGh__month_loo --flag $F
uv run python scripts/route_no_ground.py exp0233_r7c_setenc_slog_nomask$L \
  exp0263_r7c_setenc_slog_nomask_noadsb$L route_ngS__month_loo --flag $F
```

The expert's seed columns fill the base's seed slots by position, so the `s2-3-4` expert routes
into an `s1-2-3` base.

Lane head on departures without a flight-plan record, 11 runs:

```bash
lane() {  # lane <source run> <name> -> outputs/runs/<name>_lane
  uv run python scripts/no_nm_lane_model.py fit --run "$1" --out "outputs/lane/$2.parquet"
  uv run python scripts/no_nm_lane_model.py apply --run "$1" --lane "outputs/lane/$2.parquet" \
    --out "$2_lane" --auto-airports
}
for e in exp0250_r7d_cat_sqrt_taxi_distance_xplane exp0235_r7d_cat_stack_gru exp0225_r7c_realmlp_taxi \
         exp0227_r7c_realmlp_nomask exp0232_r7c_seqenc_slog_nomask exp0233_r7c_setenc_slog_nomask \
         exp0234_r7c_tabm_slog_nomask; do
  lane "$e$L" "$e"
done
for t in ng ngRh2 ngGh ngS; do lane "route_${t}__month_loo" "route_$t"; done
```

`--auto-airports` chooses the airports where the lane head replaces the member, fold-safe.

## 5.13 Mixture v2

The member-independent frame once, then one GPU fit per lane run. The fit settings are fixed in
`jobs/slurm/mixture-v2.sbatch` (`--device cuda --batch 4096 --lr 2e-3 --epochs 40 --patience 5
--n-seeds 8`).

```bash
V15=exp0250_r7d_cat_sqrt_taxi_distance_xplane_lane,exp0235_r7d_cat_stack_gru_lane,exp0225_r7c_realmlp_taxi_lane,exp0227_r7c_realmlp_nomask_lane,exp0232_r7c_seqenc_slog_nomask_lane,exp0233_r7c_setenc_slog_nomask_lane,exp0234_r7c_tabm_slog_nomask_lane
FIN=route_ng_lane,exp0235_r7d_cat_stack_gru_lane,exp0225_r7c_realmlp_taxi_lane,route_ngRh2_lane,route_ngGh_lane,route_ngS_lane,exp0234_r7c_tabm_slog_nomask_lane

J=$(MODE=build sbatch --parsable --export=ALL,MODE jobs/slurm/mixture-v2.sbatch)
# C >= 1 h stack: one full fit per member, seed 0  -> outputs/mixture_v2/<run>__v2.parquet
RUNS=$V15 sbatch --dependency=afterok:$J --array=0-6 --export=ALL,RUNS jobs/slurm/mixture-v2.sbatch
# C < 1 h stack: two fold-averaged fits per member -> <run>__v2__fa0, <run>__v2__fa1
RUNS=$FIN TAG=fa0 MX_ARGS=--test-foldavg \
  sbatch --dependency=afterok:$J --array=0-6 --export=ALL,RUNS,TAG,MX_ARGS jobs/slurm/mixture-v2.sbatch
RUNS=$FIN TAG=fa1 MX_ARGS="--test-foldavg --seed 1" \
  sbatch --dependency=afterok:$J --array=0-6 --export=ALL,RUNS,TAG,MX_ARGS jobs/slurm/mixture-v2.sbatch
```

Without SLURM: `uv run python scripts/mixture_v2.py build --audit`, then for each run
`uv run python scripts/mixture_v2.py fit --run <run> --device cuda --batch 4096 --lr 2e-3
--epochs 40 --patience 5 --n-seeds 8` with `--tag fa0 --test-foldavg` or
`--tag fa1 --test-foldavg --seed 1` for the C < 1 h members.

Apply (CPU):

```bash
for m in ${V15//,/ }; do uv run python scripts/mixture_v2.py apply --run "$m" --out "${m}_mx2"; done
for m in ${FIN//,/ }; do
  for t in fa0 fa1; do uv run python scripts/mixture_v2.py apply --run "$m" --tag "$t" --out "${m}_mx2$t"; done
  uv run python scripts/avg_runs.py "${m}_mx2fa0" "${m}_mx2fa1" "${m}_mx2faA"
done
```

## 5.14 Stacks and gate

```bash
uv run python scripts/stack_l2.py $(for m in ${V15//,/ }; do echo "${m}_mx2"; done) \
  --family cell --out stack_l2_cell_mx2_xp
uv run python scripts/stack_l2.py $(for m in ${FIN//,/ }; do echo "${m}_mx2faA"; done) \
  --family cell --out stack_l2_cell_ng4h2G
uv run python scripts/gate_runs.py stack_l2_cell_mx2_xp stack_l2_cell_ng4h2G stack_l2_cell_ng4h2G_lt1h
```

`gate_runs.py` takes the second run on rows with C < 3,600 s and the first elsewhere. It reads
C from `outputs/mixture_v2/exp0227_r7c_realmlp_nomask_lane__v2.parquet` (written in §5.13).
Member order changes only the log.

## 5.15 Candidate file and checks

```bash
mkdir -p submissions/candidates
uv run python scripts/write_candidate.py stack_l2_cell_ng4h2G_lt1h \
  submissions/candidates/stack_l2_cell_ng4h2G_lt1h_candidate.parquet
```

`write_candidate.py` joins the test predictions to the organizers' submission template, clips
at 0, rounds to the template's integer type and validates: the two columns, the template's
344,841 ids, no duplicates, no nulls, no non-finite or negative values.

## 5.16 What "reproduced" means

- GPU training is seeded but not bit-deterministic. A rebuild is not byte-identical to our
  file.
- Measured on our runs: two mixture v2 fits of the same member with different seeds differ by
  about 14 s rmsd on the test rows, even with fold averaging. A full rebuild differs from our
  submission by an rmsd of that order, with a mean shift under 1 s.
- The gate structure does not change: rows with C ≥ 1 h come from the `stack_l2_cell_mx2_xp`
  stack, rows with C < 1 h from `stack_l2_cell_ng4h2G`.
- Out-of-fold scores of the C < 1 h stack should fall within seed noise of ours: RMSE 230.21 s
  on all 2025 rows, 196.98 s without the 15 day-rollover rows.
- The CPU steps (routing, lane heads, applies, averaging, stacks, gate, candidate) are
  deterministic given their inputs: reruns of these steps on our inputs gave sha256-identical
  outputs.

## 5.17 Known limits

| limit | effect | mitigation |
|---|---|---|
| Live external sources (OSM without a date clause, Open-Meteo revisions, X-Plane packs re-approved) | features can differ slightly from ours | each archive's `manifest.json` records the sha256 of every file for comparison |
| `delayed_flight_model.py` and the mixture frame read a block's newest cache file by modification time | a stale cache file of a block can be read | build into a clean `data/features/`; keep one cache file per block version |
| a cell can finish with a NaN fold | its rows are missing from the out-of-fold file | check `status.json` before merging (§5.7) |
| GPU non-determinism | predictions differ at seed-noise level | inherent (§5.16) |
