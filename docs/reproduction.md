# Reproducing the submission

This page runs the whole build in order, from a fresh clone to
`submissions/humble-rocket_final.parquet`. Each step gives the commands or
names the exact section that does. Two layers get built: the **team stack**
(`src/prc`, whose gated prediction the strongest final member consumes as an
expert) and the **final members** (`pipeline/`), which the blend and assembly
layer (`final/`) turns into the submission.

```text
challenge data + external archives
   │
   ├── team stack (src/prc, scripts/) ── gated run ── expert table ─┐
   │        steps 4 and 5                                           │
   └── feature matrix (pipeline/) ── final members (pipeline/) ─────┤
            step 6                        steps 7 and 8             │
                                                                    ▼
                         final/blend.py ── final/assemble.py ── final/verify.py
                                         step 9
```

Budget: the team stack and the final members are GPU work (one 48 GB card
suffices; we used L40S and H200 nodes). The matrix builds, the expert table
and everything in `final/` run on CPU; per-step resource figures are in
[team stack reproduction §5.2](05-reproduction.md#52-requirements).

## 1. Environment

```bash
git clone <repository URL> && cd <repository>
uv sync --group nn          # Python 3.12 and the locked packages, PyTorch included
```

[uv](https://docs.astral.sh/uv/) installs Python 3.12 if it is missing;
`uv.lock` pins every package. The GPU driver must support the CUDA version of
the locked PyTorch wheels.

## 2. Challenge data

The organizers issue access keys to registered teams
([data page](https://prc-data-challenge-2026.netlify.app/data.html)). Keys
never go into a file in the repository.

```bash
mc alias set prc https://s3.opensky-network.org <access-key> <secret-key>   # MinIO client, once
cp .env.example .env              # set PRC_TEAM_NAME; PRC_DATA_BUCKET is the organizers' bucket
uv run prc data pull              # mc mirror -> data/raw/
uv run prc data prepare           # -> data/interim/movements.parquet (team-stack input)
```

`data/raw/` must end up holding the twelve `training_2025-*.parquet` files,
`ranking.parquet`, `final_ranking.parquet` and `final_submitting.parquet`
(`data/README.md` lists them).

## 3. External data

Download the EUROCONTROL arrival-ATFM workbook to
`data/external/Airport_Arrival_ATFM_Delay.xlsx` (the link is in
`data/README.md` §2). Every other external source is fetched by script, on a
CPU machine with internet access
([sources and licences](data-sources.md)):

```bash
uv run python scripts/fetch_external.py ourairports   # first: the ADS-B build reads runway thresholds
uv run python scripts/fetch_external.py metar --years 2025 2026
uv run python scripts/fetch_external.py osm
uv run python scripts/fetch_external.py xplane
uv run python scripts/fetch_external.py openmeteo
uv run python scripts/adsb_build.py run --stage 1 --workers 3   # ADS-B ground tracks, ~550 GB streamed
uv run python scripts/adsb_build.py run --stage 2 --workers 3
uv run python scripts/adsb_v2.py archive
```

These are the same commands as
[team stack reproduction §5.5](05-reproduction.md#55-external-archives-state-h),
which also explains the two ADS-B archive states (110 early-2025 days are
rebuilt from full traces partway through the chain).

## 4. The team stack

Follow [team stack reproduction](05-reproduction.md) §5.6 to §5.14 exactly as
written: feature cache, the member cells with three seeds each, seed
merging, the routing flag, the traces rebuild, the no-flight-plan lane heads,
the stamp mixture, the two cell stacks and the gate. The chain ends with the
gated run `outputs/runs/stack_l2_cell_ng4h2G_lt1h`, which is the team-stack
prediction everything below consumes. One extra member cell is needed here
and is not part of §5.8: the set-attention variant the class model reads as
an expert, trained and merged like any other cell
([§5.7](05-reproduction.md#57-running-one-member-cell)):

```bash
e=exp9004_setenc_overnight_extra
for s in 1 2 3; do
  uv run python scripts/run_cell.py --config "experiments/$e.yaml" --seed "$s" \
    --scheme month_loo --out "outputs/runs/loo-$e"
done
uv run python scripts/merge_seeds.py merge-seeds "outputs/runs/loo-$e"
```

## 5. The team-stack expert table

CPU only; run from the repository root after step 4 (and after the matrix of
step 6, for the last command):

```bash
python pipeline/make_twins.py        # label-free arrival-twin table -> work/twins.parquet
python pipeline/team_base.py         # label-free projections of the gated stack -> work/v22_{oof,test}.parquet
python pipeline/team_corrector.py    # month-held residual corrector -> work/a14_*.parquet
python pipeline/team_dmodel.py       # month-held bounded-offset model -> work/a17_{oof,test}.parquet
python pipeline/team_table.py        # expert table -> work/v23_{oof,test}.parquet
python pipeline/team_adapt.py exp9004_setenc_overnight_extra__s1-2-3__month_loo t9004
```

Each script's docstring states what it fits and the pre-set rule under which
a component is kept. The result is the expert table and the `team_t9004`
member tag that the class-structured combiner reads.

## 6. The final feature matrix

```bash
python pipeline/prepare_data.py     # input check + ATFM day table
python pipeline/strike_feats.py     # strike calendar -> work/strike_feats.parquet
python pipeline/fm_build.py         # raw-derived + ADS-B columns -> work/fm.parquet
FM_IN=work/fm.parquet FM_OUT=work/fm2.parquet python pipeline/fm_v2.py   # + ATFM columns
```

`fm_build.py` reads the per-departure ADS-B ground-track feature files from
`data/external/` (`data/README.md` §3); they are the `adsb_ground` and
`adsb_ground_v2` blocks built in step 3. The retrain matrix for the final
months is the same build against the full submission set:

```bash
RANKING_FILE=final_ranking.parquet FM_OUT=work/fm_final.parquet python pipeline/fm_build.py
FM_IN=work/fm_final.parquet FM_OUT=work/fm2_final.parquet python pipeline/fm_v2.py
```

## 7. The final members

GPU. Train each member for folds 0 to 12 with the exact flags of the member
table in `pipeline/RUNS.md` (also explained on the
[member pipeline page](reference/pipeline.md)), in dependency order:

1. `gbm_v2` (its stage-1 cache is reused by every other GBM via `S1TAG=v2`),
   then `gbm_v3a`, `gbm_vap3`;
2. the transformers `ft_ftB7`, `ft_ftC7`, `ft_ftSC`, `ft_ftNE`, `ft_ftN10`
   and the auxiliary tags `ft_ftB3`, `tr_trE`, `nn_nn2`, `gbm_hf10`
   (scripts and flags in the auxiliary-tags paragraph of `pipeline/RUNS.md`);
3. `gbm_eval.py` with `SAVE_MEMBERS=1` over the trained GBM tags, which writes
   the member-deviation tables, then the anomaly specialist `gbm_anom1`;
4. the class models `cm_cmA` and `cm_cmNA` (they read the expert table of
   step 5 and every tag above).

For example, one member over all folds on a single GPU:

```bash
for FOLD in $(seq 0 12); do python pipeline/gbm_nested.py $FOLD --tag v2; done
```

or as a Slurm array (`pipeline/jobs/py.sbatch`, one fold per task). Every
member writes `work/<tag>/outer_NN.parquet`.

## 8. The fold-0 retrains

The months the primary members never predicted (February and June) are served
by fold-0 retrains on the final matrix: rerun fold 0 of `gbm_v2`, `gbm_v3a`,
`gbm_vap3`, `ft_ftB7`, `ft_ftC7`, `ft_ftSC`, `ft_ftNE`, `ft_ftN10` with
`FM=work/fm2_final.parquet` and the `F`-suffixed tags of `pipeline/RUNS.md`
(transformers with 4 seeds). The class and anomaly models are deliberately not
retrained; [the final submission](06-final-model.md) explains why.

## 9. Blend, assemble, verify

CPU, deterministic ([blend and assembly](reference/final.md)):

```bash
python final/blend.py       # per-lane month-held simplex weights, blend OOF and test rows
python final/assemble.py    # corrections + label-free rules -> submissions/humble-rocket_final.parquet
python final/verify.py      # template check, per-month stats, sha256
```

## What "reproduced" means

Given our member files, step 9 reproduces the submitted parquet byte for
byte; the CPU steps are deterministic throughout. GPU training (steps 4, 7
and 8) is seeded but not bit-deterministic, so a full rebuild matches at seed-
noise level rather than byte level; `pipeline/RUNS.md` carries the fidelity
notes, and [team stack reproduction §5.16](05-reproduction.md#516-what-reproduced-means)
quantifies the equivalent statement for the team stack.
