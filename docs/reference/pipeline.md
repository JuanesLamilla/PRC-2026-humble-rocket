# Member pipeline (`pipeline/`)

`pipeline/` holds the self-contained scripts that build the feature matrix and
train every member behind the submitted file. Each script runs as a plain
`python` call from the repository root; the GPU scripts also ran as Slurm array
jobs with the generic templates in `pipeline/jobs/` (one H200, 8 CPUs;
`jobs/py.sbatch <script> <args>` or an `--array=0-12` loop over the fold
argument). `pipeline/RUNS.md` is the canonical record of every training run:
the tag it wrote, and the flags that define it.

## Paths and environment variables

The scripts share `pipeline/common.py`, which resolves every path from the
environment with these defaults:

| Variable | Default | Meaning |
|---|---|---|
| `RAW_DIR` | `data/raw` | challenge movement tables |
| `FEATURES_DIR` | `data/external` | ADS-B ground-track feature files |
| `RANKING_FILE` | `ranking.parquet` | the evaluation set joined as the test split |
| `RESULTS_DIR` | `results` | text reports |

Member outputs land under `work/` (created on first use). `common.py` also
carries the join-window constant `W = 3606` (the verified maximum of
|BLOCK − LOBT| over every 2025 departure with an NM record) and the shared
loaders for the movement tables.

## The fold scheme

Every member is trained thirteen times:

- **folds 1 to 12**: that 2025 month is held out and the member writes its
  out-of-fold prediction for it;
- **fold 0**: the member fits on all twelve 2025 months and predicts the
  evaluation rows named by `RANKING_FILE`.

Each fold writes `work/<tag>/outer_NN.parquet`. The blend layer
([final/](final.md)) consumes these files by tag.

## Stage 0 and 1: inputs and the feature matrix

```bash
python pipeline/prepare_data.py                     # input check + ATFM day table
python pipeline/strike_feats.py                     # strike calendar -> work/strike_feats.parquet
python pipeline/fm_build.py                         # raw-derived + ADS-B columns -> work/fm.parquet
FM_IN=work/fm.parquet FM_OUT=work/fm2.parquet python pipeline/fm_v2.py   # + ATFM columns
```

- **`prepare_data.py`** verifies the challenge tables and ADS-B feature files
  are in place (`data/README.md` lists where each one comes from) and converts
  the EUROCONTROL ATFM workbook to `data/raw/apt_arr_atfm.parquet`.
- **`strike_feats.py`** turns `assets/strike_calendar.csv` (hand-compiled from
  public news sources) into per-departure strike and disruption features keyed
  by `MVT_ID_mvt`.
- **`fm_build.py`** builds the departure feature matrix: the raw-derived
  blocks of `newfeat.py` (stand handover, arrival-default shares, NM-neighbour
  taxi, NM plan residual, day no-NM share), the ADS-B ground-track columns,
  and the LOBT-window and label-structure columns. The test split is whatever
  `RANKING_FILE` names.
- **`fm_v2.py`** appends the EUROCONTROL daily airport-arrival ATFM delay
  features (airport × UTC date; open data; null means no regulation and maps
  to 0).

The **retrain matrix** for the final months is the same build against the full
submission set:

```bash
RANKING_FILE=final_ranking.parquet FM_OUT=work/fm_final.parquet python pipeline/fm_build.py
FM_IN=work/fm_final.parquet FM_OUT=work/fm2_final.parquet python pipeline/fm_v2.py
```

## Stage 2 and 3: the member scripts

Three model families, all run per fold
(`python pipeline/<script>.py $FOLD --tag <tag> [flags]`):

### Two-stage GBM: `gbm_nested.py`

Nested two-stage GPU XGBoost on the full matrix. Stage 1 predicts properties
of the off-block stamp, fitted with inner month-held folds over the training
months so no stage-2 training row sees an in-fold stage-1 fit:

| Stage-1 output | Meaning |
|---|---|
| `d_hat` | mean of d = BLOCK − LOBT (NM rows; bounded ±3606) |
| `dq10/50/90` | quantiles of d, giving predicted bounds on the label |
| `p_def` | P(\|BLOCK − SCHED\| ≤ 10 s), the schedule-default probability |
| `p_gt1h` | P(label > 1 h and not a schedule default) |

Stage 2 fits the taxi time: `y_base` (no stage-1 features), `y_s1` (with
them), and the mixture `y_mix = p_def · C + (1 − p_def) · normal-model`.
Variants are selected with `--cfg` (and reuse a tagged stage-1 cache via
`S1TAG`): a CatBoost mixture, the mixture fitted per airport, and a four-class
stamp specialist that sees the other members' disagreement as features
(`work/members_oof.parquet` / `members_test.parquet`, written by `gbm_eval.py`
with `SAVE_MEMBERS=1`).

### Distributional FT-transformer: `ftdist.py`

A feature-token transformer (after Gorishniy et al. 2021) whose head is one
softmax over taxi-time bins **masked to the NM join window**, plus a point
mass at y = C masked where a schedule default is impossible. Targets are
Gaussian-smoothed histograms (HL-Gauss); the prediction is the distribution
mean, the squared-error optimum. The loss is cross-entropy plus a small MSE
term on the mean. Rollover rows are left out of training, as for every member.
Variants come from flags: `--s1 v2` adds the nested stage-1 outputs as extra
tokens, `--strikes` adds the strike-calendar flags, `--drop eobt1,arvt1,=me`
removes every feature anchored on the filed EOBT (the EOBT-free variant),
`--w-nonm 10` upweights the no-flight-plan rows.

### Class-structured combiner: `class_mix.py`

The strongest single member combines member *components* instead of final
predictions:

```text
ŷ = p_def·C + p_near·(C − offset) + p_long·v_long + p_normal·Σ_j g_j·n_j
```

One MLP outputs the class probabilities (the default class masked where
infeasible), the gate g over the members' conditional normal means n_j, and
small bounded corrections, from every member's class signals: the
transformer's p_def, the GBM stage-1 p_def and p_gt1h, the anomaly
specialist's class probabilities and long-stamp value, the bounded offset
model, the [team stack's](../02-method.md) prediction, and context features.
The loss is MSE plus a class log-loss on the observed recording class.
`--lane nonm` trains a second instance on the no-flight-plan rows only.

## The trained members

From `pipeline/RUNS.md` (the primary runs; `FM=work/fm2.parquet`):

| Tag | Script and flags | What it is |
|---|---|---|
| `gbm_v2` | `gbm_nested.py $FOLD --tag v2` | two-stage XGBoost: nested stage-1 stamp models, then `y_base`/`y_s1`/`y_mix` |
| `gbm_v3a` | `S1TAG=v2 gbm_nested.py $FOLD --tag v3a --only-cat --cfg cat_d10s1,catmix` | CatBoost mixture (`y_catmix`) + plain deep CatBoost |
| `gbm_vap3` | `S1TAG=v2 gbm_nested.py $FOLD --tag vap3 --only-cat --cfg catmixapy_s201` | the mixture fitted per airport |
| `gbm_anom1` | `S1TAG=v2 gbm_nested.py $FOLD --tag anom1 --only-cat --cfg anom_s1` | four-class stamp specialist over member deviations |
| `ft_ftB7` | `ftdist.py $FOLD --tag ftB7 --seeds 0..6` | distributional transformer, 7 seeds |
| `ft_ftC7` | `ftdist.py $FOLD --tag ftC7 --seeds 0..6 --s1 v2` | with stage-1 outputs as extra tokens |
| `ft_ftSC` | `ftdist.py $FOLD --tag ftSC --seeds 0,1,2 --strikes work/strike_feats.parquet` | strike-calendar tokens |
| `ft_ftNE` | `ftdist.py $FOLD --tag ftNE --seeds 1..18 --drop eobt1,arvt1,=me` | EOBT-free variant |
| `ft_ftNE4` | first four seeds of `ft_ftNE`, merged | 4-seed average for the long-delay lane |
| `ft_ftN10` | `ftdist.py $FOLD --tag ftN10 --seeds 0..9 --w-nonm 10` | no-flight-plan rows upweighted |
| `cm_cmA` | `class_mix.py --tag cmA --more --plus --seeds 0..16` | class-structured combiner, 17 seeds |
| `cm_cmNA` | `class_mix.py --tag cmNA --lane nonm --seeds 0..44` | the no-flight-plan instance, 45 seeds |

**Fold-0 retrains** for the final months (`gbm_v2F`, `gbm_v3aF`, `gbm_vap3F`,
`ft_ftCF`, `ft_ftNEFF`, `ft_ftB7F`, `ft_ftSCF`, `ft_ftN10F`) use the same
flags against the retrain matrix, fold 0 only, transformers with 4 seeds. No
retrain exists for the class and anomaly models; the
[final-model page](../06-final-model.md) explains why they are excluded on the
retrain months.

**Auxiliary tags** consumed by `cm_cmA` as inputs (not blend members):
`ft_ftB3` (3-seed base transformer), `tr_trE` (`tabr.py`, a retrieval-augmented
network in the TabR style), `nn_nn2` (`nn_mix.py`, a window-bounded neural
mixture), `gbm_hf10` (`gbm_nested.py --cfg catmixd_json` with trial 10 of
`hpo_trials.json`), and `team_t9004` (the team member
`experiments/exp9004_setenc_overnight_extra.yaml`, converted by
`team_adapt.py`).

## The team-stack expert table

The class-structured combiner reads the team stack's prediction as one of its
experts. Five CPU scripts turn the gated team-stack run
(`outputs/runs/stack_l2_cell_ng4h2G_lt1h`, the end of the
[team stack reproduction](../05-reproduction.md)) into the table and tag the
combiner consumes; the ordered commands are in
[reproducing the submission](../reproduction.md#5-the-team-stack-expert-table)
and in `pipeline/RUNS.md`:

| Script | What it does |
|---|---|
| `make_twins.py` | the label-free arrival-twin table (`work/twins.parquet`) |
| `team_base.py` | label-free projections of the gated stack: window clip, in-window model, rollover identity, twin cap, a LIRF offset (`work/v22_{oof,test}.parquet`) |
| `team_corrector.py` | a month-held residual corrector with stand-handover and arrival-stamp features the members do not see directly (`work/a14_*.parquet`) |
| `team_dmodel.py` | a month-held model of the bounded off-block offset d = BLOCK − LOBT on NM rows (`work/a17_{oof,test}.parquet`) |
| `team_table.py` | the expert table: the base plus each corrector that passes its pre-set month-held rule (`work/v23_{oof,test}.parquet`) |
| `team_adapt.py` | converts a team run directory into a member tag (`work/team_<short>/outer_NN.parquet`) |

**`gbm_eval.py TAG [extra_tags...]`** scores nested member outputs against the
expert table and stacks them with it, per lane, month-held; with
`SAVE_MEMBERS=1` it writes the member-deviation tables the anomaly specialist
reads.

## Reproduction notes

The table above reconstructs each run's flags from the scripts' defaults and
our run records; the full `argv` of every historical run was not retained, and
retraining a GPU member reproduces its predictions only up to hardware
nondeterminism. The blend and assembly layer ([final/](final.md)) is exactly
deterministic given the member files: re-running it reproduces the submitted
parquet byte for byte.
