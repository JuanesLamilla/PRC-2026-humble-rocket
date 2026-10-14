# Member training runs

This file records how every member behind `humble-rocket_final.parquet` was
trained: the scripts in this directory, the matrix they read, the tag each run
wrote, and the flags that define it. The blend layer (`final/`) consumes the
member outputs by tag; `docs/06-final-model.md` explains what each member is
and why it is in the blend.

All GPU runs used the generic templates in `jobs/` on a Slurm cluster (one
H200, 8 CPUs; `jobs/py.sbatch <script> <args>` or an `--array=0-12` loop over
the fold argument). Everything also runs as a plain `python` call with CUDA.

## Matrix

```bash
python pipeline/prepare_data.py                       # input check + ATFM day table
python pipeline/strike_feats.py                       # strike calendar -> per-departure features
python pipeline/fm_build.py                           # raw-derived + ADS-B columns -> work/fm.parquet
FM_IN=work/fm.parquet FM_OUT=work/fm2.parquet python pipeline/fm_v2.py   # + ATFM columns
```

The primary matrix carries `ranking.parquet` (January/July) as its test split.
The retrain matrix for the final months is the same build against the full
submission set:

```bash
RANKING_FILE=final_ranking.parquet FM_OUT=work/fm_final.parquet python pipeline/fm_build.py
FM_IN=work/fm_final.parquet FM_OUT=work/fm2_final.parquet python pipeline/fm_v2.py
```

## Members (13 folds each: fold 1..12 = that 2025 month held out, fold 0 = test)

Every member writes `work/<tag>/outer_NN.parquet`. `FM=work/fm2.parquet` for
the primary runs; the fold-0 retrains (tags ending in `F`) ran fold 0 only with
`FM=work/fm2_final.parquet` and are otherwise identical.

| tag | script and flags | what it is |
|---|---|---|
| `gbm_v2` | `gbm_nested.py $FOLD --tag v2` | two-stage XGBoost: nested stage-1 stamp models, then `y_base`/`y_s1`/`y_mix` |
| `gbm_v3a` | `S1TAG=v2 gbm_nested.py $FOLD --tag v3a --only-cat --cfg cat_d10s1,catmix` | CatBoost mixture (`y_catmix`) + plain deep CatBoost |
| `gbm_vap3` | `S1TAG=v2 gbm_nested.py $FOLD --tag vap3 --only-cat --cfg catmixapy_s201` | the mixture fitted per airport, y-target |
| `gbm_anom1` | `S1TAG=v2 gbm_nested.py $FOLD --tag anom1 --only-cat --cfg anom_s1` | four-class stamp specialist; needs `work/members_oof.parquet` / `members_test.parquet` (written by `gbm_eval.py` with `SAVE_MEMBERS=1`) |
| `ft_ftB7` | `ftdist.py $FOLD --tag ftB7 --seeds 0,1,2,3,4,5,6` | distributional FT-transformer, 7 seeds |
| `ft_ftC7` | `ftdist.py $FOLD --tag ftC7 --seeds 0,1,2,3,4,5,6 --s1 v2` | the same with the nested stage-1 outputs as extra tokens |
| `ft_ftSC` | `ftdist.py $FOLD --tag ftSC --seeds 0,1,2 --strikes work/strike_feats.parquet` | strike-calendar flags as extra tokens |
| `ft_ftNE` | `ftdist.py $FOLD --tag ftNE --seeds 1..18 --drop eobt1,arvt1,=me` | EOBT-free variant (no features anchored on the filed EOBT) |
| `ft_ftNE4` | first four seeds of `ft_ftNE`, merged | the 4-seed average used in the long-delay lane |
| `ft_ftN10` | `ftdist.py $FOLD --tag ftN10 --seeds 0..9 --w-nonm 10` | no-flight-plan rows upweighted ten times |
| `cm_cmA` | `class_mix.py --tag cmA --more --plus --seeds 0..16` | class-structured combiner over member components (17 seeds); also consumes the team-stack expert table (`work/v23_oof.parquet` / `v23_test.parquet`, built below) and the auxiliary tags below |
| `cm_cmNA` | `class_mix.py --tag cmNA --lane nonm --seeds 0..44` | the same model trained on the no-flight-plan rows only (45 seeds) |

Fold-0 retrains for the final months: `gbm_v2F`, `gbm_v3aF`, `gbm_vap3F`,
`ft_ftCF`, `ft_ftNEFF`, `ft_ftB7F`, `ft_ftSCF`, `ft_ftN10F` (same flags, fold 0,
retrain matrix; transformers with 4 seeds). No retrain exists for the class
and anomaly models; `docs/06-final-model.md` explains why they are excluded
on the retrain months.

Auxiliary tags consumed by `cm_cmA` as inputs (not blend members): `ft_ftB3`
(3-seed base transformer), `tr_trE` (`tabr.py`, retrieval-augmented model),
`nn_nn2` (`nn_mix.py`), `gbm_hf10` (`gbm_nested.py --cfg catmixd_json` with
trial 10 of `hpo_trials.json`), `team_t9004` (the team member
`experiments/exp9004_setenc_overnight_extra.yaml`, trained on `month_loo` per
`docs/05-reproduction.md` §5.7 and converted with `team_adapt.py` below).

## Team-stack expert table (inputs of `cm_cmA`)

Built from the team stack of `docs/02-method.md` after its full chain
(`docs/05-reproduction.md`) has produced the gated run
`outputs/runs/stack_l2_cell_ng4h2G_lt1h` and the feature cache. CPU except the
member run behind `team_t9004`:

```bash
python pipeline/make_twins.py        # label-free arrival-twin table -> work/twins.parquet
python pipeline/team_base.py         # label-free projections of the gated stack -> work/v22_{oof,test}.parquet
python pipeline/team_corrector.py    # month-held residual corrector -> work/a14_*.parquet
python pipeline/team_dmodel.py       # month-held bounded-offset model -> work/a17_{oof,test}.parquet
python pipeline/team_table.py        # expert table -> work/v23_{oof,test}.parquet
python pipeline/team_adapt.py exp9004_setenc_overnight_extra__s1-2-3__month_loo t9004
```

`team_base.py` and `team_adapt.py` read the team runs from `RUNS_DIR` (default
`outputs/runs` of this repository); `team_adapt.py` needs `work/fm2.parquet`,
so run it after the matrix build.

A note on fidelity: the table above reconstructs each run's flags from the
scripts' defaults and our run records; the full `argv` of every historical run
was not retained, and retraining a GPU member reproduces its predictions only
up to hardware nondeterminism. The blend and assembly layer (`final/`) is exactly
deterministic given the member files: re-running it reproduces the submitted
parquet byte for byte (`final/verify.py` prints the sha256).
