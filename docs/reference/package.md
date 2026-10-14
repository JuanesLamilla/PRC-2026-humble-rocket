# Package, CLI and scripts (`src/prc`, `scripts/`)

The installable package `prc` and the scripts around it implement the **team
stack**: data preparation, the content-addressed feature cache, the member
experiments, and the post-processors whose end product the class-structured
combiner consumes as an expert. The
[team stack method](../02-method.md) describes what these pieces compute; the
[reproduction guide](../05-reproduction.md) runs them end to end. This page
maps the code.

## Repository layout

| Path | Content |
|---|---|
| `src/prc/` | the package: data loading, feature blocks, models, CV, evaluation, the mixture post-processor, submission tools |
| `experiments/` | one YAML per member experiment (model kind, feature blocks, target, CV scheme, seeds) |
| `scripts/` | data fetchers and the post-processing chain (routing, lane head, mixture, stacker, gate) |
| `jobs/slurm/` | Slurm templates for the team-stack stages |
| `pipeline/` | the final members ([member pipeline](pipeline.md)) |
| `final/` | the blend and assembly layer ([blend and assembly](final.md)) |
| `assets/` | the hand-compiled strike/disruption calendar |
| `data/` | inputs, never committed; `data/README.md` says where each file comes from |

## Environment

```bash
uv sync --group nn    # Python 3.12, locked packages, PyTorch included
```

`uv.lock` pins every package. The `prc` command below is the package's
console entry point (`src/prc/cli.py`); run it as `uv run prc ...`.

## The `prc` command

| Command | What it does |
|---|---|
| `prc data pull` | mirror the challenge bucket into `data/raw/` (`mc mirror`; keys come from the environment, never from a file) |
| `prc data prepare` | build `data/interim/movements.parquet` from the raw tables |
| `prc features list` | list the registered feature blocks and their cache state |
| `prc features build --all` | build the content-addressed feature cache |
| `prc train` | run one experiment YAML (one model, one CV scheme, one seed) |
| `prc status` | show a run's folds and metrics |
| `prc blend` | blend finished runs |
| `prc submit` / `prc upload` | write a submission file from a run; upload is explicit and separate |
| `prc registry` | the last N rows of the local experiment registry |

Training at scale goes through `scripts/run_cell.py` (one experiment × one
seed × all folds, resumable), which the Slurm template
`jobs/slurm/member-cell.sbatch` wraps; see the
[reproduction guide](../05-reproduction.md#57-running-one-member-cell).

## Package layout (`src/prc/`)

| Module | Content |
|---|---|
| `config.py` | typed experiment configuration (pydantic): model kind, feature blocks, target transform, CV scheme, test fit |
| `paths.py`, `runtime.py`, `cache.py` | repository paths, `.env` loading, GPU visibility, the content-addressed cache (block source hash × input hash × external-archive hash) |
| `data/` | raw-table loading for training and ranking files |
| `features/` | the feature blocks of the [method page](../02-method.md#22-features), one module per block, assembled through `features/registry.py`; no block reads the label or the off-block time of the row it describes |
| `cv/` | fold assignment: `month_loo`, `tail_nov_dec`, `month_jan_jul` |
| `models/` | the member architectures: CatBoost, LightGBM and XGBoost wrappers, RealMLP, GRU sequence encoder (`seqenc.py`), set-attention encoder (`setenc.py`), TabM, ModernNCA, plus the matrix cache shared by the neural models |
| `eval/` | run directories, metrics, the experiment registry |
| `mixture/` | the stamp-mixture building blocks used by `scripts/mixture_v2.py`: recording-class labels, per-class value expressions, feasibility masks |
| `blend/` | run blending used by `prc blend` and `scripts/blend_fit.py` |
| `submit/` | submission writing, validation and upload |

Every experiment YAML in `experiments/` names a model kind and a feature-block
list; `prc train` resolves both through registries, so the YAML is the full
definition of a member.

## Post-processing scripts

The chain after the members, in the order the
[reproduction guide](../05-reproduction.md) runs it:

| Script | Role |
|---|---|
| `scripts/write_route_flag.py` | write the per-movement "has an ADS-B ground track" flag |
| `scripts/route_no_ground.py` | route rows without a ground track to a no-ADS-B twin member |
| `scripts/no_nm_lane_model.py` | the two-branch lane head on departures without a flight-plan record |
| `scripts/mixture_v2.py` | the six-class stamp mixture (build the frame, fit per run on GPU, apply) |
| `scripts/avg_runs.py` | average the predictions of two runs (used for the two fold-averaged mixture fits) |
| `scripts/stack_l2.py` | the cell stacker: non-negative member weights per airport × NM-record × C-band cell, shrunk toward parents |
| `scripts/gate_runs.py` | the gate on C: the final stack below 1 h, the earlier stack above |
| `scripts/write_candidate.py` | a submission-shaped candidate file, validated against the template; never uploads |

Supporting tools:

| Script | Role |
|---|---|
| `scripts/fetch_external.py` | fetch the external archives (OurAirports, METAR, OSM, X-Plane Gateway, Open-Meteo), each with a sha256 manifest |
| `scripts/adsb_build.py`, `scripts/adsb_v2.py` | build the ADS-B ground-track archives from ADSB.lol releases |
| `scripts/run_cell.py`, `scripts/merge_seeds.py` | run one experiment cell; merge per-seed cells into one run with recomputed metrics |
| `scripts/stack_inner.py` | the 39 inner GRU runs and the stack table behind the `exp0235` member |
| `scripts/paired_delta.py` | paired RMSE deltas between two runs with day-block bootstrap intervals, per row group ([validation protocol](../03-validation.md)) |
| `scripts/blend_fit.py`, `scripts/lane_report.py` | blend finished runs under the validation protocol; lane breakdown of a run |
| `scripts/delayed_flight_model.py`, `scripts/eda_no_nm_block_model.py` | post-processors over a finished run for delayed flights and the no-flight-plan lane |
| `scripts/adsb_pilot.py`, `scripts/adsb_pilot_phase2.py` | the pilot studies that established ADS-B ground tracks as a feature source |

Every script prints its usage in its module docstring; the ones that take a
run name read `outputs/runs/<run>/` as written by `prc train` or
`scripts/run_cell.py`.
