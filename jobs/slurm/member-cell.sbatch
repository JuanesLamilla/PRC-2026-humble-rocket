#!/bin/bash
# One member cell per array task: one experiment YAML x one seed x every fold of one CV scheme.
#
#   mkdir -p jobs/logs
#   CONFIG=experiments/exp0225_r7c_realmlp_taxi.yaml \
#   OUT_ROOT=outputs/runs/loo-exp0225_r7c_realmlp_taxi SEEDS=1,2,3 SCHEME=month_loo \
#     sbatch --array=0-2 --export=ALL,CONFIG,OUT_ROOT,SEEDS,SCHEME jobs/slurm/member-cell.sbatch
#
# Task i runs seed i of SEEDS (scripts/run_cell.py) and writes
# OUT_ROOT/<exp>__s<seed>__<scheme>/. A cell whose metrics.json exists is skipped, so a
# resubmission resumes. Merge the seeds afterwards:
#   uv run python scripts/merge_seeds.py merge-seeds OUT_ROOT
#
# SCHEME: month_loo (12 month-held folds; with `test_fit: full` in the YAML the test
# prediction comes from one more fit on all of 2025) or tail_nov_dec (train January to
# October, validate November and December).
#
# Without SLURM:  SLURM_ARRAY_TASK_ID=0 CONFIG=... OUT_ROOT=... bash jobs/slurm/member-cell.sbatch
# DRY_RUN=1 resolves the cell (config, folds, output directory) and fits nothing.
#
# Resources: one CUDA GPU with 48 GB (NVIDIA L40S), 8 CPU cores, 80 GB RAM.

#SBATCH --job-name=prc-cell
##SBATCH --account=<account>          # site-specific
##SBATCH --partition=<gpu partition>  # site-specific
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gpus-per-node=1
##SBATCH --time=<walltime>         # set for your cluster
#SBATCH --output=jobs/logs/%x-%A_%a.out
#SBATCH --error=jobs/logs/%x-%A_%a.err

_env="${SLURM_SUBMIT_DIR:-$PWD}/jobs/slurm/_env.sh"
[ -r "$_env" ] || { echo "ERROR: $_env not found; submit from the repository root" >&2; exit 2; }
source "$_env"

CONFIG="${CONFIG:?set CONFIG=experiments/<exp>.yaml}"
SEEDS="${SEEDS:-1,2,3}"
SCHEME="${SCHEME:-month_loo}"
IDX="${SLURM_ARRAY_TASK_ID:?array job: pass --array=0-N (or set SLURM_ARRAY_TASK_ID)}"
EXP="$(basename "$CONFIG" .yaml)"
OUT_ROOT="${OUT_ROOT:-outputs/runs/${SCHEME}-${EXP}}"

IFS=',' read -ra _seeds <<< "$SEEDS"
[ "$IDX" -lt "${#_seeds[@]}" ] || { echo "ERROR: no seed at index $IDX in SEEDS=$SEEDS" >&2; exit 6; }
SEED="${_seeds[$IDX]}"

echo "[cell] config=$CONFIG seed=$SEED scheme=$SCHEME -> $OUT_ROOT"
if [ "${DRY_RUN:-0}" = "1" ]; then
  uv run python scripts/run_cell.py --config "$CONFIG" --seed "$SEED" --scheme "$SCHEME" --out "$OUT_ROOT" --dry-run
  exit 0
fi
require_gpu
uv run python scripts/run_cell.py --config "$CONFIG" --seed "$SEED" --scheme "$SCHEME" --out "$OUT_ROOT"
echo "[cell] done"
