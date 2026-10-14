#!/bin/bash
# The inner runs of the stacked member exp0235: the GRU member exp0223 refitted inside each
# outer fold's 11 training months (3 interleaved month groups), one array task per
# (seed, outer training-month set). 39 tasks for 3 seeds:
#   uv run python scripts/stack_inner.py plan --member exp0223_r7c_seqenc_slog
#
#   mkdir -p jobs/logs
#   MEMBER=exp0223_r7c_seqenc_slog OUT_ROOT=outputs/runs/inner-exp0223_r7c_seqenc_slog \
#     sbatch --array=0-38%8 --export=ALL,MEMBER,OUT_ROOT jobs/slurm/stack-inner.sbatch
#
# Optional speed switches (they do not change predictions):
#   PRC_NN_FOLD_WORKERS=2 PRC_NN_PREP_THREADS=4 PRC_CACHE_HEAD=1
#
# Then build the stack table with jobs/slurm/stack-build.sbatch.
# Resources: one CUDA GPU with 48 GB per task, 16 CPU cores, 160 GB RAM.

#SBATCH --job-name=prc-stack-inner
##SBATCH --account=<account>          # site-specific
##SBATCH --partition=<gpu partition>  # site-specific
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1
##SBATCH --time=<walltime>         # set for your cluster
#SBATCH --output=jobs/logs/%x-%A_%a.out
#SBATCH --error=jobs/logs/%x-%A_%a.err

_env="${SLURM_SUBMIT_DIR:-$PWD}/jobs/slurm/_env.sh"
[ -r "$_env" ] || { echo "ERROR: $_env not found; submit from the repository root" >&2; exit 2; }
source "$_env"

MEMBER="${MEMBER:-exp0223_r7c_seqenc_slog}"
OUT_ROOT="${OUT_ROOT:-outputs/runs/inner-${MEMBER}}"
IDX="${SLURM_ARRAY_TASK_ID:?array job: pass --array=0-38 (or set SLURM_ARRAY_TASK_ID)}"

require_gpu
echo "[stack-inner] member=$MEMBER task=$IDX -> $OUT_ROOT"
uv run python scripts/stack_inner.py inner --member "$MEMBER" --groups 3 --out "$OUT_ROOT" --index "$IDX"
echo "[stack-inner] done"
