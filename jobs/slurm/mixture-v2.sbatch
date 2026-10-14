#!/bin/bash
# Mixture v2 (scripts/mixture_v2.py): the stamp-class mixture fitted on one lane run, on one GPU.
#
#   MODE=build   the member-independent frame of every departure (once, before any fit);
#                writes outputs/mixture_v2/_frames/
#   MODE=fit     (default) one fit per array task over RUNS (comma-separated lane run names
#                under outputs/runs/), or one fit of RUN; writes
#                outputs/mixture_v2/<run>__v2[__<TAG>].{parquet,json}
#
#   mkdir -p jobs/logs
#   J=$(MODE=build sbatch --parsable --export=ALL,MODE jobs/slurm/mixture-v2.sbatch)
#   RUNS=a_lane,b_lane TAG=fa0 MX_ARGS=--test-foldavg \
#     sbatch --dependency=afterok:$J --array=0-1 --export=ALL,RUNS,TAG,MX_ARGS jobs/slurm/mixture-v2.sbatch
#
# The fit settings of the final chain are fixed here: --device cuda --batch 4096 --lr 2e-3
# --epochs 40 --patience 5 --n-seeds 8. MX_ARGS adds flags (e.g. "--test-foldavg --seed 1").
# Apply the fits on CPU afterwards: scripts/mixture_v2.py apply.
# Resources: one CUDA GPU with 48 GB, 8 CPU cores, 64 GB RAM.

#SBATCH --job-name=prc-mixture-v2
##SBATCH --account=<account>          # site-specific
##SBATCH --partition=<gpu partition>  # site-specific
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
##SBATCH --time=<walltime>         # set for your cluster
#SBATCH --output=jobs/logs/%x-%A_%a.out
#SBATCH --error=jobs/logs/%x-%A_%a.err

_env="${SLURM_SUBMIT_DIR:-$PWD}/jobs/slurm/_env.sh"
[ -r "$_env" ] || { echo "ERROR: $_env not found; submit from the repository root" >&2; exit 2; }
source "$_env"

if [ "${MODE:-fit}" = "build" ]; then
  echo "[mixture-v2] build the all-departures frame"
  uv run python scripts/mixture_v2.py build --audit
  exit 0
fi

if [ -n "${RUNS:-}" ]; then
  IFS=',' read -ra _runs <<< "$RUNS"
  IDX="${SLURM_ARRAY_TASK_ID:?RUNS needs an array job: --array=0-N}"
  [ "$IDX" -lt "${#_runs[@]}" ] || { echo "ERROR: no run at index $IDX in RUNS=$RUNS" >&2; exit 6; }
  RUN="${_runs[$IDX]}"
fi
RUN="${RUN:?set RUN=<lane run under outputs/runs/> or RUNS=a,b with --array}"
for f in "outputs/runs/${RUN}/oof.parquet" "outputs/runs/${RUN}/test_pred.parquet"; do
  [ -f "$f" ] || { echo "ERROR: $f missing" >&2; exit 7; }
done

require_gpu
args=(fit --run "$RUN" --device cuda --batch 4096 --lr 2e-3 --epochs 40 --patience 5 --n-seeds 8)
[ -n "${TAG:-}" ] && args+=(--tag "$TAG")
# shellcheck disable=SC2206
[ -n "${MX_ARGS:-}" ] && args+=(${MX_ARGS})
echo "[mixture-v2] + uv run python scripts/mixture_v2.py ${args[*]}"
uv run python scripts/mixture_v2.py "${args[@]}"
echo "[mixture-v2] done: outputs/mixture_v2/${RUN}__v2${TAG:+__$TAG}.parquet"
