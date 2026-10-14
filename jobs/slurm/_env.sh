# Sourced by every template in jobs/slurm/. Not executable on its own.
#
# The templates are written for a SLURM cluster: submit them from the repository root with
# sbatch. Each one also runs without SLURM (`bash jobs/slurm/<name>.sbatch`, with
# SLURM_ARRAY_TASK_ID set by hand for the array templates).
#
# Site-specific settings (account, partition, time limit, QoS, module loads) are not in the
# templates. Each template carries them as disabled `##SBATCH` lines: remove one `#` and fill in.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f pyproject.toml ] || [ ! -d src/prc ]; then
  echo "ERROR: run from the repository root (pyproject.toml and src/prc/ must be here)" >&2
  exit 2
fi
mkdir -p jobs/logs

# One process per task, all of the task's cores.
_threads="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="$_threads" POLARS_MAX_THREADS="$_threads" OPENBLAS_NUM_THREADS="$_threads"
export MKL_NUM_THREADS="$_threads" NUMEXPR_NUM_THREADS="$_threads"
unset _threads
# The challenge parquet files carry an R extension type that polars loads as its storage type.
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="${POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR:-load_as_storage}"

command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH (https://docs.astral.sh/uv/)" >&2; exit 3; }
[ -x .venv/bin/python ] || { echo "ERROR: no .venv; run once: uv sync --group nn" >&2; exit 3; }

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none}" \
     "cpus=${SLURM_CPUS_PER_TASK:-?} git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

# require_gpu: fail early when a GPU template lands on a node without a visible GPU.
require_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    echo "ERROR: no GPU visible. Request one (--gpus-per-node=1) on a GPU partition." >&2
    exit 7
  fi
  nvidia-smi -L
}
