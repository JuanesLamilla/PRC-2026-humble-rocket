#!/bin/bash
#SBATCH --job-name=prc-gpu
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gpus-per-node=1
#SBATCH --account=p65425
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
# usage: sbatch [--job-name X] jobs/py.sbatch script.py [args...]
set -euo pipefail
echo "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST} cpus ${SLURM_CPUS_PER_TASK} start $(date)"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
WORK_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "${PROJECT_DIR}/results"
if ! type module >/dev/null 2>&1; then set +eu; source /etc/profile >/dev/null 2>&1; set -eu; fi
module load python/3.11
module load gcc arrow cuda
VENV="${PROJECT_DIR}/.venv_gpu"
if [ ! -x "${VENV}/bin/python" ]; then
  python -m venv "${VENV}"; source "${VENV}/bin/activate"
  pip install --no-index --upgrade pip; pip install --no-index -r "${PROJECT_DIR}/requirements_gpu.txt"
else source "${VENV}/bin/activate"; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 RESULTS_DIR="${PROJECT_DIR}/results"
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
cd "${PROJECT_DIR}"
set +e
python pipeline/tabr.py ${SLURM_ARRAY_TASK_ID} "$@"
EXIT_CODE=$?
set -e
echo "end $(date) exit ${EXIT_CODE}"
exit ${EXIT_CODE}
