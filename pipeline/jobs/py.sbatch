#!/bin/bash
#SBATCH --job-name=prc-py
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=p65425
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
# usage: sbatch [--job-name X] jobs/py.sbatch script.py [args...]
set -euo pipefail
echo "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST} cpus ${SLURM_CPUS_PER_TASK} start $(date)"
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
WORK_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "${PROJECT_DIR}/results"
if ! type module >/dev/null 2>&1; then set +eu; source /etc/profile >/dev/null 2>&1; set -eu; fi
module load python/3.11
module load gcc arrow
VENV="${PROJECT_DIR}/.venv311"
if [ ! -x "${VENV}/bin/python" ]; then
  python -m venv "${VENV}"; source "${VENV}/bin/activate"
  pip install --no-index --upgrade pip; pip install --no-index -r "${PROJECT_DIR}/requirements.txt"
else source "${VENV}/bin/activate"; fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" POLARS_MAX_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 RESULTS_DIR="${PROJECT_DIR}/results"
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
cd "${PROJECT_DIR}"
set +e
python "$@"
EXIT_CODE=$?
set -e
echo "end $(date) exit ${EXIT_CODE}"
exit ${EXIT_CODE}
