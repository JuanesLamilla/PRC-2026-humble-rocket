#!/bin/bash
# Challenge data and feature cache (CPU): mirror the organizers' bucket into data/raw (only
# when data/raw is empty), build data/interim/movements.parquet, then every feature block.
# Run it twice: once before the external archives exist (blocks that read them are built as
# all-null columns, under a different cache key) and once after (docs/05-reproduction.md).
#
# The mirror needs an `mc` alias named prc with the organizer-issued keys, set once by hand:
#   mc alias set prc https://s3.opensky-network.org <access-key> <secret-key>
# and PRC_DATA_BUCKET in .env (see .env.example). Keys never go into a file in the repository.
#
#   mkdir -p jobs/logs && sbatch jobs/slurm/stage-data.sbatch

#SBATCH --job-name=prc-stage-data
##SBATCH --account=<account>          # site-specific
##SBATCH --partition=<cpu partition>  # site-specific
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
##SBATCH --time=<walltime>         # set for your cluster
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err

_env="${SLURM_SUBMIT_DIR:-$PWD}/jobs/slurm/_env.sh"
[ -r "$_env" ] || { echo "ERROR: $_env not found; submit from the repository root" >&2; exit 2; }
source "$_env"

if [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
  command -v mc >/dev/null 2>&1 || { echo "ERROR: mc (MinIO client) not on PATH" >&2; exit 5; }
  uv run prc data pull
fi
uv run prc data prepare
uv run prc features build --all
du -sh data/interim data/features 2>/dev/null || true
