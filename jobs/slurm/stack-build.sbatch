#!/bin/bash
# The stack table of exp0235 (CPU), after every inner run and the GRU member's month_loo and
# tail_nov_dec cells. Writes data/stack/exp0223_r7c_seqenc_slog__g3/.
#
#   sbatch jobs/slurm/stack-build.sbatch
#
# Defaults match the run roots of docs/05-reproduction.md; override MEMBER, INNER_ROOT, LOO
# and SERVED (with --export=ALL,...) for other roots.

#SBATCH --job-name=prc-stack-build
##SBATCH --account=<account>          # site-specific
##SBATCH --partition=<cpu partition>  # site-specific
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
##SBATCH --time=<walltime>         # set for your cluster
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err

_env="${SLURM_SUBMIT_DIR:-$PWD}/jobs/slurm/_env.sh"
[ -r "$_env" ] || { echo "ERROR: $_env not found; submit from the repository root" >&2; exit 2; }
source "$_env"

MEMBER="${MEMBER:-exp0223_r7c_seqenc_slog}"
INNER_ROOT="${INNER_ROOT:-outputs/runs/inner-${MEMBER}}"
LOO="${LOO:-outputs/runs/loo-${MEMBER}/${MEMBER}__s{seed}__month_loo}"
SERVED="${SERVED:-x11-12=outputs/runs/fwd-${MEMBER}/${MEMBER}__s{seed}__tail_nov_dec}"

uv run python scripts/stack_inner.py build --member "$MEMBER" --groups 3 \
  --inner-root "$INNER_ROOT" --loo "$LOO" --served "$SERVED"
