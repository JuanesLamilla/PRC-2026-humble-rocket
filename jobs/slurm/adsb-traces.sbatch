#!/bin/bash
# Rebuild 2025-01-01..2025-04-20 (110 days) of the ADS-B ground-track archive from the full
# ADSB.lol traces. The heatmap files of those days drop ground tracks at EGLL, EDDF and EHAM.
# Run it only after the H members are trained and the routing flag is written
# (docs/05-reproduction.md). Three modes, chained with afterok:
#
#   MODE=prep (1 task)        back up the 110 heatmap days to
#                             data/external/_backup_heatmap_2025-01-01_2025-04-20/ (kept)
#                             and warm the release-list cache the pull tasks share
#   MODE=pull (11 tasks)      task i rebuilds days 10*i .. 10*i+9 from the traces; needs
#                             internet access; transient files (<= 9 GB per task) in .adsb_tmp/
#   MODE=v2   (1 task)        check that all 110 days say source=traces, rebuild the v2 archive
#                             for those days, then the adsb_ground / adsb_ground_v2 feature cache
#
#   mkdir -p jobs/logs
#   P=$(MODE=prep sbatch --parsable --export=ALL,MODE jobs/slurm/adsb-traces.sbatch)
#   Q=$(MODE=pull sbatch --parsable --dependency=afterok:$P --array=0-10%8 --export=ALL,MODE jobs/slurm/adsb-traces.sbatch)
#   MODE=v2 sbatch --dependency=afterok:$Q --export=ALL,MODE jobs/slurm/adsb-traces.sbatch
#
# To return to the heatmap state: copy the backup's four folders and manifest.json back over
# data/external/ and delete the v2 cache files of those days.
# Resources: CPU only; internet access; up to 9 GB transient disk per pull task.

#SBATCH --job-name=prc-adsb-traces
##SBATCH --account=<account>          # site-specific
##SBATCH --partition=<cpu partition>  # site-specific
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
##SBATCH --time=<walltime>         # set for your cluster
#SBATCH --output=jobs/logs/%x-%A_%a.out
#SBATCH --error=jobs/logs/%x-%A_%a.err

_env="${SLURM_SUBMIT_DIR:-$PWD}/jobs/slurm/_env.sh"
[ -r "$_env" ] || { echo "ERROR: $_env not found; submit from the repository root" >&2; exit 2; }
source "$_env"

MODE="${MODE:?set MODE=prep|pull|v2}"
FIRST=2025-01-01
NDAYS=110
EXT=data/external
BK="$EXT/_backup_heatmap_2025-01-01_2025-04-20"

days_list() {  # the 110 ISO days, one per line
  uv run python -c "
from datetime import date, timedelta
d = date.fromisoformat('$FIRST')
for i in range($NDAYS): print(d + timedelta(days=i))"
}

case "$MODE" in
prep)
  if [ -d "$BK" ]; then
    echo "[prep] backup exists ($BK), not overwritten"
  else
    mkdir -p "$BK"/adsblol_ground "$BK"/adsblol_points "$BK"/adsblol_ground_v2 "$BK"/adsblol_v2_cache
    for d in $(days_list); do
      cp -p "$EXT/adsblol_ground/$d.parquet" "$BK/adsblol_ground/"
      cp -p "$EXT/adsblol_points/$d.parquet" "$BK/adsblol_points/"
      cp -p "$EXT/adsblol_ground_v2/$d.parquet" "$BK/adsblol_ground_v2/"
      cp -p "$EXT/adsblol_v2_cache/$d".{dep,arr}.parquet "$BK/adsblol_v2_cache/"
    done
    cp -p "$EXT/adsblol_ground/manifest.json" "$EXT/adsblol_ground_v2/stand_table.parquet" "$BK/"
    echo "[prep] backed up $(ls "$BK"/adsblol_ground | wc -l) days to $BK"
  fi
  # warm the release-list cache that the pull tasks share
  uv run python -c "
import sys; sys.path.insert(0, 'scripts')
from datetime import date
from adsb_pilot import preferred_urls
print('[prep] preferred release for 2025-01-01:', preferred_urls(date(2025, 1, 1))[0])"
  ;;
pull)
  i="${SLURM_ARRAY_TASK_ID:?pull runs as an array: --array=0-10}"
  a=$(uv run python -c "from datetime import date, timedelta as t; print(date.fromisoformat('$FIRST') + t(days=10*$i))")
  b=$(uv run python -c "from datetime import date, timedelta as t; print(min(date.fromisoformat('$FIRST') + t(days=10*$i+9), date.fromisoformat('$FIRST') + t(days=$NDAYS-1)))")
  export PRC_ADSB_TRANSIENT="$PWD/.adsb_tmp/task$i"
  mkdir -p "$PRC_ADSB_TRANSIENT"
  echo "[pull] task $i: $a .. $b, transient $PRC_ADSB_TRANSIENT"
  uv run python scripts/adsb_build.py run --source traces --days "$a" "$b" --workers 2
  rm -rf "$PRC_ADSB_TRANSIENT"
  ;;
v2)
  n=$(uv run python -c "
import json
m = json.load(open('$EXT/adsblol_ground/manifest.json'))['days']
from datetime import date, timedelta
ds = [str(date.fromisoformat('$FIRST') + timedelta(days=i)) for i in range($NDAYS)]
print(sum(m.get(d, {}).get('source') == 'traces' for d in ds))")
  echo "[v2] $n of $NDAYS days rebuilt from traces"
  [ "$n" -eq "$NDAYS" ] || { echo "ERROR: not all days rebuilt; rerun MODE=pull (it resumes)" >&2; exit 4; }
  for d in $(days_list); do rm -f "$EXT/adsblol_v2_cache/$d".{dep,arr}.parquet; done
  uv run python scripts/adsb_v2.py archive --keep-stand-table
  uv run prc features build adsb_ground adsb_ground_v2
  ;;
*) echo "ERROR: unknown MODE=$MODE" >&2; exit 2 ;;
esac
echo "[adsb-traces] $MODE done $(date -Is)"
