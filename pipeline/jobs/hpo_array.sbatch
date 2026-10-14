#!/bin/bash
#SBATCH --job-name=hpo
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=240G
#SBATCH --gpus-per-node=1
#SBATCH --account=p65425
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
# task id -> trial (id // 2), outer month (1 or 7)
T=$(( SLURM_ARRAY_TASK_ID / 2 )); O=$(( (SLURM_ARRAY_TASK_ID % 2) * 6 + 1 ))
export CFG_JSON=$(python3 -c "import json;print(json.dumps(json.load(open('pipeline/hpo_trials.json'))[$T]))")
S1TAG=v2 FM=work/fm2.parquet bash pipeline/jobs/gpu.sbatch pipeline/gbm_nested.py $O --tag hpo_t$(printf %02d $T) --only-cat --cfg catmixd_json
