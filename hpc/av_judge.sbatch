#!/bin/bash
# GPU job array: Qwen2.5-Omni rates each trailer's 30 s pieces for emotions, from picture and sound
# (trailercut.av_judge). Needs data/trailers/*.mp4 on scratch (rsync, hpc/README.md step 2).
# Each array task takes every 4th trailer and saves per trailer, so a requeue resumes.
#   pilot (5 trailers, check the answers parse):
#     sbatch --export=NONE,JUDGE_LIMIT=5 --array=0 --account=<account> hpc/av_judge.sbatch
#   full run, picture and sound:
#     sbatch --export=NONE --array=0-3 --account=<account> hpc/av_judge.sbatch
#   picture only (what the sound adds), on 200 trailers:
#     sbatch --export=NONE,JUDGE_MODE=video,JUDGE_LIMIT=50 --array=0-3 --account=<account> hpc/av_judge.sbatch
#SBATCH --job-name=tc-avjudge
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
echo "job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
export PATH=$HOME/.local/bin:$PATH
$PY -c "import qwen_omni_utils, transformers" 2>/dev/null || VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install -e ".[hpc,judge]"

MODE=${JUDGE_MODE:-av}; LIMIT=${JUDGE_LIMIT:-0}
$PY -m trailercut.av_judge --mode "$MODE" --shard "$SLURM_ARRAY_TASK_ID" --shards 4 --limit "$LIMIT"
$PY -m trailercut.av_judge --mode "$MODE" --collect
echo "finished $(date)"
