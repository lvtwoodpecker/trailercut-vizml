#!/bin/bash
# GPU job: embed only the hand-labeled films and their trailers with one backbone (h14 default),
# then grade alignment against the labels (trailercut.backbone_trial). Minutes on one GPU.
# Needs: setup done, and those films' keyframes + trailer frames copied up (README, "Backbone trial").
#   sbatch --export=NONE --account=<slurm account> hpc/backbone_trial.sbatch
#SBATCH --job-name=tc-trial
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "CLIP backbone: $TRAILERCUT_CLIP"
export HF_HUB_OFFLINE=1

IDS=$($PY -c "from trailercut.label_eval import load_gold, one_per_shot; print(' '.join(sorted(one_per_shot(load_gold()).imdb_id.unique())))" | tail -1)
echo "films: $IDS"
$PY -m trailercut.embed film    --ids $IDS --bs 128 --workers 7
$PY -m trailercut.embed trailer --ids $IDS --bs 128 --workers 7
$PY -m trailercut.backbone_trial
echo "finished $(date)"
