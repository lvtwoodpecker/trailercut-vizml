#!/bin/bash
# GPU job: embed only the hand-labeled films and their trailers with one backbone (h14 default),
# then grade alignment against the labels (trailercut.backbone_trial). Minutes on one GPU.
# Needs: setup.sh done, and those films' keyframes + trailer frames copied up (README, "Backbone trial").
#SBATCH --job-name=tc-trial
##SBATCH --account=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "CLIP backbone: $TRAILERCUT_CLIP"
export HF_HUB_OFFLINE=1

IDS=$($PY -c "from trailercut.label_eval import load_gold, one_per_shot; print(' '.join(sorted(one_per_shot(load_gold()).imdb_id.unique())))" | tail -1)
echo "films: $IDS"
$PY -m trailercut.embed film    --ids $IDS --bs 128 --workers 7
$PY -m trailercut.embed trailer --ids $IDS --bs 128 --workers 7
$PY -m trailercut.backbone_trial
