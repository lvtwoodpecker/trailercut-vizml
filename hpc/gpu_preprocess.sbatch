#!/bin/bash
# GPU job: TransNetV2 shots for every trailer, then CLIP embeddings for film + trailer keyframes.
# Needs: trailers rsynced to $TRAILERCUT_DATA/trailers and fetch_keyframes.sbatch finished.
# Resumable: every step skips trailers/films that are already done.
#   sbatch --export=NONE --account=<account> hpc/gpu_preprocess.sbatch
#SBATCH --job-name=tc-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "CLIP backbone: $TRAILERCUT_CLIP"

# offline: weights were cached by setup.sh
export HF_HUB_OFFLINE=1

$PY -m trailercut.trailer_prep
$PY -m trailercut.embed film    --bs 256 --workers 7
$PY -m trailercut.embed trailer --bs 256 --workers 7
$PY -m trailercut.concepts
$PY -m trailercut.align

ls "$TRAILERCUT_DATA/cache/clip/$TRAILERCUT_CLIP/film" | wc -l
ls "$TRAILERCUT_DATA/cache/clip/$TRAILERCUT_CLIP/trailer" | wc -l
