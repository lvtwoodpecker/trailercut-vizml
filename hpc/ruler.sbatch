#!/bin/bash
# GPU job: grade one backbone (TRAILERCUT_CLIP) with rulers that don't come from B/32 (trailercut.ruler):
# its picks for every trailer shot of the hand-labeled films, and the synthetic test on their keyframes.
# Then grades every backbone whose files are there. Minutes on one GPU. Reuses the backbone trial's
# embeddings and upload (README, "Backbone trial").
#   sbatch --export=TRAILERCUT_CLIP=l14 --account=<slurm account> hpc/ruler.sbatch
#SBATCH --job-name=tc-ruler
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
echo "backbone: $TRAILERCUT_CLIP"
$PY -m trailercut.ruler cands
$PY -m trailercut.ruler synth
$PY -m trailercut.ruler grade
echo "finished $(date)"
