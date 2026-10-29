#!/bin/bash
# CPU + network job: MovieNet keyframes for all 1,100 films, one zip per film (~185 GB, ~1,100 files).
# Resumable: rerun after a timeout and it skips finished films.
#SBATCH --job-name=tc-keyframes
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh

# Compute nodes may have no internet. If this fails with connection errors, run the same
# command on a login node / data-transfer node inside tmux instead.
$PY src/trailercut/movienet.py keyframes --ids data/ids/films_1100.csv \
  --out "$TRAILERCUT_DATA/movienet" --pack
