#!/bin/bash
# One-time setup as a job (no installs or downloads on the login node): runs hpc/setup.sh.
# Needs internet on the compute node. Safe to rerun.
#   sbatch --export=NONE --account=<slurm account> hpc/setup.sbatch
#SBATCH --job-name=tc-setup
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash hpc/setup.sh
