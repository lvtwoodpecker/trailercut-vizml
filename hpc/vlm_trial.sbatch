#!/bin/bash
# GPU job: Qwen2.5-VL picks the matching film shot among CLIP's top 5 for every hand-labeled trailer
# shot, graded against the labels (trailercut.vlm_rerank). Needs the labeled films' keyframes and
# trailer frames on scratch (same upload as the backbone trial). Resumable: answers are cached.
#   sbatch --export=NONE --account=<account> hpc/vlm_trial.sbatch
#   sbatch --export=VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct --account=<account> hpc/vlm_trial.sbatch
#SBATCH --job-name=tc-vlm
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# the vlm extra (transformers) isn't in the base setup; installs into .venv once
export PATH=$HOME/.local/bin:$PATH
$PY -c "import transformers" 2>/dev/null || VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install -e ".[hpc,vlm]"

$PY -m trailercut.vlm_rerank --model "${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
echo "finished $(date)"
