#!/bin/bash
# GPU job: Qwen2.5-VL as a same-shot verifier for CLIP's top 5 on the hand-labeled shots
# (trailercut.vlm_rerank): a 60-shot pilot first, then all 630 shots (answers are cached, so the full run
# reuses the pilot's). Needs the labeled films' keyframes, trailer frames and cache/trailer_frames_hr/ on scratch.
#   sbatch --export=VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct,VLM_FRAMES=3 --account=<account> hpc/vlm_pair.sbatch
#   sbatch --export=VLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct,VLM_FRAMES=3 --constraint=h200 --account=<account> hpc/vlm_pair.sbatch
#SBATCH --job-name=tc-vlmpair
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
export PATH=$HOME/.local/bin:$PATH
$PY -c "import transformers" 2>/dev/null || VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install -e ".[hpc,vlm]"

M=${VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}; F=${VLM_FRAMES:-3}
$PY -m trailercut.vlm_rerank --model "$M" --frames "$F" --pilot 60
$PY -m trailercut.vlm_rerank --model "$M" --frames "$F"
echo "finished $(date)"
