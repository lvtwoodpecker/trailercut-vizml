#!/bin/bash
# GPU job: Whisper large-v3 transcripts of every trailer into cache/trailer_asr_large-v3/ (the laptop's
# Whisper small ones stay in cache/trailer_asr/). Needs cache/trailer_audio/ copied up. Resumable.
# Transcripts are copyrighted text: they stay on scratch and get rsynced down, never committed.
#   sbatch --export=NONE --account=<account> hpc/asr.sbatch
#SBATCH --job-name=tc-asr
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
# faster-whisper (CTranslate2) looks for cuBLAS / cuDNN on the library path; use the ones torch installed
export LD_LIBRARY_PATH=$($PY -c "import nvidia.cublas.lib as b, nvidia.cudnn.lib as c; print(b.__path__[0] + ':' + c.__path__[0])"):${LD_LIBRARY_PATH:-}

$PY -m trailercut.trailer_asr --model large-v3
ls data/cache/trailer_asr_large-v3 | wc -l
echo "finished $(date)"
