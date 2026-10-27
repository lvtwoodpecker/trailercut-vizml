# Source this in every shell and job on the cluster:  source hpc/env.sh
# Edit SCRATCH_ROOT if your cluster's scratch lives somewhere else.

SCRATCH_ROOT=${SCRATCH:-/scratch/$USER}

export TRAILERCUT_DATA=$SCRATCH_ROOT/trailercut/data   # MovieNet, keyframes, trailers, embeddings
export TRAILERCUT_CLIP=${TRAILERCUT_CLIP:-h14}          # b32 | l14 | h14 (h14 matches Argaw et al.)

# Keep model weights and package caches off the small home quota
export HF_HOME=$SCRATCH_ROOT/trailercut/hf
export TORCH_HOME=$SCRATCH_ROOT/trailercut/torch
export UV_CACHE_DIR=$SCRATCH_ROOT/trailercut/uv-cache
export UV_PROJECT_ENVIRONMENT=$SCRATCH_ROOT/trailercut/venv

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=$UV_PROJECT_ENVIRONMENT/bin/python
mkdir -p "$TRAILERCUT_DATA" "$HF_HOME" "$TORCH_HOME"
