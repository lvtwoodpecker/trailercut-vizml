# Source this in every job on the cluster:  source hpc/env.sh
# Everything lives inside the repo checkout, which sits on scratch (clone it under $SCRATCH, not $HOME):
# data/ (MovieNet, keyframes, trailers, embeddings), .venv/, .cache/ (model weights, package caches, temp).
# All of it is gitignored.

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export TRAILERCUT_DATA=$REPO/data
export TRAILERCUT_CLIP=${TRAILERCUT_CLIP:-h14}          # b32 | l14 | h14 (h14 matches Argaw et al.)

# Model weights and package caches off the small home quota
export HF_HOME=$REPO/.cache/hf
export TORCH_HOME=$REPO/.cache/torch
export UV_CACHE_DIR=$REPO/.cache/uv
export UV_PROJECT_ENVIRONMENT=$REPO/.venv
# temp files on scratch: /tmp on the login node is not the compute node's /tmp
export TMPDIR=$REPO/.cache/tmp

PY=$UV_PROJECT_ENVIRONMENT/bin/python
mkdir -p "$TRAILERCUT_DATA" "$HF_HOME" "$TORCH_HOME" "$TMPDIR" "$REPO/logs"
