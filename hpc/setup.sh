#!/usr/bin/env bash
# One-time setup on a login node (needs internet). Safe to rerun.
#   git clone git@github.com:lvtwoodpecker/trailercut-vizml.git && cd trailercut-vizml
#   bash hpc/setup.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"

# uv: single static binary in ~/.local/bin
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

# venv on scratch; Linux torch wheels from PyPI already include CUDA
uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT"
VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install -e ".[hpc]"
# first call downloads the static ffmpeg binaries
command -v ffmpeg >/dev/null || $PY -c "import static_ffmpeg; static_ffmpeg.add_paths()"

# MovieNet annotations/meta/subtitles/shots (~670 MB) + unzip
$PY src/trailercut/movienet.py small --out "$TRAILERCUT_DATA/movienet"
for z in annotation subtitle1K movie1K.shot_detection; do
  unzip -qn "$TRAILERCUT_DATA/movienet/$z.v1.zip" -d "$TRAILERCUT_DATA/movienet"
done
# meta.v1.zip holds ~375k tiny files; only extract the films we use (inode quotas)
tail -n +2 data/ids/core_films_293.csv | cut -d, -f1 | sed 's|^|meta/|; s|$|.json|' \
  | xargs unzip -qn "$TRAILERCUT_DATA/movienet/meta.v1.zip" -d "$TRAILERCUT_DATA/movienet"

# Pre-download CLIP weights so GPU jobs don't depend on internet (h14 is ~4 GB)
$PY -c "import open_clip; from trailercut.embed import MODEL, PRETRAINED; open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)"
$PY -c "from transnetv2_pytorch import TransNetV2; TransNetV2(device='cpu')"

echo
echo "Setup done. Data in $TRAILERCUT_DATA"
echo "Check inode/space quota with 'myquota' (or your cluster's equivalent)."
