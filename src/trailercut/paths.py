import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Large data can live outside the repo (e.g. HPC scratch): TRAILERCUT_DATA=/scratch/$USER/trailercut
DATA = Path(os.environ.get("TRAILERCUT_DATA", ROOT / "data"))
MOVIENET = DATA / "movienet"
FRAMES = MOVIENET / "frames"
TRAILERS = DATA / "trailers"
CACHE = DATA / "cache"
# Small, committed outputs always stay in the repo
FEATURES = ROOT / "data" / "features"
IDS = ROOT / "data" / "ids" / "core_films_293.csv"
LABELS = ROOT / "labels"
FIGURES = ROOT / "figures"

# CLIP backbone for every step that reads embeddings: b32 (default), l14, h14
CLIP_TAG = os.environ.get("TRAILERCUT_CLIP", "b32")
CLIP = CACHE / "clip" / CLIP_TAG

for p in (FEATURES, CACHE, LABELS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
