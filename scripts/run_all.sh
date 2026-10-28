#!/usr/bin/env bash
# Full pipeline, in order. Download, shot and embedding steps skip finished films/trailers;
# align onward recomputes everything (each takes a minute or so, except the model).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# 1. data (raw files stay in data/, gitignored)
$PY src/trailercut/movienet.py small --out data/movienet
for z in annotation meta subtitle1K movie1K.shot_detection; do unzip -qn data/movienet/$z.v1.zip -d data/movienet; done
$PY src/trailercut/movienet.py keyframes --ids data/ids/core_films_293.csv --out data/movienet
$PY -m trailercut.trailers                   # -> data/ids/trailer_status.csv

# 2. trailer shots, keyframes, audio; CLIP embeddings
$PY -m trailercut.trailer_prep
$PY -m trailercut.embed film
$PY -m trailercut.embed trailer
$PY -m trailercut.concepts

# 3. alignment + hand-label tool
$PY -m trailercut.recover                    # dead links: search, keep only if >=25% of shots align
$PY -m trailercut.align
$PY -m trailercut.label_tool --n 10 --seed 0
$PY -m trailercut.label_eval                 # needs labels/gold_*.json

# 4. features, clustering, figures
$PY -m trailercut.film_features
$PY -m trailercut.model                      # RQ1: grouped CV, baselines, ablations, SHAP
$PY -m trailercut.model_eval                 # same predictions scored with ±k shot tolerance
$PY -m trailercut.trailer_features
$PY -m trailercut.cluster
$PY -m trailercut.figures

# 5. data for the D3 views in viz/
$PY -m trailercut.viewer_export
