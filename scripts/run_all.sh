#!/usr/bin/env bash
# The pipeline's stages in their recommended order. Nothing runs unless you name it:
#   bash scripts/run_all.sh              list the stages
#   bash scripts/run_all.sh rq1 viz      run only these, in the order given
#
#   data   MovieNet files, keyframes, trailer videos           (network; hours the first time)
#   prep   trailer shots, keyframes, audio; CLIP embeddings    (hours the first time, then only new items)
#   align  trailer shot -> film shot, cutoff from hand labels  (minutes)
#   audio  Whisper transcripts, dialogue alignment, sync       (minutes per new trailer)
#   rq1    trailerness model, evaluation, figures              (~30 min)
#   viz    data for the D3 pages                                (minutes)
#   rq2    trailer features, clustering, case studies          (RQ2; minutes)
#   ebm    glass-box model for the report's curves             (~30 min)
# Download, shot, embedding and transcription steps skip finished films/trailers; the rest recompute.
# Changing the inputs (dead links, trailer swaps, labels) is separate: see scripts/curate.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

data() {   # raw files stay in data/, gitignored
  $PY src/trailercut/movienet.py small --out data/movienet
  for z in annotation meta subtitle1K movie1K.shot_detection; do unzip -qn data/movienet/$z.v1.zip -d data/movienet; done
  $PY src/trailercut/movienet.py keyframes --ids data/ids/films_1100.csv --out data/movienet
  $PY -m trailercut.trailers                 # -> data/ids/trailer_status.csv
}

prep() {   # trailer shots, keyframes, audio; CLIP embeddings
  $PY -m trailercut.trailer_prep
  $PY -m trailercut.embed film
  $PY -m trailercut.embed trailer
  $PY -m trailercut.concepts
}

align() {  # picture alignment, with the cutoff picked from the hand labels once they exist
  $PY -m trailercut.align
  if ls labels/gold_*.json >/dev/null 2>&1; then
    $PY -m trailercut.label_eval
    $PY -m trailercut.align --threshold "$($PY -c 'import json; print(json.load(open("data/features/alignment_eval.json"))["best_f1"]["threshold"])')"
  fi
  $PY -m trailercut.align_diagnostics
}

audio() {  # transcripts, dialogue alignment, sound vs picture
  $PY -m trailercut.trailer_asr
  $PY -m trailercut.dialogue_align
  $PY -m trailercut.av_sync
}

rq1() {    # which film shots make the trailer
  $PY -m trailercut.film_features
  $PY -m trailercut.model
  $PY -m trailercut.model_eval
  $PY -m trailercut.model_figures
  $PY -m trailercut.figures                # data funnel, alignment and story figures
}

rq2() {    # trailer features and clusters (opt-in)
  $PY -m trailercut.trailer_features
  $PY -m trailercut.cluster
  $PY -m trailercut.case_studies
}

viz() {    # data for the D3 pages in viz/
  $PY -m trailercut.viewer_export
  $PY -m trailercut.rq1_export
}

ebm() {    # glass-box model for the report's curves
  $PY -m trailercut.model_ebm
}

stages=("$@")
if [ ${#stages[@]} -eq 0 ]; then sed -n '2,/^set /p' "$0" | grep '^#'; exit 0; fi
for s in "${stages[@]}"; do
  echo "== $s"
  "$s"
done
