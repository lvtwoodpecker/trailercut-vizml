#!/usr/bin/env bash
# Steps that change the inputs, run on purpose (not part of run_all.sh). After any of them, rebuild
# with: bash scripts/run_all.sh prep align audio rq1 rq2 viz
#
#   bash scripts/curate.sh recover           replace dead trailer links (search + alignment check)
#   bash scripts/curate.sh audit             flag trailers that aren't the original theatrical trailer
#   bash scripts/curate.sh swap ID [ID ...]  replace those films' trailers (keeps the old one if nothing passes)
#   bash scripts/curate.sh swap-prepare --set expand   first-choice trailers for new films (metadata only)
#   bash scripts/curate.sh swap-search       ranked replacement candidates for the audit's `swap` films
#   bash scripts/curate.sh swap-download     download them slowly, one at a time (resumable)
#   bash scripts/curate.sh swap-apply        check and swap in the downloaded candidates (resumable)
#   bash scripts/curate.sh swap-sheet        contact sheets of the swaps to skim (cache/sheets/)
#   (add --set expand to the swap-* steps to work on new films instead of flagged core ones)
#   bash scripts/curate.sh labels            regenerate the hand-labeling page
#   bash scripts/curate.sh tune              compare alignment and model variants against the labels
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

case "${1:-}" in
  recover) $PY -m trailercut.recover ;;
  audit)   $PY -m trailercut.trailer_audit ;;
  swap)    shift; $PY -m trailercut.recover --ids "$@" ;;
  swap-prepare)  shift; $PY -m trailercut.swap prepare "$@" ;;
  swap-search)   shift; $PY -m trailercut.swap search "$@" ;;
  swap-download) shift; $PY -m trailercut.swap download "$@" ;;
  swap-apply)    shift; $PY -m trailercut.swap apply "$@" ;;
  swap-sheet)    shift; $PY -m trailercut.swap sheet "$@" ;;
  labels)  $PY -m trailercut.label_tool --n 10 --seed 0 ;;
  tune)    $PY -m trailercut.align_tune; $PY -m trailercut.rerank --eval; $PY -m trailercut.model_tune ;;
  *) sed -n '2,/^set /p' "$0" | grep '^#'; exit 1 ;;
esac
