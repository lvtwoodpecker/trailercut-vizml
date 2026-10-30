#!/usr/bin/env bash
# Steps that change the inputs, run on purpose (not part of run_all.sh). After any of them, rebuild
# with: bash scripts/run_all.sh prep align audio rq1 viz (plus rq2 and ebm when needed)
#
#   bash scripts/curate.sh recover           replace dead trailer links (search + alignment check)
#   bash scripts/curate.sh audit             flag trailers that aren't the original theatrical trailer
#   bash scripts/curate.sh audit-review      page to confirm or overrule the audit's flags (labels/audit_review.html)
#   bash scripts/curate.sh swap ID [ID ...]  replace those films' trailers (keeps the old one if nothing passes)
#   bash scripts/curate.sh swap-prepare --set expand   first-choice trailers for new films (metadata only)
#   bash scripts/curate.sh swap-search       ranked replacement candidates for the audit's `swap` films
#   bash scripts/curate.sh swap-download     download them slowly, one at a time (resumable)
#   bash scripts/curate.sh swap-apply        check and swap in the downloaded candidates (resumable; --batch for hundreds)
#   bash scripts/curate.sh swap-review       page to check replaced trailers by eye (labels/swap_review.html)
#   (add --set expand to swap-prepare/-search/-download/-apply to work on new films instead of flagged core ones)
#   bash scripts/curate.sh split-propose     propose cuts inside glued trailer shots with CLIP (changes nothing)
#   bash scripts/curate.sh split-review      page to spot-check a sample of them (labels/split_review.html)
#   bash scripts/curate.sh split-report      share of real cuts at each similarity cutoff, from the verdicts
#   bash scripts/curate.sh split-apply --max-sim S   apply the cuts below the cutoff picked from the report
#   bash scripts/curate.sh labels            regenerate the hand-labeling page
#   bash scripts/curate.sh tune              compare alignment and model variants against the labels
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

case "${1:-}" in
  recover) $PY -m trailercut.recover ;;
  audit)   $PY -m trailercut.trailer_audit ;;
  audit-review) $PY -m trailercut.audit_review ;;
  swap)    shift; $PY -m trailercut.recover --ids "$@" ;;
  swap-prepare)  shift; $PY -m trailercut.swap prepare "$@" ;;
  swap-search)   shift; $PY -m trailercut.swap search "$@" ;;
  swap-download) shift; $PY -m trailercut.swap download "$@" ;;
  swap-apply)    shift; $PY -m trailercut.swap apply "$@" ;;
  swap-review)   $PY -m trailercut.swap_review ;;
  split-propose) shift; $PY -m trailercut.resplit_clip propose "$@" ;;
  split-review)  shift; $PY -m trailercut.resplit_clip review "$@" ;;
  split-report)  $PY -m trailercut.resplit_clip report ;;
  split-apply)   shift; $PY -m trailercut.resplit_clip apply "$@" ;;
  labels)  $PY -m trailercut.label_tool --n 10 --seed 0 ;;
  tune)    $PY -m trailercut.align_tune; $PY -m trailercut.rerank --eval; $PY -m trailercut.model_tune ;;
  *) sed -n '2,/^set /p' "$0" | grep '^#'; exit 1 ;;
esac
