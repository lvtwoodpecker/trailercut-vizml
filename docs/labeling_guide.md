# Labeling guide

We hand-check the aligner on 10 trailers to pick the final match threshold and measure how often the
"best match" is the wrong shot. Takes about 30 to 45 min per trailer.

## Setup

1. Make sure you have the film keyframes and trailer frames for the sampled films locally (the page
   loads images from `data/`, nothing is uploaded).
2. `python -m trailercut.label_tool --n 10 --seed 0` (already run; the sample is in `labels/label_sample.json`).
3. Open `labels/label_tool.html` in a browser and type your name.

## What you see

One trailer shot at a time (its 3 keyframes), with the aligner's top-5 film shot candidates below it.

## Keys

| key | label | use when |
|---|---|---|
| `1` to `5` | that candidate is the source | it is the same shot: same framing, same moment (small crops, color grading or text on top are fine) |
| `s` | same scene, other shot | the right scene but none of the 5 is the exact shot (reverse angle, a different take of the moment) |
| `n` | not in film | title card, logo, footage that isn't in the film (alternate take, trailer-only VFX, another film) |
| `u` | unsure | you can't tell; these are left out of precision/recall |
| `j` / `k` | next / previous shot | |

Progress is kept in the browser while you work. **Export** when done and save the file as
`labels/gold_<yourname>.json`. If the browser storage gets cleared you lose unsaved progress, so
export after each trailer.

## Rules of thumb

- Match the *shot*, not the scene. If candidate 2 is the same scene but from the other side of the
  room, that's `s`, not `2`.
- Black frames, fades and pure text are `n` unless the film has the identical card.
- When two candidates look identical (neighboring shots of continuous coverage), pick the lower number.
- Don't look at the score while deciding.

## Agreement

The first 2 trailers in the sample are labeled by both of us. `label_eval` reports Cohen's kappa on
them. If kappa is under ~0.7, we go through the disagreements together and tighten these rules
before labeling the rest.

## After labeling

`python -m trailercut.label_eval` writes the precision/recall curve (`figures/alignment_pr.png`) and
`data/features/alignment_eval.json`. Pick the threshold, then rerun `align` and everything after it.
