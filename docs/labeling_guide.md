# Labeling guide

We hand-check the aligner on 10 trailers to pick the final match threshold and measure how often the
"best match" is the wrong shot. Budget roughly 30 to 45 min per trailer (a guess, not measured yet).

## Setup

1. Make sure you have the film keyframes (loose JPEGs or `--pack` zips) and trailer frames for the
   sampled films locally, under `TRAILERCUT_DATA` if set. `label_tool` copies the images it needs into
   `labels/img/` (gitignored); the page loads them from there. Nothing is uploaded.
2. `python -m trailercut.label_tool --n 10 --seed 0` (already run; the 10 sampled IMDb IDs are in
   `labels/label_sample.json`). Sampling is round-robin over decades. Same seed and same set of
   aligned trailers = same sample, so everyone labels from the same page (the 293-trailer set gives the
   same 10 films as the earlier 244 set).
3. Open `labels/label_tool.html` in a browser and type your name in the **Labeler** box at the top.
   It is only read at export time and becomes the file name. Always fill it: an empty box exports
   `gold_anon.json`, which `label_eval` treats as a separate labeler: it can become one of the two
   kappa labelers, and on overlap the alphabetically first file wins.

## What you see

One trailer shot at a time (its 3 keyframes), with the aligner's top-5 film shot candidates next to
it (one keyframe each, with its score). Shots are listed trailer by trailer, in shot order.

## Keys

| key | label | use when |
|---|---|---|
| `1` to `5` | that candidate is the source | it is the same shot: same framing, same moment (small crops, color grading or text on top are fine) |
| `s` | same scene, other shot | the right scene but none of the 5 is the exact shot (reverse angle, a different take of the moment) |
| `n` | not in film | title card, logo, footage that isn't in the film (alternate take, trailer-only VFX, another film) |
| `u` | unsure | you can't tell |
| `j` / `k` | next / previous shot | |

Clicking a candidate or a button does the same thing. Any label jumps to the next shot; relabeling
a shot overwrites the old label. Keys are ignored while the cursor is in the name box.

Labels are saved in the browser's `localStorage` (one shared slot for the page, not per name) as
you go. **Export JSON** downloads `gold_<name>.json` (`gold_anon.json` if the name box is empty)
with `{labeler, labels}`, where `labels` maps `imdb_id:shot_idx` to a film shot index or
`s`/`n`/`u`. Move it to `labels/gold_<name>.json`. Export after each trailer: clearing site data
loses everything not exported. If two people share one browser, the second person starts on top of
the first person's labels, so use separate browsers or profiles.

## Rules of thumb

- Match the *shot*, not the scene. If candidate 2 is the same scene but from the other side of the
  room, that's `s`, not `2`.
- Black frames, fades and pure text are `n` unless the film has the identical card.
- When two candidates look identical (neighboring shots of continuous coverage), pick the lower number.
- Don't look at the score while deciding.

## Agreement

Team convention (the tool doesn't enforce it): both of us label the first 2 trailers in the sample.
`label_eval` computes Cohen's kappa over every shot both labelers labeled (the first two
`gold_*.json` files by name), on the raw labels: candidate shot index, `s`, `n` or `u`. If kappa is
under ~0.7, we go through the disagreements together and tighten these rules before labeling the rest.

## How labels are scored

`label_eval` keeps one label per shot (on overlap, the alphabetically first file wins), then:

- `u` shots are dropped.
- A shot has a source if it was labeled with a candidate. `top1_right` = that candidate is the
  aligner's best match.
- For each threshold from 0.4 to 0.95 (steps of 0.005), a shot is predicted matched if its score clears it and it
  isn't blank. Precision = share of predicted-matched shots with `top1_right`; recall = predicted
  matched and `top1_right` over all shots with a source.
- `s` and `n` have no source: they never count toward recall, and count as errors for precision when
  predicted matched. `s` shows up only in `label_mix`; it isn't scored as partly right.

## After labeling

`python -m trailercut.label_eval` writes `figures/alignment_pr.png` and
`data/features/alignment_eval.json` (`kappa`, `kappa_n`, `n`, `label_mix`, `top1_acc_when_source`,
`best_f1`, `precision_0.9` = lowest threshold with precision at least 0.9, `impostor_threshold`).
`scripts/run_all.sh` then re-runs `align --threshold` with `best_f1.threshold` and everything after
it. `python -m trailercut.align_tune` scores raw vs smoothed matching against the same labels
(`align_tune.json`). Current result: best F1 at 0.535 (see `results.md`).
