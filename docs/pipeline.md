# Pipeline

How data flows from MovieNet and YouTube to the model, clusters and figures. `scripts/run_all.sh`
runs steps 1 to 17 in this order. The download and preprocessing steps (1 to 7) skip films and
trailers that are already done. From `align` on, each step recomputes everything, which takes minutes.

```
MovieNet (HF mirror)                    YouTube
  keyframes, shots, story,                trailer videos (360p)
  subtitles, meta                               │
        │                                 trailer_prep: TransNetV2 shots,
        │                                 3 keyframes/shot, mono audio
        │                                       │
        └──────── embed: CLIP per keyframe ─────┘
                           │
                  recover: replace dead links (search + alignment check)
                           │
                  align: trailer shot → film shot (or no match)
                     │                         │
        film_features: one row per      trailer_features: one row per
        film shot + in_trailer label    trailer (pacing, audio, story,
                     │                  dialogue, visual)
                  model (RQ1)                  │
                  model_eval (±k shots)        │
                                            cluster (RQ2)
                           │
                        figures
                           │
            viewer_export: per-trailer JSON for viz/ (Trailer View)
```

## Steps

| # | command | reads | writes | time (laptop, 293 trailers) |
|---|---|---|---|---|
| 1 | `movienet.py small`, then unzip | HF mirror | `data/movienet/*.zip` (annotations, meta, subtitles, shots) and video info JSON; `run_all.sh` unzips them | a few min |
| 2 | `movienet.py keyframes` | HF mirror (HTTP range requests) | `data/movienet/frames/{id}/` (JPEGs), or `frames/{id}.zip` with `--pack` | hours (network bound, ~125 MB/film) |
| 3 | `trailercut.trailers` | `data/ids/core_films_293.csv` | `data/trailers/{id}.mp4`, `data/ids/trailer_status.csv` | ~20 min |
| 4 | `trailercut.trailer_prep` | trailer videos | `data/features/trailer_shots/{id}.csv`, `cache/trailer_frames/`, `cache/trailer_audio/` | ~15 s/trailer CPU |
| 5 | `trailercut.embed film` | film keyframes | `cache/clip/<tag>/film/{id}.npy` [shots, 3, d] | ~1 h (B/32, MPS) |
| 6 | `trailercut.embed trailer` | trailer keyframes | `cache/clip/<tag>/trailer/{id}.npy` | ~3 min |
| 7 | `trailercut.concepts` | CLIP text encoder | `cache/concept_text_<tag>.npy` | seconds |
| 8 | `trailercut.recover` | status log, MovieNet meta, film embeddings, existing `alignment_threshold.json` | replacement trailers (runs `trailer_prep` and `embed trailer` itself), `trailer_recovery.csv`, updated status | ~30 min (approx.) |
| 9 | `trailercut.align` | embeddings, trailer shot tables, MovieNet shots | `data/features/alignment/{id}.csv`, `alignment_threshold.json`, `cache/alignment_null_<tag>.npy` | ~1 min |
| 10 | `trailercut.label_tool` | alignment | `labels/label_tool.html`, `labels/label_sample.json`, `labels/img/` (copied keyframes) | seconds |
| 11 | `trailercut.label_eval` | `labels/gold_*.json`, alignment, `alignment_threshold.json` | `alignment_eval.json`, `figures/alignment_pr.png` | seconds |
| 12 | `trailercut.film_features` | alignment, embeddings, subtitles, story | `data/features/film_shots.parquet` | ~1 min |
| 13 | `trailercut.model` | film shot table, film embeddings | `model_report.json`, `model_preds.parquet`, `model_shap.parquet` | ~15 min (approx., not timed yet) |
| 14 | `trailercut.model_eval` | `model_preds.parquet` | `model_eval.json` | seconds |
| 15 | `trailercut.trailer_features` | alignment, trailer shots, audio, subtitles, trailer embeddings, meta | `trailers.parquet`, `trailer_curves.json` | ~1 min |
| 16 | `trailercut.cluster` | trailer table | `clusters.csv`, `cluster_report.json`, `figures/clusters_all_umap.png` | ~1 min |
| 17 | `trailercut.figures` | alignment, status log, trailer shots, impostor null cache | `figures/{data_funnel,alignment_scores,alignment_examples,story_depth,corpus}.png`, `summary.json` | seconds |
| 18 | `trailercut.viewer_export` | alignment, `film_shots.parquet`, `trailer_curves.json`, model preds/SHAP if present, keyframes | `viz/data/index.json`, `viz/data/{id}.json`, `cache/shot_colors/{id}.json` | seconds once colors are cached; first color pass ~1 ms per keyframe |

`cache/` is `$TRAILERCUT_DATA/cache` (default `data/cache`). `<tag>` is the CLIP backbone
(`b32`, `l14`, `h14`). Steps 2, 4, 5 and 6 are the heavy ones and have Slurm jobs in `hpc/` (the
cluster job uses `--pack`; `run_all.sh` doesn't). The label tool copies its images into `labels/img/`,
so it works with either. The Trailer View's frame thumbnails load straight from `data/movienet/frames/`
(loose JPEGs) and `data/cache/trailer_frames/`, so they only show on a machine with that data in the
default location.

`recover` reads the threshold from the committed `alignment_threshold.json`, so it runs before `align`
with the previous run's threshold; `align` then recomputes it on the full set.
`align --threshold T` applies a fixed cutoff instead (e.g. the one picked from hand labels); the null is
still computed and its false-match rate at `T` is saved as `fpr`.

## Configuration

| env var | default | effect |
|---|---|---|
| `TRAILERCUT_DATA` | `data/` | root for MovieNet files, trailers and caches (point it at scratch on the cluster) |
| `TRAILERCUT_CLIP` | `b32` | CLIP backbone; caches are keyed by it so backbones never mix |

Features, figures and labels always live in the repo checkout, not under `TRAILERCUT_DATA`.

## Rerunning after a change

| you changed | rerun from |
|---|---|
| trailer list or recovered trailers | `trailer_prep` (skips done ones), `embed trailer`, then `align` onward |
| CLIP backbone | `embed film`, `embed trailer`, `concepts`, then `align` onward |
| match threshold or alignment logic | `align` (`--threshold T` for a hand-picked cutoff) onward |
| film shot features | `film_features`, `model`, `model_eval`, `viewer_export` |
| trailer features | `trailer_features`, `cluster`, `viewer_export` (it reads `trailer_curves.json`) |
| model | `model`, `model_eval`, `viewer_export` |
| tolerance eval only | `model_eval` (reuses saved predictions, no retraining) |
| the Trailer View page only | nothing; reload `viz/index.html` |

"`align` onward" includes `viewer_export`.

Alignment outputs are named by IMDb ID only, so switching backbones overwrites them. Commit or copy
`data/features/alignment/` first if you need both.
