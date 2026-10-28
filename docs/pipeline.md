# Pipeline

How data flows from MovieNet and YouTube to the model, clusters and figures. `scripts/run_all.sh`
runs every step in this order. Every step skips work that is already done, so rerunning is cheap.

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
                                            cluster (RQ2)
                           │
                        figures
```

## Steps

| # | command | reads | writes | time (laptop, 244 trailers) |
|---|---|---|---|---|
| 1 | `movienet.py small` | HF mirror | `data/movienet/*.zip` (annotations, meta, subtitles, shots) | a few min |
| 2 | `movienet.py keyframes --pack` | HF mirror (HTTP range requests) | `data/movienet/frames/{id}.zip` | hours (network bound, ~125 MB/film) |
| 3 | `trailercut.trailers` | `data/ids/core_films_293.csv` | `data/trailers/{id}.mp4`, `data/ids/trailer_status.csv` | ~20 min |
| 4 | `trailercut.trailer_prep` | trailer videos | `data/features/trailer_shots/{id}.csv`, `cache/trailer_frames/`, `cache/trailer_audio/` | ~15 s/trailer CPU |
| 5 | `trailercut.embed film` | film keyframes | `cache/clip/<tag>/film/{id}.npy` [shots, 3, d] | ~1 h (B/32, MPS) |
| 6 | `trailercut.embed trailer` | trailer keyframes | `cache/clip/<tag>/trailer/{id}.npy` | ~3 min |
| 7 | `trailercut.concepts` | CLIP text encoder | `cache/concept_text_<tag>.npy` | seconds |
| 8 | `trailercut.recover` | status log, embeddings | replacement trailers, `trailer_recovery.csv`, updated status | ~30 min |
| 9 | `trailercut.align` | embeddings, shot tables | `data/features/alignment/{id}.csv`, `alignment_threshold.json` | ~1 min |
| 10 | `trailercut.label_tool` | alignment | `labels/label_tool.html`, `labels/label_sample.json` | seconds |
| 11 | `trailercut.label_eval` | `labels/gold_*.json` | `alignment_eval.json`, `figures/alignment_pr.png` | seconds |
| 12 | `trailercut.film_features` | alignment, embeddings, subtitles, story | `data/features/film_shots.parquet` | ~1 min |
| 13 | `trailercut.model` | film shot table, film embeddings | `model_report.json`, `model_preds.parquet`, `model_shap.parquet` | ~15 min |
| 14 | `trailercut.trailer_features` | alignment, shots, audio, subtitles | `trailers.parquet`, `trailer_curves.json` | ~1 min |
| 15 | `trailercut.cluster` | trailer table | `clusters.csv`, `cluster_report.json`, `figures/clusters_all_umap.png` | ~1 min |
| 16 | `trailercut.figures` | everything above | `figures/*.png`, `summary.json` | seconds |

`cache/` is `$TRAILERCUT_DATA/cache` (default `data/cache`). `<tag>` is the CLIP backbone
(`b32`, `l14`, `h14`). Steps 2, 4, 5 and 6 are the heavy ones and have Slurm jobs in `hpc/`.

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
| match threshold or alignment logic | `align` onward |
| film shot features | `film_features`, `model` |
| trailer features | `trailer_features`, `cluster`, `figures` |

Alignment outputs are named by IMDb ID only, so switching backbones overwrites them. Commit or copy
`data/features/alignment/` first if you need both.
