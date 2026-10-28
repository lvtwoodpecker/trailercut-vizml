# What Makes The Cut?

Interpretable shot selection and clustering of movie trailers. DS-GA 3001 Data Visualization for ML, Team 13 (Cedric Lam, Estee Ng).

We match every trailer shot back to its source shot in the film (MovieNet), then ask
1. **RQ1** which film shots make it into the trailer, and how deep into the story trailers reach, using readable features so the explanations mean something;
2. **RQ2** whether trailers fall into groups by pacing, audio, story sampling and dialogue, and whether those groups are just genre/era.

Builds on Argaw et al., *Towards Automated Movie Trailer Generation* (CVPR 2024). Proposal: [`docs/proposal_team13.pdf`](docs/proposal_team13.pdf).

## Data policy

Nothing copyrighted is committed. The repo holds YouTube IDs, download scripts and extracted features;
raw MovieNet archives, keyframes, trailer videos and CLIP embeddings live in gitignored `data/` folders and are rebuilt by the scripts.

| committed | what |
|---|---|
| `data/ids/core_films_293.csv` | the 293 films with keyframes + synopsis-to-shot story alignment + a trailer URL |
| `data/ids/trailer_status.csv` | which trailers still download (ok / unavailable / error); `source` says whether the link is MovieNet's or a search replacement |
| `data/ids/trailer_recovery.csv` | every search candidate tried for a dead link, with its match share and whether it was accepted |
| `data/features/trailer_shots/*.csv` | TransNetV2 shots per trailer |
| `data/features/alignment/*.csv` | per trailer shot: matched film shot, score, margin, top-5 candidates |
| `data/features/film_shots.parquet` | RQ1 table: one row per film shot, readable features + `in_trailer` label (not committed, ~65 MB; `python -m trailercut.film_features` rebuilds it) |
| `data/features/model_report.json` | RQ1 metrics per model (baselines, readable, full, ablations) and SHAP summaries; per-shot predictions and SHAP (`model_preds.parquet`, `model_shap.parquet`) are not committed |
| `data/features/trailers.parquet` | RQ2 table: one row per trailer |
| `data/features/clusters.csv`, `cluster_report.json` | cluster labels, stability, genre/decade association |
| `labels/` | hand-labeling sample + gold labels for alignment (the generated page and `labels/img/` are not committed) |
| `viz/index.html` | the D3 Trailer View; its data (`viz/data/`) is generated and not committed |

## Setup

```bash
uv venv --python 3.12 && uv pip install -e .
brew install ffmpeg   # needed for trailer decoding
```

Shot detection and embeddings are the slow parts; [`hpc/`](hpc/README.md) runs them on a Slurm GPU
cluster. `TRAILERCUT_DATA` moves raw data and caches (e.g. to scratch) and `TRAILERCUT_CLIP` picks the
CLIP backbone (`b32` default, `l14`, `h14`).

## Pipeline

`scripts/run_all.sh` runs everything in order. Download, shot and embedding steps skip
finished films and trailers; from `align` on, each step recomputes everything.

| step | module | notes |
|---|---|---|
| MovieNet download | `src/trailercut/movienet.py` | ungated HF mirror; keyframes pulled per film with HTTP range requests (~125 MB/film instead of the 172 GB archive) |
| trailers | `trailercut.trailers` | yt-dlp at 360p, logs availability |
| dead links | `trailercut.recover` | YouTube search for replacements; accepted only if at least 25% of shots align to the film |
| shots | `trailercut.trailer_prep` | TransNetV2 boundaries, 3 keyframes/shot (MovieNet style), mono audio |
| embeddings | `trailercut.embed` | CLIP ViT-B/32 (LAION-2B), 3 keyframes per shot; trailer letterbox bars cropped per trailer |
| alignment | `trailercut.align` | mean-of-max similarity over keyframes; blank shots masked; "no match" threshold from an impostor null |
| label check | `trailercut.label_tool`, `label_eval` | hand-check top-5 candidates, then pick the threshold from precision/recall |
| readable features | `trailercut.concepts` | zero-shot shot scale + 20 content concepts, so SHAP names things people can read |
| RQ1 table | `trailercut.film_features` | position, dialogue (subtitles + VADER), scale, content, editing |
| RQ1 model | `trailercut.model` | LightGBM, folds grouped by film, baselines + ablations, out-of-fold TreeSHAP |
| RQ2 table | `trailercut.trailer_features` | pacing, audio (librosa), story sampling, dialogue, visual concepts |
| clustering | `trailercut.cluster` | k-means + HDBSCAN, bootstrap stability, NMI vs genre/decade against a permutation null |
| figures | `trailercut.figures` | data funnel, score distributions, alignment plots, story depth |
| Trailer View data | `trailercut.viewer_export` | one JSON per trailer in `viz/data/` (gitignored) plus average shot colors for the movie barcode |

## Hand-labeling alignments

```bash
.venv/bin/python -m trailercut.label_tool --n 10   # writes labels/label_tool.html
open labels/label_tool.html
```
Keys: `1`-`5` pick a candidate, `s` same scene but other shot, `n` not in film, `u` unsure, `j`/`k` next/prev.
Export and save as `labels/gold_<name>.json`, then run `python -m trailercut.label_eval`.

## Trailer View

D3 page for one trailer at a time: the trailer's cuts on top, the film as a movie barcode (average
color per shot) below, and a curve from each matched trailer shot to where it sits in the film. Hover
to preview, click to lock a shot and see trailer and film frames side by side, arrow keys step
through matched shots. Below that, an order scatter (trailer position vs film position) and the
grouped SHAP for the selected film shot (click a group to see its features), when the model has been run.

```bash
python -m trailercut.viewer_export   # writes viz/data/
```
Then serve the repo root with any static server, e.g. `python -m http.server [port]`, and open
`/viz/` (add `?id=<imdb_id>` for a film).
Frame thumbnails load from your local `data/movienet/frames/` (loose JPEGs) and
`data/cache/trailer_frames/`, so they only show on a machine that has the data. Nothing copyrighted is
committed or hosted.
