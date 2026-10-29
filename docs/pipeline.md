# Pipeline

How data flows from MovieNet and YouTube to the model, clusters and figures. Two scripts:

- `scripts/run_all.sh [stage ...]` rebuilds every output from the inputs, one stage after another:
  `data prep align audio rq1 rq2 viz` by default, or only the stages named, in the order given.
  `ebm` is opt-in (slow) and not in the default run. Download, shot, embedding and transcription steps
  skip finished films and trailers; the rest recompute, which takes minutes.
- `scripts/curate.sh <step>` holds the steps that change the inputs (trailer links, labels) and are
  run on purpose: `recover`, `audit`, `swap`, `swap-search`, `swap-download`, `swap-apply`, `labels`,
  `tune`. After any of them, rebuild with `bash scripts/run_all.sh prep align audio rq1 rq2 viz`.

```
MovieNet (HF mirror)                    YouTube
  keyframes, shots, story,                trailer videos (360p)
  subtitles, meta                               │
        │                                 trailer_prep: TransNetV2 shots,
        │                                 3 keyframes/shot, mono audio
        │                                       │
        └──────── embed: CLIP per keyframe ─────┘
                           │
         curate.sh (on purpose): recover dead links, audit, swap trailers
                           │
                  align: trailer shot → film shot (or no match)
                  label_tool → hand labels → label_eval
                  align --threshold <best F1 from the labels>
                  align_diagnostics
                     │                         │
        film_features: one row per      trailer_features: one row per
        film shot + in_trailer label    trailer (pacing, audio, story,
                     │                  dialogue, visual)
                  model (RQ1)           trailer_asr: Whisper transcripts
                  model_eval (±k shots) dialogue_align: lines → film subtitles
                  model_figures         av_sync: speech features, sync rate
                  model_ebm (opt-in)           │
                                            cluster (RQ2)
                           │
                  figures, case_studies
                           │
            viewer_export: per-trailer JSON for viz/ (Trailer View)
            rq1_export: lifts across all films for viz/rq1.html
```

## Steps (`run_all.sh`)

| stage | command | reads | writes | time (laptop, 293 trailers) |
|---|---|---|---|---|
| data | `movienet.py small`, then unzip | HF mirror | `data/movienet/*.zip` (annotations, meta, subtitles, shots) and video info JSON; `run_all.sh` unzips them | a few min |
| data | `movienet.py keyframes` | HF mirror (HTTP range requests) | `data/movienet/frames/{id}/` (JPEGs), or `frames/{id}.zip` with `--pack` | hours (network bound, ~125 MB/film) |
| data | `trailercut.trailers` | `data/ids/core_films_293.csv` | `data/trailers/{id}.mp4`, `data/ids/trailer_status.csv` | ~20 min |
| prep | `trailercut.trailer_prep` | trailer videos | `data/features/trailer_shots/{id}.csv`, `cache/trailer_frames/`, `cache/trailer_audio/` | ~15 s/trailer CPU |
| prep | `trailercut.embed film` | film keyframes | `cache/clip/<tag>/film/{id}.npy` [shots, 3, d] | ~1 h (B/32, MPS) |
| prep | `trailercut.embed trailer` | trailer keyframes | `cache/clip/<tag>/trailer/{id}.npy` | ~3 min |
| prep | `trailercut.concepts` | CLIP text encoder | `cache/concept_text_<tag>.npy` | seconds |
| align | `trailercut.align` | embeddings, trailer shot tables, MovieNet shots | `data/features/alignment/{id}.csv`, `alignment_threshold.json`, `cache/alignment_null_<tag>.npy` | ~1 min |
| align | `trailercut.label_eval` (only if `labels/gold_*.json` exist) | `labels/gold_*.json`, alignment, `alignment_threshold.json` | `alignment_eval.json`, `figures/alignment_pr.png` | seconds |
| align | `trailercut.align --threshold T` (same condition) | as `align`; `T` = `best_f1.threshold` from `alignment_eval.json` | same as `align`, `source: manual` | ~1 min |
| align | `trailercut.align_diagnostics` | `labels/gold_*.json`, alignment, `alignment_threshold.json`, `cache/trailer_asr/`, film and trailer embeddings | `figures/alignment_diagnostics.png`, `alignment_diagnostics.json` | not timed |
| audio | `trailercut.trailer_asr` | `cache/trailer_audio/` | `cache/trailer_asr/{id}.json` (skips done ones) | ~13 s/trailer |
| audio | `trailercut.dialogue_align` | transcripts, MovieNet subtitles | `data/features/dialogue_alignment/{id}.csv` | not timed |
| audio | `trailercut.av_sync` | alignment, dialogue alignment, transcripts, MovieNet shots, `labels/gold_*.json` | `av_sync.csv`, `av_sync.json`, `figures/av_sync.png` | not timed |
| rq1 | `trailercut.film_features` | alignment, embeddings, subtitles, story | `data/features/film_shots.parquet` | ~1 min |
| rq1 | `trailercut.model` | film shot table, film embeddings | `model_report.json`, `model_preds.parquet`, `model_shap.parquet` | ~15 min (approx., not timed yet) |
| rq1 | `trailercut.model_eval` | `model_preds.parquet` | `model_eval.json` | seconds |
| rq1 | `trailercut.model_figures` | `model_preds.parquet`, `model_shap.parquet`, `film_shots.parquet`, `model_report.json`, `model_eval.json` | `figures/{model_pr,model_ablation,model_tolerance,model_shapes}.png` | seconds |
| rq2 | `trailercut.trailer_features` | alignment, trailer shots, audio, subtitles, trailer embeddings, meta, `av_sync.csv` if present | `trailers.parquet`, `trailer_curves.json` | ~1 min |
| rq2 | `trailercut.cluster` | trailer table | `clusters.csv`, `cluster_report.json`, `figures/clusters_all_umap.png` | ~1 min |
| rq2 | `trailercut.figures` | alignment, status log, trailer shots, impostor null cache | `figures/{data_funnel,alignment_scores,alignment_examples,story_depth,corpus}.png`, `summary.json` | seconds |
| rq2 | `trailercut.case_studies` | `trailers.parquet`, `av_sync.csv`, `model_preds.parquet` | `case_studies.json` | seconds |
| viz | `trailercut.viewer_export` | alignment, `film_shots.parquet`, `trailer_curves.json`, model preds/SHAP if present, keyframes, dialogue alignment and transcripts if present | `viz/data/index.json`, `viz/data/{id}.json`, `viz/data/{id}.vtt`, `cache/shot_colors/{id}.json` | seconds once colors are cached; first color pass ~1 ms per keyframe |
| viz | `trailercut.rq1_export` | film shot table, film embeddings, `model_eval.json`, `model_tune.json` if present | `viz/data/rq1.json` | under a minute |
| ebm (opt-in) | `trailercut.model_ebm` (`--plot-only` redraws from the JSON) | film shot table, film embeddings, `model_eval.json` | `model_ebm.json`, `figures/ebm_shapes.png` | ~30 min |

## Curate steps (`curate.sh`)

| step | command | reads | writes |
|---|---|---|---|
| `recover` | `trailercut.recover` | status log, MovieNet meta, film embeddings, `impostor_threshold` from `alignment_threshold.json` | replacement trailers for dead links (runs `trailer_prep` and `embed trailer` itself), `trailer_recovery.csv`, updated status |
| `audit` | `trailercut.trailer_audit` | status log, YouTube metadata (no download; cached in `cache/yt_meta/`), `cache/trailer_asr/`, alignment, MovieNet meta | `data/ids/trailer_audit.csv` |
| `swap ID ...` | `trailercut.recover --ids ID ...` | as `recover` | replaces these films' working trailers; old files kept in `cache/replaced/{id}/` and restored if no candidate passes |
| `swap-search` | `trailercut.swap search` | `trailer_audit.csv` (`swap` rows whose link is still MovieNet's), status log, YouTube search | `swap_candidates.csv` |
| `swap-download` | `trailercut.swap download [--sleep 30] [--per-film 1] [--recheck]` | `swap_candidates.csv`, `swap_downloads.csv` | `data/trailers/candidates/{id}__{youtube_id}.mp4`, `swap_downloads.csv`, `unavailable` column of `swap_candidates.csv` |
| `swap-apply` | `trailercut.swap apply` | candidates, downloaded videos, `impostor_threshold` | swapped trailers (runs `trailer_prep` and `embed trailer`), updated status, rows appended to `trailer_recovery.csv` |
| `labels` | `trailercut.label_tool --n 10 --seed 0` | alignment, keyframes | `labels/label_tool.html`, `labels/label_sample.json`, `labels/img/` |
| `tune` | `trailercut.align_tune`, `rerank --eval`, `model_tune` | labels, embeddings, keyframes, film shot table; `rerank` downloads SSCD weights to `cache/models/` | `align_tune.json`, `rerank_eval.json`, `model_tune.json` |

`trailercut.swap` also works on films new to the dataset with `--set expand`: `swap prepare --set
expand` takes films in `data/ids/films_1100.csv` that aren't in the status log yet, keeps MovieNet's
link as rank 0 when its YouTube metadata looks like a clean trailer (alive, title passes the filter,
60 to 240 s) and searches otherwise. Nothing is downloaded in `prepare`. `download --set expand` and
`apply --set expand` then use `expand_candidates.csv` and `expand_downloads.csv`. Not in `curate.sh`
yet.

`cache/` is `$TRAILERCUT_DATA/cache` (default `data/cache`). `<tag>` is the CLIP backbone
(`b32`, `l14`, `h14`). Keyframes, `trailer_prep` and both `embed` steps are the heavy ones and have Slurm jobs in `hpc/` (the
cluster job uses `--pack`; `run_all.sh` doesn't). The label tool copies its images into `labels/img/`,
so it works with either. The Trailer View's frame thumbnails load straight from `data/movienet/frames/`
(loose JPEGs) and `data/cache/trailer_frames/`, so they only show on a machine with that data in the
default location.

`recover` and `swap apply` read `impostor_threshold` from the committed `alignment_threshold.json`
(the 5% impostor cutoff, kept even when `align` runs with the label-picked cutoff).
`align` without `--threshold` writes the 5% impostor cutoff; the `align` stage then re-aligns with the
best-F1 cutoff from the hand labels. The null is still computed and its pass rate at `T` is saved as
`fpr`. `label_eval` runs between the two, so it sees the impostor cutoff as "current".
`model`, `model_tune` and `rq1_export` drop film shots in the first and last 1% of each film
(`model.EDGE`); `viewer_export` joins model scores by shot, so those shots have no score.

## Configuration

| env var | default | effect |
|---|---|---|
| `TRAILERCUT_DATA` | `data/` | root for MovieNet files, trailers and caches (point it at scratch on the cluster) |
| `TRAILERCUT_CLIP` | `b32` | CLIP backbone; caches are keyed by it so backbones never mix |

Features, figures and labels always live in the repo checkout, not under `TRAILERCUT_DATA`.

## Rerunning after a change

| you changed | rerun from |
|---|---|
| trailer list, recovered or swapped trailers | `bash scripts/run_all.sh prep align audio rq1 rq2 viz` (prep skips done trailers; delete `cache/trailer_asr/{id}.json` of a swapped trailer so it is re-transcribed) |
| CLIP backbone | `embed film`, `embed trailer`, `concepts`, then `align` onward |
| match threshold or alignment logic | `align` (`--threshold T` for a hand-picked cutoff) onward |
| hand labels | `run_all.sh align audio rq1 rq2 viz`; `curate.sh tune` for the variant comparisons |
| film shot features | `run_all.sh rq1 rq2 viz`, then `ebm` |
| trailer features | `trailer_features`, `cluster`, `viewer_export` (it reads `trailer_curves.json`) |
| transcripts or dialogue matching | `dialogue_align`, `av_sync`, `trailer_features`, `cluster`, `viewer_export` (delete `cache/trailer_asr/{id}.json` to re-transcribe) |
| model | `run_all.sh rq1 rq2 viz` (`case_studies` reads the predictions), `ebm` if the readable features changed |
| tolerance eval only | `model_eval` (reuses saved predictions, no retraining) |
| EBM figure only | `model_ebm --plot-only` (reads `model_ebm.json`, no retraining) |
| the Trailer View or RQ1 page only | nothing; reload `viz/index.html` or `viz/rq1.html` |

"`align` onward" includes `viewer_export` and `rq1_export`.

Alignment outputs are named by IMDb ID only, so switching backbones overwrites them. Commit or copy
`data/features/alignment/` first if you need both.
