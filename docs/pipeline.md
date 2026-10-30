# Pipeline

How data flows from MovieNet and YouTube to the model, clusters and figures. Two scripts:

- `scripts/run_all.sh [stage ...]` runs the stages you name, one after another, in the order given.
  With no arguments it lists the stages and runs nothing. Listed order: `data prep align audio rq1 viz`,
  then `rq2` (RQ2 trailer features, clustering, case studies) and `ebm` (slow glass-box model), which
  run only when named. Download, shot, embedding and transcription steps skip finished films and
  trailers; the rest recompute, which takes minutes.
- `scripts/curate.sh <step>` holds the steps that change the inputs (trailer links, labels) and are
  run on purpose: `recover`, `audit`, `swap`, `swap-prepare`, `swap-search`, `swap-download`,
  `swap-apply`, `swap-review`, `split-propose`, `split-review`, `split-report`, `split-apply`,
  `labels`, `tune`. After any of them, rebuild with
  `bash scripts/run_all.sh prep align audio rq1 viz` (plus `rq2` and `ebm` when needed).

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
                  figures                      │
                  model_ebm (opt-in)    cluster, case_studies (rq2, opt-in)
                           │
            viewer_export: per-trailer JSON for viz/ (Trailer View)
            rq1_export: lifts across all films for viz/rq1.html
```

## Steps (`run_all.sh`)

| stage | command | reads | writes | time (laptop, 293 trailers) |
|---|---|---|---|---|
| data | `movienet.py small`, then unzip | HF mirror | `data/movienet/*.zip` (annotations, meta, subtitles, shots) and video info JSON; `run_all.sh` unzips them | a few min |
| data | `movienet.py keyframes` | HF mirror (HTTP range requests) | `data/movienet/frames/{id}/` (JPEGs), or `frames/{id}.zip` with `--pack` | hours (network bound, ~125 MB/film) |
| data | `trailercut.trailers` | `data/ids/trailer_status.csv` (the link in use per film: MovieNet's or a replacement); new films get a status row through `swap --set expand` first | `data/trailers/{id}.mp4`, `data/ids/trailer_status.csv` (rewritten after every trailer) | ~20 min (293 trailers) |
| prep | `trailercut.trailer_prep` | trailer videos | `data/features/trailer_shots/{id}.csv`, `cache/trailer_frames/`, `cache/trailer_audio/` | ~15 s/trailer CPU |
| prep | `trailercut.embed film` | film keyframes | `cache/clip/<tag>/film/{id}.npy` [shots, 3, d] | ~1 h (B/32, MPS) |
| prep | `trailercut.embed trailer` | trailer keyframes | `cache/clip/<tag>/trailer/{id}.npy` | ~3 min |
| prep | `trailercut.concepts` | CLIP text encoder | `cache/concept_text_<tag>.npy` | seconds |
| align | `trailercut.align [--threshold T]` (T = last `best_f1.threshold` from `alignment_eval.json`, if it exists) | embeddings, trailer shot tables, MovieNet shots | `data/features/alignment/{id}.csv`, `alignment_threshold.json`, `cache/alignment_null_<tag>.npy` | ~1 min |
| align | `trailercut.label_eval` (only if `labels/gold_*.json` exist) | `labels/gold_*.json`, alignment, `alignment_threshold.json` | `alignment_eval.json`, `figures/alignment_pr.png` | seconds |
| align | `trailercut.align --threshold T` (only if `label_eval` now picks a different T) | as `align` | same as `align`, `source: manual` | ~1 min |
| align | `trailercut.align_diagnostics` | `labels/gold_*.json`, alignment, `alignment_threshold.json`, `cache/trailer_asr/`, film and trailer embeddings | `figures/alignment_diagnostics.png`, `alignment_diagnostics.json` | not timed |
| audio | `trailercut.trailer_asr` | `cache/trailer_audio/` | `cache/trailer_asr/{id}.json` (skips done ones) | ~13 s/trailer |
| audio | `trailercut.dialogue_align` | transcripts, MovieNet subtitles | `data/features/dialogue_alignment/{id}.csv` | not timed |
| audio | `trailercut.av_sync` | alignment, dialogue alignment, transcripts, MovieNet shots, `labels/gold_*.json` | `av_sync.csv`, `av_sync.json`, `figures/av_sync.png` | not timed |
| rq1 | `trailercut.film_features` | alignment, embeddings, subtitles, story | `data/features/film_shots.parquet` | ~1 min |
| rq1 | `trailercut.model` | film shot table, film embeddings | `model_report.json`, `model_preds.parquet`, `model_shap.parquet` | ~15 min (approx., not timed yet) |
| rq1 | `trailercut.model_eval` | `model_preds.parquet` | `model_eval.json` | seconds |
| rq1 | `trailercut.model_figures` | `model_preds.parquet`, `model_shap.parquet`, `film_shots.parquet`, `model_report.json`, `model_eval.json` | `figures/{model_pr,model_ablation,model_tolerance,model_shapes}.png` | seconds |
| rq1 | `trailercut.figures` | alignment, status log, trailer shots, impostor null cache | `figures/{data_funnel,alignment_scores,alignment_examples,story_depth,corpus}.png`, `summary.json` | seconds |
| rq2 (opt-in) | `trailercut.trailer_features` | alignment, trailer shots, audio, subtitles, trailer embeddings, meta, `av_sync.csv` if present | `trailers.parquet`, `trailer_curves.json` | ~1 min |
| rq2 (opt-in) | `trailercut.cluster` | trailer table | `clusters.csv`, `cluster_report.json`, `figures/clusters_all_umap.png` | ~1 min |
| rq2 (opt-in) | `trailercut.case_studies` | `trailers.parquet`, `av_sync.csv`, `model_preds.parquet` | `case_studies.json` | seconds |
| viz | `trailercut.viewer_export` | alignment, trailer shots, `cache/trailer_audio/`, `film_shots.parquet`, model preds/SHAP if present, keyframes, dialogue alignment and transcripts if present | `viz/data/index.json`, `viz/data/{id}.json`, `viz/data/{id}.vtt`, `cache/shot_colors/{id}.json`, `cache/curves/{id}.json` | seconds once colors and curves are cached; first color pass ~1 ms per keyframe |
| viz | `trailercut.rq1_export` | film shot table, film embeddings, `model_eval.json`, `model_tune.json` and `model_ebm.json` if present | `viz/data/rq1.json` | under a minute |
| ebm (opt-in) | `trailercut.model_ebm` (`--plot-only` redraws from the JSON) | film shot table, film embeddings, `model_eval.json` | `model_ebm.json`, `figures/ebm_shapes.png` | ~30 min |

## Curate steps (`curate.sh`)

| step | command | reads | writes |
|---|---|---|---|
| `recover` | `trailercut.recover` | status log, MovieNet meta, film embeddings, `impostor_threshold` from `alignment_threshold.json` | replacement trailers for dead links (runs `trailer_prep` and `embed trailer` itself), `trailer_recovery.csv`, updated status |
| `audit` | `trailercut.trailer_audit` | status log, YouTube metadata (no download; cached in `cache/yt_meta/`), `cache/trailer_asr/`, alignment, MovieNet meta | `data/ids/trailer_audit.csv` |
| `swap ID ...` | `trailercut.recover --ids ID ...` | as `recover` | replaces these films' working trailers; old files kept in `cache/replaced/{id}/` and restored if no candidate passes |
| `swap-prepare` | `trailercut.swap prepare --set expand` | `films_1100.csv`, status log, YouTube metadata and search (no download) | `expand_candidates.csv`, appended per film (resumes) |
| `swap-search` | `trailercut.swap search` | `trailer_audit.csv` (`swap` rows whose link is still MovieNet's), status log, YouTube search | `swap_candidates.csv` |
| `swap-download` | `trailercut.swap download [--sleep 30] [--per-film 1] [--recheck]` | `swap_candidates.csv`, `swap_downloads.csv` | `data/trailers/candidates/{id}__{youtube_id}.mp4`, `swap_downloads.csv`, `unavailable` column of `swap_candidates.csv` |
| `swap-apply` | `trailercut.swap apply [--batch]` | candidates, downloaded videos, `impostor_threshold`, `trailer_recovery.csv` (to skip decided films) | swapped trailers (runs `trailer_prep` and `embed trailer`), status and recovery logs saved after every film |
| `swap-review` | `trailercut.swap_review` | status log (`source=search`, `status=ok`), film list (MovieNet's old link), YouTube metadata cache and candidate logs, alignment, trailer and film keyframes | `labels/swap_review.html`, `labels/swap_img/{id}/` (not committed); run after the rebuild so the alignment is the new trailer's. The page's Export gives `swap_review_<name>.json` for `labels/` |
| `split-propose` | `trailercut.resplit_clip propose [--core]` | `alignment_diagnostics.json` (`cuts.film_floor_1pct`), trailer shot tables, embeddings, videos and keyframes, status log, `labels/label_trailers.csv` | `split_proposals.csv`, appended per trailer (resumes); changes no shots |
| `split-review` | `trailercut.resplit_clip review [--n 50]` | `split_proposals.csv`, trailer videos | `labels/split_review.html`, `labels/split_img/` (not committed); a sample of `--n` proposed cuts spread evenly over `sim`, shown shuffled. The page's Export gives `split_review_<name>.json` for `labels/` |
| `split-report` | `trailercut.resplit_clip report` | `split_proposals.csv`, `labels/split_review_*.json` | printed only: share of real cuts and of all proposals kept at each `sim` cutoff |
| `split-apply --max-sim S` | `trailercut.resplit_clip apply --max-sim S` | `split_proposals.csv`, trailer shot tables and videos | rewritten `trailer_shots/{id}.csv` and `cache/trailer_frames/{id}/`, trailer embedding removed, `split_applied.csv`; a trailer whose shots changed since the proposals is skipped. Then `embed trailer` and the rebuild |
| `labels` | `trailercut.label_tool --n 10 --seed 0` | alignment, keyframes | `labels/label_tool.html`, `labels/label_sample.json`, `labels/img/` |
| `tune` | `trailercut.align_tune`, `rerank --eval`, `model_tune` | labels, embeddings, keyframes, film shot table; `rerank` downloads SSCD weights to `cache/models/` | `align_tune.json`, `rerank_eval.json`, `model_tune.json` |

The `swap-*` steps take `--set expand` to work on films new to the dataset: `swap-prepare --set
expand` takes films in `data/ids/films_1100.csv` that aren't in the status log yet, keeps MovieNet's
link as rank 0 when its YouTube metadata looks like a clean trailer (alive, title passes the filter,
60 to 240 s) and searches otherwise. Nothing is downloaded in `prepare`. `download --set expand` and
`apply --set expand` then use `expand_candidates.csv` and `expand_downloads.csv`.

`python -m trailercut.swap sheet` (contact sheets of the swaps in `cache/sheets/`) still exists, but
`curate.sh` no longer runs it; `swap-review` replaced it.

`swap apply --batch` is for hundreds of films: each round puts every undecided film's next downloaded
candidate in place, runs `trailer_prep` and `embed trailer` once over all of them, then aligns and
decides each film. Rejected films try their next candidate in the next round. Plain `apply` runs both
per film. Both keep the original trailer in `cache/replaced/{id}/` while a film is undecided; the
backup is written once (a resumed run doesn't overwrite it with a candidate) and removed after it is
restored.

## One-off steps (not in either script)

| step | command | reads | writes |
|---|---|---|---|
| re-split | `trailercut.resplit` | `alignment_diagnostics.json` (`cuts.film_floor_1pct`), trailer shot tables, trailer embeddings and videos, `labels/label_trailers.csv` | rewritten `trailer_shots/{id}.csv` and `cache/trailer_frames/{id}/`, trailer embedding removed, `resplit_log.csv` (appended per trailer; logged trailers are skipped on rerun) |
| release | `trailercut.release` | status, film list, YouTube metadata cache and candidate logs, alignment, dialogue alignment, `av_sync.csv`, `labels/gold_*.json` | `data/release/` (gitignored): see `data_dictionary.md` |

After `resplit`, run `python -m trailercut.embed trailer` (it re-embeds the trailers whose embedding was
removed), then `bash scripts/run_all.sh align audio rq1 viz`. Hand-labeled trailers are skipped.
The same holds after `curate.sh split-apply`.

`cache/` is `$TRAILERCUT_DATA/cache` (default `data/cache`). `<tag>` is the CLIP backbone
(`b32`, `l14`, `h14`). Keyframes, `trailer_prep` and both `embed` steps are the heavy ones and have Slurm jobs in `hpc/` (the
cluster job uses `--pack`; `run_all.sh` doesn't). The label tool copies its images into `labels/img/`,
so it works with either. The Trailer View's frame thumbnails load straight from `data/movienet/frames/`
(loose JPEGs) and `data/cache/trailer_frames/`, so they only show on a machine with that data in the
default location.

`recover` and `swap apply` read `impostor_threshold` from the committed `alignment_threshold.json`
(the 5% impostor cutoff, kept even when `align` runs with the label-picked cutoff).
`align` without `--threshold` writes the 5% impostor cutoff. The `align` stage reuses the last
best-F1 cutoff from the hand labels, so it is usually one pass; `label_eval` then re-scores, and only if
it picks a different cutoff does `align` run again with it. The null is still computed and its pass
rate at `T` is saved as `fpr`.
`model`, `model_tune` and `rq1_export` drop film shots in the first and last 1% of each film
(`model.EDGE`); `viewer_export` joins model scores by shot, so those shots have no score.

## Configuration

| env var | default | effect |
|---|---|---|
| `TRAILERCUT_DATA` | `data/` | root for MovieNet files, trailers and caches (point it at scratch on the cluster) |
| `TRAILERCUT_CLIP` | `b32` | CLIP backbone; caches are keyed by it so backbones never mix |
| `TRAILERCUT_FILMS` | `data/ids/films_1100.csv` | film list (`imdb_id`, `youtube_id`, `core`) for `embed film` and `figures`; the status, audit, swap and recovery logs always live in `data/ids/` |

Features, figures and labels always live in the repo checkout, not under `TRAILERCUT_DATA`.

## Rerunning after a change

| you changed | rerun from |
|---|---|
| trailer list, recovered or swapped trailers | `bash scripts/run_all.sh prep align audio rq1 viz` (prep skips done trailers; delete `cache/trailer_asr/{id}.json` of a swapped trailer so it is re-transcribed) |
| CLIP backbone | `embed film`, `embed trailer`, `concepts`, then `align` onward |
| match threshold or alignment logic | `align` (`--threshold T` for a hand-picked cutoff) onward |
| hand labels | `run_all.sh align audio rq1 viz`; `curate.sh tune` for the variant comparisons |
| film shot features | `run_all.sh rq1 viz`, then `ebm` (the RQ1 page shows its curves) |
| trailer features | `run_all.sh rq2` |
| transcripts or dialogue matching | `run_all.sh audio viz`, plus `rq2` (delete `cache/trailer_asr/{id}.json` to re-transcribe) |
| model | `run_all.sh rq1 viz`, plus `rq2` (`case_studies` reads the predictions), `ebm` if the readable features changed |
| tolerance eval only | `model_eval` (reuses saved predictions, no retraining) |
| EBM figure only | `model_ebm --plot-only` (reads `model_ebm.json`, no retraining) |
| the Trailer View or RQ1 page only | nothing; reload `viz/index.html` or `viz/rq1.html` |

"`align` onward" includes `viewer_export` and `rq1_export`. The `rq2` stage is opt-in: add it to any rebuild that should refresh the RQ2 outputs.

`viewer_export` computes each trailer's loudness and cut curves itself (100 bins, same as
`trailer_features`) and caches them in `cache/curves/{id}.json`, keyed on the shot table's
modification time, so a swap or re-split recomputes them. The views don't need the `rq2` stage.

Alignment outputs are named by IMDb ID only, so switching backbones overwrites them. Commit or copy
`data/features/alignment/` first if you need both.
