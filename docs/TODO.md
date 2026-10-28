# TODO

Move items to **Done** with the date when finished. Step-by-step detail for the
current block lives in `plan.md`.

## Now

**Audio alignment** (`plan.md` section A)
- [ ] Check dialogue anchors against the hand labels
- [ ] Sound/picture sync rate per trailer: RQ2 features + one figure
- [ ] Trailer dialogue features from the transcripts (speech share, words per minute, share of lines found in the film) to replace the subtitle proxy
- [ ] Fuse dialogue into shot matching, only if it beats the current matching on the labels

**Trailer View**
- [ ] Custom player: drop the native control bar; the trailer strip is the scrubber (click/drag to seek), small play/pause button, our own caption overlay (on by default, toggle), space plays, ← → step. The `<video>` element only shows the picture
- [ ] Split long caption cues (Whisper merges up to ~35 s, e.g. Rebel 0:01 to 0:36) into chunks of at most ~6 s or 12 words using the word timestamps

**Data**
- [ ] Replace Scarface (tt0086250, VHS-quality TV spot): add `recover --ids` to re-search one film (see `data/ids/trailer_review.csv`)

**Docs and submission**
- [ ] Doc-keeper pass for everything since 47614fd: playable view, sound track, sync panel, captions, terse labels, `trailer_asr`, `dialogue_align`, `rerank` (negative result), `model_figures`, `trailer_review.csv`; README browser note (trailers are AV1/VP9/H.264: use Chrome or Firefox)
- [ ] Refresh the numbers in the project update draft (293 trailers, cutoff 0.535) before submitting

**Clustering and Overview**
- [ ] Label the first 2 trailers of `labels/label_sample.json` with a second labeler (name box filled) for Cohen's kappa against `gold_ced.json`
- [ ] Residual clustering: regress genre and decade out of the trailer features, re-cluster, same stability/NMI checks. Note: dialogue clusters now follow genre (NMI 0.105 vs 0.033 null); add sync features once they exist
- [ ] Decide what the Overview shows if structure stays weak (continuous axes vs clusters)
- [ ] D3 Overview skeleton, sharing the trailer JSON format and selection events with the Trailer View

## Next (Nov 17 to Dec 1)

- [ ] Cluster: get HPC access, set the `CHANGE_ME` account in `hpc/*.sbatch`, check scratch in `hpc/env.sh`, follow `hpc/README.md`
- [ ] ViT-H/14 rerun on the cluster, compare alignment and model numbers to B/32
- [ ] VLM judge on the cluster (open model served with vLLM) for uncertain shots only; adopt only if it beats the labels
- [ ] Overview linked to the Trailer View
- [ ] Pick case studies (model misses, cluster extremes, sound/picture mismatches)

## Later (Dec 1 to 15)

- [ ] Final ablations and evaluation tables
- [ ] Report and presentation

## Maybe

- Expand beyond the 293 core films (~800 MovieNet films have keyframes + trailer, but no synopsis alignment)
- Letterboxd list as a filter for case studies
- Film mode in `trailer_prep` for user-owned films outside MovieNet
- Transcode trailers to H.264 so Safari plays all of them

## Dropped

- Keypoint verification of CLIP's top 5: copy detection (SSCD) already showed stricter visual matching doesn't help with 3 keyframes per film shot
- Whisper on the cluster: it runs locally (~13 s per trailer)
- Temporal smoothing of the alignment: lower F1 than raw matching on the labels

## Done

- 2026-10-28: all 293 trailers transcribed; dialogue alignment: 2,368 of 8,273 speech segments (29%) found in the film, 187 trailers with at least one
- 2026-10-28: Trailer View: playable trailer with timeline following it, one sound track (loudness, film dialogue in violet, one sound link at a time), sound vs picture panel, captions on the video (`.vtt`), terse labels, no act labels
- 2026-10-28: `trailer_asr` (Whisper) and `dialogue_align` (trailer lines to film subtitles)
- 2026-10-28: SSCD copy-detection re-ranking and gate evaluated against the labels (`rerank.py`): worse than or equal to CLIP, not used
- 2026-10-28: model figures for the report: PR curves, ablation, tolerance (`model_figures.py`)
- 2026-10-28: reviewed the 15 low-match trailers and the 3 weakest recovered ones: all the right film (`data/ids/trailer_review.csv`)
- 2026-10-28: `recover` uses the impostor cutoff (`impostor_threshold`), not the label-picked one
- 2026-10-28: RQ1 page `viz/rq1.html` (`rq1_export`): pick-rate lifts across all films with film-bootstrap intervals
- 2026-10-28: model variants evaluated (`model_tune.py`): none beat the plain model
- 2026-10-28: logo/credit shots (first and last 1% of each film) dropped from the RQ1 model and page (`model.EDGE`)
- 2026-10-28: match threshold picked from the hand labels (best F1, 0.535); everything downstream rerun
- 2026-10-28: hand labels for the 10 sampled trailers, one labeler (`labels/gold_ced.json`, 710 usable shots)
- 2026-10-28: model eval with a ±k shot tolerance (`model_eval.py`): neighbor-shot errors don't explain the weak RQ1 score
- 2026-10-28: dead-link recovery: all 49 replaced, 293/293 trailers
- 2026-10-28: RQ1 model (grouped CV, baselines, ablations, out-of-fold SHAP) and first results
- 2026-10-28: D3 Trailer View: cut timeline, movie barcode, shot-to-film links, click to lock, order scatter, grouped SHAP
- 2026-10-28: `align --threshold` for a hand-picked cutoff
- 2026-10-27: Slurm setup for the cluster (`hpc/`)
- 2026-10-27: labeling tool and label evaluation
- 2026-10-27: scraping, shot detection, CLIP embeddings, alignment, initial clustering
