# TODO

Owner in brackets. Move items to **Done** with the date when finished.

## Open right now (2026-10-28, end of session)

- [ ] **[Cedric]** `trailer_asr` is mid-run (165/293 done); rerun `python -m trailercut.trailer_asr` (skips done ones), then `dialogue_align`, `viewer_export`
- [ ] **[Cedric]** Split long caption cues (Whisper merges up to ~35 s, e.g. Rebel 0:01 to 0:36) into chunks of at most ~6 s or 12 words using the word timestamps in `viewer_export.vtt()`
- [ ] **[Cedric]** Check the new player captions (`viz/data/{id}.vtt`, `<track default>`) in Chrome and Firefox; CC button should toggle them
- [ ] **[Cedric]** Custom player (chosen 2026-10-28): drop the native control bar; the trailer strip is the scrubber (click/drag to seek, playhead already there), a small play/pause button, caption overlay drawn by us (VTT cues or `D.speech` words, toggle, on by default), keyboard: space play, ← → step. Keep the `<video>` element only as the picture.
- [ ] **[Cedric]** Plan section A (audio): dialogue anchors vs hand labels; sound/picture sync rate per trailer as RQ2 features + one figure; fusion only if it beats the labels; trailer dialogue features from transcripts (see `plan.md`)
- [ ] **[Cedric]** Replace Scarface (tt0086250, VHS-quality TV spot): add `recover --ids` to re-search a working link (`data/ids/trailer_review.csv`)
- [ ] **[Cedric]** Doc-keeper pass for everything since 47614fd: playable view, sound track, sync panel, captions, terse labels, `trailer_asr`, `dialogue_align`, `rerank` (negative result), `model_figures`, `trailer_review.csv`; README browser note (AV1/VP9: Chrome/Firefox)
- [ ] **[Cedric]** Cluster: get HPC access, set `CHANGE_ME` account in `hpc/*.sbatch`, check scratch in `hpc/env.sh`, follow `hpc/README.md`; then write the VLM-judge job (vLLM, open model, uncertain shots only)
- [ ] **[Cedric]** Refresh numbers in the project update draft (293 trailers, label-picked cutoff 0.535) before submitting
- [ ] **[Estee]** Label the first 2 sample trailers (name box filled) for kappa
- [ ] **[Estee]** RQ2: dialogue clusters now follow genre (NMI 0.105 vs 0.033 null); add sound/picture sync features once they exist
- Stale, drop: "keypoint verification" (SSCD showed stricter visual matching doesn't help); "Whisper on the cluster" (runs locally)

## Now (Nov 3 to 17)

- [ ] **[Cedric]** Check the 15 trailers that had under 15% of shots matched at the 0.74 cutoff (all original MovieNet links): wrong video, teaser, or re-release cut? At 0.535 none is under 15%, but that hides wrong videos: Scarface tt0086250 went from 0% to 92% matched with no shot above 0.74. Replace through `recover` if wrong
- [ ] **[Estee]** Label the first 2 trailers of `labels/label_sample.json` in the label tool (name box filled, so it exports `gold_estee.json`) for Cohen's kappa against `gold_ced.json`
- [ ] **[Cedric]** Prototype keypoint-based verification to rerank CLIP's top-5 candidates (status: proposed, not started)
- [ ] **[Cedric]** Eyeball the recovered trailers with the lowest match share: *Basic* (31%), *Dogma* (33%), *Serenity* (34%)
- [ ] **[Cedric]** PR curve and ablation figures for the model
- [ ] **[Estee]** Residual clustering: regress genre and decade out of the trailer features, re-cluster, same stability/NMI checks
- [ ] **[Estee]** Decide what the Overview shows if structure stays weak (continuous axes vs clusters)
- [ ] **[Estee]** D3 Overview skeleton, sharing the trailer JSON format and selection events with the Trailer View

## Next (Nov 17 to Dec 1)

- [ ] **[Estee]** Overview linked to the Trailer View
- [ ] **[Estee]** Pick case studies (use the model's misses and the cluster extremes)
- [ ] **[Cedric]** ViT-H/14 rerun on the cluster (`hpc/`), compare alignment and model numbers to B/32
- [ ] **[Cedric]** Whisper on trailer audio to replace the subtitle dialogue proxy (cluster GPU)

## Later (Dec 1 to 15)

- [ ] **[Both]** Final ablations and evaluation tables
- [ ] **[Both]** Report and presentation

## Maybe

- Expand beyond the 293 core films (~800 MovieNet films have keyframes + trailer, but no synopsis alignment)
- Letterboxd list as a filter for case studies
- Film mode in `trailer_prep` for user-owned films outside MovieNet

## Done

- 2026-10-28: `recover` uses the impostor cutoff (`impostor_threshold`), not the label-picked one
- 2026-10-28: RQ1 page shows only the models in `model_eval.json` (no tuned bar from an older threshold)

- 2026-10-28: RQ1 page `viz/rq1.html` (`rq1_export`): pick-rate lifts across all films with film-bootstrap intervals, linked from the Trailer View
- 2026-10-28: model variants evaluated (`model_tune.py`, `model_tune.json`): context/relative features, lambdarank, LightGBM settings; none beat the plain model
- 2026-10-28: logo/credit shots (first and last 1% of each film) dropped from the RQ1 model and page (`model.EDGE`)
- 2026-10-28: temporal smoothing in alignment evaluated against the labels (`align.smooth`, `align_tune.json`): lower F1 than raw, dropped
- 2026-10-28: match threshold picked from the hand labels (best F1, 0.535); everything downstream rerun at 0.535
- 2026-10-28: hand labels for the 10 sampled trailers, one labeler (`labels/gold_ced.json`, 710 usable shots)

- 2026-10-28: model eval with a ±k shot tolerance (`model_eval.py`, `model_eval.json`); neighbor-shot errors don't explain the weak RQ1 score
- 2026-10-28: dead-link recovery finished: all 49 replaced, 293/293 trailers
- 2026-10-28: full rerun on 293 trailers (align, features, model, cluster, figures)
- 2026-10-28: RQ1 model first full run, results written into `results.md`
- 2026-10-28: Trailer View SHAP bars per film shot (grouped, click a group to expand) and keyframes on hover
- 2026-10-28: D3 Trailer View skeleton (`viewer_export`, `viz/index.html`): cut timeline, movie barcode, shot-to-film links, click to lock, order scatter, grouped SHAP
- 2026-10-28: `align --threshold` for a hand-picked cutoff
- 2026-10-28: full pipeline on the 244 then-available trailers (except `model`)
- 2026-10-28: RQ1 model module with grouped CV, baselines, ablations, out-of-fold SHAP
- 2026-10-28: dead-link recovery module
- 2026-10-27: Slurm setup for the cluster (`hpc/`)
- 2026-10-27: labeling tool and label evaluation
- 2026-10-27: scraping, shot detection, CLIP embeddings, alignment, initial clustering
