# TODO

Move items to **Done** with the date when finished. Step-by-step detail for the
current block lives in `plan.md`.

## Now

- [ ] Finish the trailer swaps: candidates found for 83 of 84 flagged films (`data/ids/swap_candidates.csv`). Downloads run with `bash scripts/curate.sh swap-download --top 1` (resumable, stops on bot checks; rerun later); then `bash scripts/curate.sh swap-apply`, then rebuild with `bash scripts/run_all.sh prep align audio rq1 rq2 viz`. Second candidates (`--top 2`) for films whose first pick fails

- [ ] Doc pass: `run_all.sh` is now staged (`bash scripts/run_all.sh rq1 viz`), input-changing steps moved to `scripts/curate.sh`; update README, pipeline.md, workflow.md

**Clustering and Overview**
- [ ] Label the first 2 trailers of `labels/label_sample.json` with a second labeler (name box filled) for Cohen's kappa against `gold_ced.json`
- [ ] Residual clustering: regress genre and decade out of the trailer features, re-cluster, same stability/NMI checks. Note: dialogue clusters follow genre (NMI 0.103 vs 0.033 null); the speech group follows decade and its k = 8 is at the edge of the search range
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

- Fusing dialogue into shot matching: a found line is within 30 s of the hand-labeled source shot only 59% of the time (93 shots), well below the picture matcher's precision

- Keypoint verification of CLIP's top 5: copy detection (SSCD) already showed stricter visual matching doesn't help with 3 keyframes per film shot
- Whisper on the cluster: it runs locally (~13 s per trailer)
- Temporal smoothing of the alignment: lower F1 than raw matching on the labels

## Done

- 2026-10-29: project update draft refreshed with the final numbers (local, not committed)
- 2026-10-28: docs pass for the sound work, Trailer View player and Scarface replacement
- 2026-10-28: dialogue anchors checked against the hand labels: 59% within 30 s of the labeled shot (`av_sync.json`)
- 2026-10-28: sound vs picture sync rate per trailer (`av_sync.py`, `figures/av_sync.png`): pooled 0.51, median 0.50 over 167 trailers with 3+ lines
- 2026-10-28: trailer speech features (`speech_share`, `words_per_min`, `lines_found`, `sync_rate`) as the RQ2 "speech" group
- 2026-10-28: Trailer View custom player: no native controls, play/pause button, time, CC toggle, drag the strip or sound track to scrub
- 2026-10-28: caption cues split at ~6 s, 12 words or pauses using the word timestamps
- 2026-10-28: `recover --ids`; Scarface's TV spot replaced by an HD trailer (64% of shots match at the impostor cutoff vs 0%)

- 2026-10-28: all 293 trailers transcribed; dialogue alignment: 2,399 of 8,330 speech segments (29%) found in the film, 188 trailers with at least one
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
