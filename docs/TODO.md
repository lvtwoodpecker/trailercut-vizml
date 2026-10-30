# TODO

Move items to **Done** with the date when finished. Step-by-step detail for the
current block lives in `plan.md`.

## Now

- [ ] Review the 132 replaced core trailers on the swap review page (`curate.sh swap-review`), export the verdicts to `labels/`
- [ ] `split-apply --max-sim 1.0` on the core (all 919 proposed cuts; running overnight), then `embed trailer` and the rebuild (`align audio rq1 viz`, `ebm`)
- [ ] Swap retry on 12 core films (`swap apply --batch --retry`, running overnight): the 6 `subtitled` ones, Twelve Monkeys (Spanish dub), Bourne Identity, Hitchhiker's Guide, Ocean's Twelve, Buried, Freddy vs. Jason; check the results on the swap review page, then transcribe the swapped trailers. The Wizard of Oz is left out on purpose: a swap would drop its ~70 hand labels
- [ ] New films: `swap apply --set expand --batch` (running); rerun it once at the end to pick up the two late downloads (Rio Bravo, The Theory of Everything)
- [ ] New films: Whisper (`audio`), `resplit`, `embed trailer`, then rebuild (`align audio rq1 viz`, `ebm`)
- [ ] Push the commits

- [ ] Rerun `case_studies` after the swaps: Moonstruck and Ferris Bueller top the order and ending lists only because their old trailers were scene clips
- [x] Status-file race (fixed: `recover.save` re-reads the file and updates only the films it decided; `swap download` skips films already swapped): `swap apply` loads `trailer_status.csv` at start and rewrites the whole file after every film, so two runs at once undo each other's swaps (an overnight expand batch reverted 12 core swaps; a rerun picked the same trailers again). Fix: each save re-reads the file and updates only its own films
- [ ] Retry the 117 new films whose candidates were all rejected (running: re-judge, up to 2 more downloads each, then the per-trailer pipeline and `rq1 viz`); ~48 films are out of candidates and need a wider search: that batch ran with the old 25% rule, before the beats-other-films check
- [ ] Match classifier: instead of the single 0.515 cutoff, decide matched / not matched with a small model trained on the hand labels (logistic regression on match score, margin between the best and second-best film shot, shot length, blankness, ...), cross-validated grouped by trailer. Aimed at the main error, right film but wrong shot, which the cutoff can't separate
- [ ] Rerun `trailer_audit` after the rerun (also writes the new `subtitled` flag): the committed file still shows the old span and `clip` flag for the four replaced scene clips, and the swapped trailers can only be checked for dubs after Whisper

**Clustering and Overview**
- [ ] Label the first 2 trailers of `labels/label_sample.json` with a second labeler (name box filled) for Cohen's kappa against `gold_ced.json`
- [ ] Residual clustering: regress genre and decade out of the trailer features, re-cluster, same stability/NMI checks. Note: dialogue clusters follow genre (NMI 0.103 vs 0.033 null); the speech group follows decade and its k = 8 is at the edge of the search range
- [ ] Decide what the Overview shows if structure stays weak (continuous axes vs clusters)
- [ ] D3 Overview skeleton, sharing the trailer JSON format and selection events with the Trailer View

## Next (Nov 17 to Dec 1)

- [ ] Dataset release on Hugging Face: `release` builds `data/release/` locally (annotations under CC BY-NC 4.0); publish (`plan.md` section G)
- [ ] Expand to all 1,100 MovieNet keyframe films (`plan.md` section H): keyframes for the new films, then the batch apply and rebuild above

- [ ] Cluster: get HPC access, set the `CHANGE_ME` account in `hpc/*.sbatch`, check scratch in `hpc/env.sh`, follow `hpc/README.md`
- [ ] ViT-H/14 rerun on the cluster, compare alignment and model numbers to B/32
- [ ] VLM judge on the cluster (open model served with vLLM) for uncertain shots only; adopt only if it beats the labels
- [ ] Overview linked to the Trailer View
- [ ] Pick case studies from `case_studies.json` (after the rerun) plus cluster extremes

## Later (Dec 1 to 15)

- [ ] Final ablations and evaluation tables
- [ ] Report and presentation

## Maybe

- Letterboxd list as a filter for case studies
- Film mode in `trailer_prep` for user-owned films outside MovieNet
- Transcode trailers to H.264 so Safari plays all of them

## Dropped

- Fusing dialogue into shot matching: a found line is within 30 s of the hand-labeled source shot only 59% of the time (93 shots), well below the picture matcher's precision

- Keypoint verification of CLIP's top 5: copy detection (SSCD) already showed stricter visual matching doesn't help with 3 keyframes per film shot
- Whisper on the cluster: it runs locally (~13 s per trailer)
- Temporal smoothing of the alignment: lower F1 than raw matching on the labels
- Automatic synopsis-to-shot alignment for the extra films with a written synopsis: time position already covers what `synopsis_pos` adds to RQ1
- Bulk trailer swaps in one `recover --ids` run: blocked by YouTube's bot check; replaced by the three-step `swap`

## Done

- 2026-10-30: split review: 48 verdicts (`labels/split_review_ced.json`), 77% real cuts, flat at about 80% for every `sim` cutoff, so all proposals get applied (`--max-sim 1.0`)
- 2026-10-30: audit `subtitled` hard flag; swap check accepts weak matches that beat 5 random other films (`beats_others` 0.45); `swap apply --batch --retry`; swap search asks for 20 results; `recover` title filter ignores the film's own title and rejects subtitled uploads
- 2026-10-30: `run_all.sh align` reuses the last label cutoff
- 2026-10-29: EBM rerun on the rebuilt core: lift 2.74 vs 2.82 for LightGBM; `rq1_export` rerun
- 2026-10-29: `align_diagnostics` rerun: language split covers all 293 trailers; 3.4% of 24,188 trailer shots below the film floor
- 2026-10-29: second split pass with CLIP (`resplit_clip.py`, `curate.sh split-*`): propose, review page, report, apply; `resplit.rewrite` takes any number of cuts per shot

- 2026-10-29: new-film trailer downloads: a pick for all 798 new films (`expand_downloads.csv`), with a wider re-search (20 results) for Rio Bravo and The Theory of Everything
- 2026-10-29: release table trimmed to the link in use and its match quality (`trailers.csv`); swap history stays in `data/ids/`
- 2026-10-29: swap review page (`swap_review.py`, `curate.sh swap-review`) replaces the contact sheets
- 2026-10-29: core rebuild after the 77 swaps and the re-split: cutoff 0.515 from 639 labels (P 0.87, R 0.92), full lift 3.15x (`results.md`)
- 2026-10-29: re-split of the core trailers: 1,225 shots split in 242 of 259 checked trailers (`resplit_log.csv`)
- 2026-10-29: `run_all.sh` is a menu (no arguments lists the stages); `rq2` and `ebm` opt-in; `figures` in `rq1`; the Trailer View computes its own audio/cut curves
- 2026-10-29: RQ1 page: "Effect of each feature" section with the 6 most important EBM curves

- 2026-10-29: film list switched to all 1,100 films (`paths.IDS`, `TRAILERCUT_FILMS`); funnel figure counts MovieNet links and working trailers
- 2026-10-29: `release` builder for the dataset (`data/release/`, local): links, shots, alignments, sync, anonymized hand labels, dataset card
- 2026-10-29: `swap apply --batch`, `swap sheet`, `curate.sh` passes `--set`; trailers, recover, swap prepare and apply save after every film and resume
- 2026-10-29: labels only count for the trailer that was labeled (`labels/label_trailers.csv`); Miami Vice's 71 labels dropped after its swap (641 left, 639 usable: 2 are "unsure")
- 2026-10-29: 77 flagged core trailers swapped for official theatrical ones (6 kept); first-choice trailers picked for 798 new films (349 MovieNet links, 449 search)
- 2026-10-29: docs pass for the staged pipeline, `curate.sh`, EBM, shapes, diagnostics, case finder, audit and swaps
- 2026-10-29: `run_all.sh` split into stages (`data prep align audio rq1 rq2 viz`, opt-in `ebm`); input-changing steps in `scripts/curate.sh`
- 2026-10-29: swap candidates downloaded for 83 of 83 searchable flagged films, one at a time, over a different network after the bot check (`swap_downloads.csv`)
- 2026-10-29: three-step trailer swap (`swap.py` search, download, apply); unavailable candidates marked with the reason; `prepare --set expand` for new films (metadata first)
- 2026-10-29: film list of all 1,100 MovieNet keyframe films, 993 with a trailer link (`films_1100.csv`)
- 2026-10-29: trailer audit with hard and soft flags (`trailer_audit.csv`); stricter `recover` search (official trailers first; dubs, teasers, TV spots, re-releases rejected)
- 2026-10-29: Mulholland Drive's German dub replaced by the official trailer
- 2026-10-29: `recover --min-span 0.2`; four scene clips replaced (Moonstruck, Commando, Ferris Bueller, Phone Booth)
- 2026-10-29: EBM glass-box model (`model_ebm.py`): lift 2.83 vs 2.88 for LightGBM on the same features (before the swaps); `figures/ebm_shapes.png`
- 2026-10-29: shape plots from binned SHAP (`figures/model_shapes.png`)
- 2026-10-29: case finder (`case_studies.py`, `case_studies.json`)
- 2026-10-29: alignment diagnostics (`align_diagnostics.py`): score by label type, match rate by language, likely missed cuts
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
