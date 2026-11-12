# TODO

Move items to **Done** with the date when finished. Step-by-step detail for the
current block lives in `plan.md`.

## Now

- [ ] Review the replaced trailers on the swap review page (`curate.sh swap-review`: all core swaps plus 30 sampled new films, weakest first), export the verdicts to `labels/`
- [ ] Audit rerun on all 1,082 trailers (2026-11-12, first run since the new films; the 789 new ones were never audited): check the flags on the audit review page (`curate.sh audit-review`) and export the verdicts; `swap search` follows them
- [ ] Drop TV titles from the dataset (IMDb `title_type`): only High School Musical (TV movie) is in it; needs a full rerun from `film_features` on, best done once with any swaps from the audit

- [ ] Match classifier: instead of the single 0.515 cutoff, decide matched / not matched with a small model trained on the hand labels (logistic regression on match score, margin between the best and second-best film shot, shot length, blankness, ...), cross-validated grouped by trailer. Aimed at the main error, right film but wrong shot, which the cutoff can't separate

**Clustering and Overview**
- [ ] Label the first 2 trailers of `labels/label_sample.json` with a second labeler (name box filled) for Cohen's kappa against `gold_ced.json`

- [ ] Final writeup: results and discussion, figures picked and annotated, RQ2 section; methods draft in `notes/`. Check the film-theory references in `notes/narrative_notes.md` before citing them
- [ ] Course guidelines, open items (`course_learnings.md`): projector check; a small user evaluation (3 to 5 people); report figures and contributions per the paper-writing lab

## Next (Nov 17 to Dec 1)

- [ ] VLM trial on the cluster (`vlm_rerank`, `hpc/vlm_trial.sbatch`): Qwen2.5-VL picks the match among CLIP's top 5 for the labeled shots; adopt only if it beats CLIP on the labels (F1 0.897 on the same shots)
- [ ] DINOv2 alignment trial on the cluster (`backbone_trial`, `TRAILERCUT_CLIP=dinov2`), plus the per-film recheck of H/14 and L/14
- [ ] Whisper large-v3 transcripts on the cluster (`hpc/asr.sbatch`; needs the trailer audio copied up); compare dialogue matching with Whisper small
- [ ] Pick case studies from `case_studies.json` (after the rerun) plus cluster extremes

## Later (Dec 1 to 15)

- [ ] Final ablations and evaluation tables
- [ ] Report and presentation

## Maybe

- Letterboxd list as a filter for case studies
- Film mode in `trailer_prep` for user-owned films outside MovieNet

## Dropped

- Fusing dialogue into shot matching: a found line is within 30 s of the hand-labeled source shot only 59% of the time (93 shots), well below the picture matcher's precision

- Keypoint verification of CLIP's top 5: copy detection (SSCD) already showed stricter visual matching doesn't help with 3 keyframes per film shot
- Temporal smoothing of the alignment: lower F1 than raw matching on the labels
- Automatic synopsis-to-shot alignment for the extra films with a written synopsis: time position already covers what `synopsis_pos` adds to RQ1
- ViT-H/14 for alignment (backbone trial, 9 labeled films): best F1 0.672 vs 0.893 for B/32, top-1 0.70 vs 0.94; B/32 stays
- Title-card filter (weak card matches count only at the impostor cutoff): on the labels it threw out 19 right matches to drop 9 bad ones
- Local trailer files in the Trailer View: YouTube is the video source (also makes the Safari transcode moot)
- Split review refinement (a second, 100-shot review to tune the split pass): the first review (77% real cuts, flat across cutoffs) already justified applying every proposal
- The last 18 MovieNet films without a trailer: 13 had candidates (43 downloads) that all failed the match check, 5 have none left; mostly obscure or TV titles
- Bulk trailer swaps in one `recover --ids` run: blocked by YouTube's bot check; replaced by the three-step `swap`

## Done

- 2026-11-12: `case_studies` rerun on current data (Moonstruck and Ferris Bueller no longer top the order lists; their old trailers were scene clips)
- 2026-11-12: IMDb mapping for all 1,100 MovieNet films (`imdb_meta.csv` with `title_type`; 26 cut-off titles fixed)

- 2026-11-12: public site live at https://what-makes-the-cut.pages.dev (Cloudflare Pages; `site_build`: trailers from YouTube, one thumbnail sheet per trailer and per film, all 1,082 trailers)
- 2026-11-12: Hugging Face dataset updated (titles and years fixed, `imdb_current`, A Walk to Remember's length 111 s to 81 s)
- 2026-11-11: Trailer View: YouTube is the video source (`ytmedia.js`, captions from the transcript), small frames with full-width reasons and a one-line top reason, order chart on the timeline's time scale, Sound vs picture panel dropped, "Compared with all trailers" panel, keyboard works with the YouTube player, no separator dots
- 2026-11-11: film metadata checked against IMDb (`data/ids/imdb_meta.csv`): 22 cut-off titles and their years fixed, every film has a year, 175 films gained IMDb's current genres, Force Majeure's live id
- 2026-11-11: RQ2 on all 1,082 films with every genre a film has as a control (k = 2, stability 0.92); cluster numbering kept from the previous run; the RQ2 page filters and lists all genres and links each film to the Trailer View
- 2026-11-11: cluster setup on NYU Torch (`hpc/`: setup as a job, rules in `hpc/README.md`); backbone trial run there
- 2026-11-10: Hugging Face dataset published, public (`lvtwoodpecker/movienet-trailer-match`)
- 2026-11-06: RQ1 on all 1,082 films; lift now relative to the films' mean positive rate (random 1.00x, full 2.75x)

- 2026-11-06: `shap_stability`: top reason stable for 76% of shots across retrained models (chance 16%), top-3 set for 55% (5%)
- 2026-11-06: `error_slices`: lift per genre, decade, film length, trailer size and subtitles, with bootstrap intervals and permutation tests (BH-corrected); `figures/error_slices.png`
- 2026-10-30: Residual clustering: regressed genre and decade out of feature matrix via Ridge regression; metadata NMI dropped to near zero (<= 0.011), while 2-cluster structure and high stability (overall ARI 0.86, visual ARI 0.95) persisted, proving independent structural editing styles (RQ2)
- 2026-10-30: Trailer View redesign: the video clock drives every highlight (clicks seek, no click lock), video and a "now" panel side by side above the timeline, keyboard controls, `?t=` deep links
- 2026-10-30: `release`: dataset card numbers from the pipeline outputs, Hugging Face viewer configs, `check()` (no text columns or file paths, tables agree), `--push USER/NAME`; built locally (1,058 films, 90,750 trailer shots)
- 2026-10-30: swap review page shows all core swaps plus a random sample of new films (`--new`, default 30), weakest matches first
- 2026-10-30: `swap search` appends to the candidate list; with `--set expand` and no `--ids` it searches the new films that ran out of candidates
- 2026-10-30: audit review page (`audit_review.py`, `curate.sh audit-review`); `swap search` uses its verdicts
- 2026-10-30: status-file race fixed: `recover.save` re-reads `trailer_status.csv` and updates only its own films (atomic write); `swap download` skips films already swapped
- 2026-10-30: rebuild for the new films: Whisper, `resplit`, CLIP splits applied on every trailer with proposals (`--max-sim 1.0`), `embed trailer`, align, audio, diagnostics on 968 trailers; then 90 more from the retry (1,058)
- 2026-10-30: new films: 765 of 807 have a trailer (batch apply, then a retry of the 117 rejected: re-judged under the beats-other-films rule and up to 2 more downloads each)
- 2026-10-30: swap retry on 12 core films (12 core swaps in the overnight batch)
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
