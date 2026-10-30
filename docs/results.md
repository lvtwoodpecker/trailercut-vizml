# Results log

Numbers as of each run, newest first. Source files in brackets. Old entries stay, so we can see how
numbers moved as the data and methods changed.

## 2026-10-29: core rebuild after the swaps and the re-split

Same 293 core films, CLIP ViT-B/32. The trailers are the ones after the 77 audit swaps, and likely
missed cuts were re-split, then `align`, `audio` and `rq1` were rerun. The EBM and the RQ2
stage (`rq2`) were not rerun, so `model_ebm.json`, `trailers.parquet`, `cluster_report.json` and
`case_studies.json` are still from the old trailers.

**Re-split** [`resplit_log.csv`]
- 259 trailers with likely missed cuts checked (hand-labeled trailers skipped)
- 1,225 shots split in 242 of them; 17 had nothing to split
- median TransNetV2 cut probability at the split: 0.39 (the floor is 0.1, TransNetV2's own cutoff 0.5)
- after the rebuild, 3.4% of 24,188 trailer shots are below the keyframe consistency floor (0.41), vs 6.7% of 22,313 before; the swaps changed the trailers too, so this isn't the re-split alone [`alignment_diagnostics.json`]

**Corpus and alignment** [`summary.json`]
- median trailer 138 s, 82 shots, 1.17 s per shot
- 24,188 trailer shots, 2.3% blank, 91% matched; per trailer median 92%

**Alignment vs hand labels** [`alignment_eval.json`, `alignment_threshold.json`]
- 639 labels used (554 with a source shot, 27 same scene, 58 not in film); Miami Vice's dropped after its swap
- when a source shot exists, the top-1 is right 94% of the time
- best-F1 cutoff 0.515 (was 0.535): precision 0.87, recall 0.92, F1 0.89
- lowest cutoff with precision at least 0.9: 0.635 (recall 0.84)
- impostor 5% cutoff 0.746 (was 0.740); 83% of impostor scores clear 0.515

**Alignment diagnostics** [`alignment_diagnostics.json`]
- median score by hand label: right shot 0.77 (98% above 0.515), same scene other shot 0.73 (96%), wrong shot 0.71 (84%), not in film 0.47 (43%)
- median per-trailer match rate: English 0.94 (194 trailers), other language 0.92 (14), no speech 0.95 (3); only 211 trailers have a Whisper language in this file

**Story sampling** [`summary.json`]: 33% of matched shots from the first quarter, 18% from the last;
median trailer takes 17% of its matched shots from the final act.

**Sound vs picture** [`av_sync.json`]
- 247 trailers with at least one found line; 3,152 found lines play over a matched picture, 51% of them (pooled) within 30 s of the picture's film moment
- median sync rate 0.50 over the 221 trailers with 3+ such lines
- median trailer: speech over 53% of its length, 92 words per minute
- dialogue anchors vs hand labels: 93 labeled shots under a found line, 59% within 30 s

**RQ1** [`model_report.json`, `model_eval.json`]: 464,373 film shots, 4.1% positive.

| model | AP | precision at trailer length | lift k=0 | lift ±1 | lift ±3 | lift ±5 |
|---|---|---|---|---|---|---|
| random | 0.042 | 0.046 | 1.12 | 1.11 | 1.09 | 1.09 |
| position only | 0.046 | 0.056 | 1.34 | 1.27 | 1.22 | 1.20 |
| length only | 0.054 | 0.076 | 1.83 | 1.41 | 1.27 | 1.19 |
| CLIP only | 0.070 | 0.112 | 2.70 | 1.90 | 1.57 | 1.43 |
| readable | 0.076 | 0.117 | 2.82 | 1.99 | 1.62 | 1.46 |
| full | 0.083 | 0.131 | 3.15 | 2.14 | 1.70 | 1.52 |

Base rate (share of shots within ±k of a used shot): 0.041, 0.107, 0.211, 0.295.
Precision at trailer length lost when one group is dropped from the full model: CLIP 0.014, editing
0.011, position 0.006, scale 0.005, dialogue 0.005, content 0.003. SHAP mean |value| by group: CLIP
0.28, editing 0.22, position 0.19, content 0.10, dialogue 0.07, scale 0.05.

Read: more trailer shots match and the positive rate rose (3.8% to 4.1%), so compare lifts. The full
model's exact-shot lift is 3.15x (3.22x before); the order of the models and groups is unchanged.

## 2026-10-29: trailer swaps and new-film trailer picks

Data only; nothing downstream has been rerun on the swapped trailers yet.

**Core swaps** [`trailer_status.csv`, `trailer_recovery.csv`]
- 77 of the 83 flagged core films with downloaded candidates got an official theatrical trailer; 6 kept their old one
- core trailers by source: 161 MovieNet links, 132 search picks (49 dead links, 6 earlier `recover --ids` swaps, 77 audit swaps)
- recovery log: 401 candidates tried for 138 films, 132 accepted
- labels: Miami Vice's trailer was swapped, so its 71 hand labels drop out; 641 remain over 9 trailers (639 usable, 2 are "unsure")

**New films** [`expand_candidates.csv`]
- 798 new films have a first-choice trailer from metadata: 349 keep MovieNet's link, 449 have a search pick

## 2026-10-29: EBM, feature shapes, alignment diagnostics, case studies

Same 293 films, CLIP ViT-B/32, threshold 0.535. These were computed before the full rerun after the
four scene-clip replacements and Mulholland Drive's dub swap: the alignment, model and trailer tables
still come from the old trailers for those 5 films, so numbers will shift slightly after the rerun.

**EBM vs LightGBM** [`model_ebm.json`]: glass-box EBM on the readable features, same folds.
- lift at trailer length 2.74 vs 2.82 for LightGBM on the same features
- AP 0.067, AP per film 0.088, precision at trailer length 0.107, length-adjusted AP 0.063

**Feature shapes** [`model_ebm.json` curves, `figures/ebm_shapes.png`, `figures/model_shapes.png`]
EBM curves as odds multipliers. LightGBM's binned SHAP (`model_shapes.png`, no committed numbers)
points the same way for position, shot length and visual change; shot width and words spoken stay near 1x.
- position in film: about 1.1x to 1.2x in the first quarter, 0.95x at the middle, 0.8x in the last 10%
- shot length: about 0.87x under 2 s, rising to 1.35x around 11 s
- visual change inside the shot: from 0.74x (static) to 1.2x
- the most important curve is `c_scream` (up with the score); `c_creature`, `c_night` and `c_blood` go down
- importance order (mean absolute score): scream, shot length, visual change, position, creature, night

Read: early shots, long shots, shots that change a lot and shots CLIP sees as screaming get picked
more; late shots and shots CLIP sees as creatures, night or blood get picked less.

**Alignment diagnostics** [`alignment_diagnostics.json`, `figures/alignment_diagnostics.png`]
- median match score by hand label: right shot 0.77 (97% above 0.535), same scene other shot 0.73 (97%), wrong shot 0.71 (81%), not in film 0.52 (40%)
- so wrong and same-scene matches score close to right ones; the cutoff can't separate them
- median per-trailer match rate: English 0.93 (208 trailers), other language 0.93 (78), no speech 0.95 (3); language makes no difference
- keyframe consistency floor (1% of MovieNet film shots fall below it): 0.41; 6.7% of 22,313 trailer shots fall below it, likely missed cuts

**Case studies** [`case_studies.json`]: top and bottom 5 trailers per angle.
- order (289 trailers with 15+ matched shots): most chronological Moonstruck and Ferris Bueller (1.0), Barry Lyndon 0.94; most shuffled Flash Gordon -0.38, Nashville -0.31, Vertigo -0.31
- ending: most from the last quarter Moonstruck (1.0), Flash Gordon 0.58, Hollow Man 0.56; none from it: Planet of the Apes, Dumb and Dumber, Twins, ...
- sync (142 trailers with 5+ paired lines): Twins, No Strings Attached, This Boy's Life at 1.0; The Cell, Pearl Harbor, Saving Private Ryan, Black Swan at 0.0
- model (291 trailers): best per-film precision The Sixth Sense 0.32, Star Wars: Episode VII 0.30, Cube 0.30; 0.0 for Sex, Lies, and Videotape, Flash Gordon, Vertigo, ...

Moonstruck and Ferris Bueller top the order and ending lists because their old "trailers" were scene
clips; rerun before using these lists.

## 2026-10-28: trailer sound, sync rate, SSCD rerank, Scarface replaced

Same 293 films, CLIP ViT-B/32, threshold 0.535 and edge filter as the entry below. New: Whisper
transcripts of every trailer, trailer lines matched to film subtitles, a sound vs picture sync rate,
and a "speech" feature group for RQ2. Scarface (tt0086250) now uses an HD theatrical trailer instead
of a VHS TV spot, so alignment, model and clusters were rerun.

**Dialogue alignment** [`dialogue_alignment/*.csv`]
- 8,330 Whisper speech segments over 293 trailers; 2,399 (29%) found in the film's subtitles
- 188 trailers have at least one found line
- a segment counts only with 3+ words, fuzzy score 90+ and a 10 point margin over the best match more than 30 s away

**Sound vs picture** [`av_sync.json`, `av_sync.csv`, `figures/av_sync.png`]
- 2,287 found lines play over a matched picture; 51% of them (pooled) are within 30 s of the picture's film moment
- median sync rate 0.50 over the 167 trailers with 3+ such lines
- median trailer: speech over 53% of its length, 96 words per minute

Read: about half of the film dialogue in a trailer plays over a different scene (voice-over, montage).

**Dialogue anchors vs hand labels** [`av_sync.json`]: 93 labeled shots have a found line playing
over them. For 59% the line's film time is within 30 s of the labeled source shot. Too loose to
correct or fill in picture matches, so dialogue is not fused into shot matching.

**SSCD copy-detection rerank** [`rerank_eval.json`]: CLIP's top 20 per trailer shot re-scored with
SSCD, on the labeled trailers.

| scorer | threshold | precision | recall | F1 | top pick right, no threshold |
|---|---|---|---|---|---|
| CLIP (used) | 0.535 | 0.854 | 0.905 | 0.879 | 93% |
| SSCD | 0.365 | 0.752 | 0.671 | 0.709 | 71% |
| CLIP + SSCD | 0.955 | 0.751 | 0.722 | 0.736 | 76% |

SSCD as an accept/reject gate on CLIP's pick gave no F1 gain (in-session, not saved).

**Alignment** [`summary.json`, alignment CSVs]
- 22,095 trailer shots, 1.8% blank, 91% matched; per trailer median 91%, IQR 87% to 95%
- Scarface: 97% matched (64% at the impostor cutoff, vs 0% for the old TV spot, `trailer_review.csv`)
- lowest is still *A Few Good Men* (tt0104257) at 42%, a TV spot

**Trailer review** [`trailer_review.csv`]: the 15 lowest-match trailers and the 3 weakest
recovered ones were checked by eye. All are the right film. Low match comes from foreign versions
(dubbed audio, burned-in subtitles), TV spots, dark films where shots look alike, and logos or end
credits matched to film shots.

**Story sampling** [`summary.json`]: 33% of matched shots from the first quarter, 18% from the last;
median trailer takes 16% of its matched shots from the final act.

**RQ1** [`model_report.json`, `model_eval.json`]: 464,373 film shots, 3.8% positive.

| model | AP | precision at trailer length | lift k=0 | lift ±1 | lift ±3 | lift ±5 |
|---|---|---|---|---|---|---|
| random | 0.038 | 0.043 | 1.13 | 1.10 | 1.10 | 1.11 |
| position only | 0.042 | 0.046 | 1.22 | 1.21 | 1.17 | 1.14 |
| length only | 0.049 | 0.070 | 1.85 | 1.43 | 1.28 | 1.19 |
| CLIP only | 0.064 | 0.105 | 2.77 | 1.96 | 1.60 | 1.45 |
| readable | 0.069 | 0.109 | 2.88 | 1.98 | 1.63 | 1.47 |
| full | 0.076 | 0.122 | 3.22 | 2.15 | 1.71 | 1.54 |

Lift lost when one group is dropped from the full model (precision at trailer length over the
positive rate): CLIP 0.35, editing 0.22, position 0.09, dialogue 0.03, content 0.02, scale 0.00.
SHAP mean |value| by group: CLIP 0.30, editing 0.22, position 0.19, content 0.10, dialogue 0.07,
scale 0.05. Figures: `model_pr.png`, `model_ablation.png`, `model_tolerance.png`.

Read: with the new Scarface trailer, exact-shot lifts moved by at most 0.06 (position only); same picture as below.

**Clustering** [`cluster_report.json`]

| features | k | silhouette | bootstrap ARI | genre NMI (null 95%) | decade NMI (null 95%) |
|---|---|---|---|---|---|
| all | 2 | 0.105 | 0.88 | 0.069 (0.022) | 0.024 (0.022) |
| pacing | 2 | 0.312 | 0.92 | 0.020 (0.026) | 0.079 (0.021) |
| audio | 2 | 0.260 | 0.92 | 0.017 (0.025) | 0.013 (0.022) |
| story | 3 | 0.243 | 0.91 | 0.043 (0.033) | 0.041 (0.033) |
| dialogue | 3 | 0.283 | 0.89 | 0.103 (0.033) | 0.023 (0.032) |
| visual | 2 | 0.232 | 0.94 | 0.051 (0.024) | 0.024 (0.021) |
| speech (new) | 8 | 0.291 | 0.79 | 0.072 (0.069) | 0.079 (0.061) |

HDBSCAN: 2 clusters on dialogue (65% noise) and speech (45% noise), none elsewhere.

Read: the speech group picks k = 8, the top of the search range, with silhouette almost flat
(0.27 to 0.29 for k = 2 to 8), so the k isn't meaningful. It follows decade (0.079 vs 0.061) more
than genre (0.072 vs 0.069). Adding it to "all" (56 features) leaves all clusters as weak as before.

## 2026-10-28: label-picked threshold (0.535), logo/credit edges dropped

Same 293 trailers and CLIP ViT-B/32 as the entry below. Two changes: the match threshold now comes
from hand labels, and the RQ1 model and RQ1 page drop film shots in the first and last 1% of each film.

**Alignment vs hand labels** [`alignment_eval.json`, `align_tune.json`]
- 710 usable labeled shots from 10 trailers, one labeler (`gold_ced.json`): 601 with a source shot, 38 same scene, 71 not in film
- when a source shot exists, the aligner's top-1 is right 94% of the time

| setting | threshold | precision | recall | F1 | precision within ±3 shots |
|---|---|---|---|---|---|
| raw, impostor 5% cutoff (old) | 0.74 | 0.903 | 0.601 | 0.721 | 0.908 |
| raw, best F1 (now used) | 0.535 | 0.854 | 0.905 | 0.879 | 0.862 |
| smoothed, best F1 (lam 0.02, tau 30) | 0.53 | 0.819 | 0.875 | 0.846 | 0.830 |

The lowest threshold with precision at least 0.9 is 0.74 (recall 0.60). `label_eval` sweeps in steps
of 0.005 and `align_tune` in steps of 0.01, so the raw best sits at 0.535 and 0.54 with the same scores.
Smoothing is below raw at its best setting, so it isn't used.

Error breakdown at 0.535 (measured in-session from the labels, not in a committed file): of 601
shots with a source, 544 found, 36 matched to the wrong film shot, 11 right but below the cutoff,
5 wrongly masked as blank. For the 36 wrong ones the true shot is usually rank 2, and only 5 are
within ±3 shots, so these are look-alike shots elsewhere in the film, not neighbors. 37 of 38
"same scene" and 27 of 71 "not in film" shots still get matched.

**Alignment** [`summary.json`, `alignment_threshold.json`, alignment CSVs]
- 22,031 trailer shots, 1.8% blank
- threshold 0.535, `source: manual`; 77% of impostor scores (64,875) clear it, so the impostor null no longer bounds false matches against other films
- 91% of trailer shots matched; per trailer median 91%, IQR 87% to 95%
- no trailer under 15% matched; lowest is *A Few Good Men* (tt0104257) at 42%. Scarface (tt0086250) went from 0% to 92%, with no shot above 0.74

**Story sampling** [`summary.json`]
- 33% of matched shots come from the film's first quarter, 18% from the last quarter
- median trailer takes 17% of its matched shots from the final act

**RQ1 table** [`model_report.json`]
- 464,373 film shots from 293 films after dropping the first and last 1% of each film, 3.8% used in the trailer
- the positive rate rose from 2.5% because more trailer shots match, so compare lifts, not raw AP, with older entries

**RQ1 model** [`model_report.json`]: LightGBM, 5 folds grouped by film, 64 CLIP PCs, default feature groups

| model | AP | AP per film | precision at trailer length | length-adjusted AP |
|---|---|---|---|---|
| random | 0.038 | 0.047 | 0.043 | 0.038 |
| position only | 0.042 | 0.054 | 0.048 | 0.042 |
| length only | 0.048 | 0.059 | 0.069 | 0.040 |
| CLIP only | 0.064 | 0.085 | 0.105 | 0.064 |
| readable | 0.068 | 0.088 | 0.108 | 0.064 |
| full (readable + CLIP) | 0.076 | 0.098 | 0.123 | 0.072 |
| full minus position | 0.072 | 0.093 | 0.118 | 0.068 |
| full minus dialogue | 0.075 | 0.098 | 0.121 | 0.072 |
| full minus scale | 0.075 | 0.098 | 0.121 | 0.072 |
| full minus content | 0.076 | 0.099 | 0.125 | 0.072 |
| full minus editing | 0.070 | 0.091 | 0.114 | 0.068 |
| full minus CLIP | 0.068 | 0.088 | 0.108 | 0.064 |

SHAP mean |value| by group (full model, log-odds): CLIP 0.30, editing 0.22, position 0.19,
content 0.10, dialogue 0.07, scale 0.05. Top single features: `story_pos` 0.15, `shot_len` 0.15,
`visual_change` 0.14, `dlg_rate` 0.05.

Read: full AP is 2.0x random (0.076 vs 0.038); at trailer length about 12 in 100 picks are right.
Dropping CLIP, editing or position costs the most (AP down 0.0077, 0.0062, 0.0040); dropping content
costs nothing. Length-adjusted AP holds up (0.072 vs 0.076).

**Tolerance eval** [`model_eval.json`]: precision and lift at trailer length, per film.

| model | prec k=0 | lift k=0 | prec ±1 | lift ±1 | prec ±3 | lift ±3 | prec ±5 | lift ±5 |
|---|---|---|---|---|---|---|---|---|
| random | 0.043 | 1.13 | 0.108 | 1.11 | 0.212 | 1.10 | 0.300 | 1.11 |
| position only | 0.048 | 1.28 | 0.124 | 1.27 | 0.236 | 1.22 | 0.322 | 1.19 |
| length only | 0.069 | 1.83 | 0.138 | 1.42 | 0.245 | 1.27 | 0.322 | 1.19 |
| CLIP only | 0.105 | 2.78 | 0.190 | 1.95 | 0.309 | 1.60 | 0.394 | 1.46 |
| readable | 0.108 | 2.86 | 0.193 | 1.99 | 0.318 | 1.65 | 0.403 | 1.49 |
| full | 0.123 | 3.26 | 0.211 | 2.17 | 0.335 | 1.74 | 0.422 | 1.56 |

Base rate (share of shots within ±k of a used shot): 0.038, 0.097, 0.193, 0.271.

Read: full lift at the exact shot is 3.26x (3.56x in the entry below). Position only drops the most
(1.82x to 1.28x). Threshold and edge filter changed together here, so this run can't say which one
moved it. Lift still falls as tolerance grows, so misses are still different moments, not neighbors.

**Model variants** [`model_tune.json`]: run before the re-alignment, so at threshold 0.74 (464,373
shots, 2.4% positive), with the edge filter. Same folds for every variant. Lift = precision at trailer
length over the positive rate.

| variant | AP | AP per film | precision at trailer length | lift | length-adjusted AP |
|---|---|---|---|---|---|
| random | 0.024 | 0.033 | 0.027 | 1.12 | 0.025 |
| full | 0.045 | 0.066 | 0.079 | 3.26 | 0.043 |
| full + context | 0.045 | 0.065 | 0.078 | 3.18 | 0.043 |
| full + relative | 0.047 | 0.068 | 0.081 | 3.31 | 0.045 |
| full + context + relative | 0.048 | 0.067 | 0.080 | 3.30 | 0.046 |
| ranker (lambdarank), full + context + relative | 0.041 | 0.064 | 0.077 | 3.14 | 0.040 |
| ranker, slower learning | 0.042 | 0.065 | 0.078 | 3.21 | 0.041 |
| ranker, smaller trees | 0.041 | 0.064 | 0.079 | 3.26 | 0.040 |

Read: every variant is within 0.003 of the plain full model at trailer length (lift 3.14x to
3.31x vs 3.26x). No fold-level spread is saved, so none of these gaps is clearly real; none was adopted. At the same 0.74 threshold, the full model with logo/credit edges kept had 0.088 (entry
below); without them it has 0.079, so part of the old score was logo shots matching logo shots.

**Clustering** [`cluster_report.json`]

| features | k | silhouette | bootstrap ARI | genre NMI (null 95%) | decade NMI (null 95%) |
|---|---|---|---|---|---|
| all | 2 | 0.110 | 0.91 | 0.062 (0.022) | 0.028 (0.022) |
| pacing | 2 | 0.310 | 0.90 | 0.021 (0.025) | 0.078 (0.022) |
| audio | 2 | 0.260 | 0.92 | 0.017 (0.025) | 0.013 (0.022) |
| story | 3 | 0.241 | 0.91 | 0.042 (0.032) | 0.041 (0.035) |
| dialogue | 3 | 0.283 | 0.89 | 0.105 (0.033) | 0.024 (0.031) |
| visual | 2 | 0.231 | 0.94 | 0.051 (0.024) | 0.024 (0.021) |

HDBSCAN finds no clusters (all noise) on every set except dialogue (2 clusters, 65% noise).

Read: still stable and weakly separated. Genre now beats the null for all, story, dialogue and
visual; dialogue (now k = 3) has the largest genre NMI so far, 0.105. Story and dialogue features
are computed over matched shots, so they move with the threshold. Pacing still follows decade.

## 2026-10-28: full run, 293 trailers, CLIP ViT-B/32

**Corpus** [`summary.json`, `trailer_status.csv`, `trailer_recovery.csv`]
- 293 of 293 core films have a trailer: 244 MovieNet links, 49 search replacements for dead links
- recovery tried 55 candidates for the 49 films; accepted ones match 31% to 82% of shots (median 59%); lowest are *Basic* 31%, *Dogma* 33%, *Serenity* 34%
- median trailer 134 s, 76 shots, 1.21 s per shot

**Alignment** [`summary.json`, `alignment_threshold.json`]
- 22,031 trailer shots, 1.8% blank
- threshold 0.740 at 5% false-match rate (64,875 impostor scores)
- 56% of trailer shots matched; per trailer median 59%, IQR 42% to 69% [alignment CSVs]
- 15 trailers under 15% matched, all original MovieNet links; Scarface (tt0086250) still at 0% [alignment CSVs]

**Story sampling** [`summary.json`]
- 34% of matched shots come from the film's first quarter, 16% from the last quarter
- median trailer takes 15% of its matched shots from the final act

**RQ1 table** [`model_report.json`, `film_shots.parquet`]
- 466,430 film shots from 293 films, 2.5% used in the trailer
- in-trailer film shots are longer on average: 6.4 s vs 4.6 s

**RQ1 model** [`model_report.json`]: LightGBM, 5 folds grouped by film, 64 CLIP PCs

| model | AP | AP per film | precision at trailer length | length-adjusted AP |
|---|---|---|---|---|
| random | 0.024 | 0.032 | 0.027 | 0.025 |
| position only | 0.028 | 0.042 | 0.045 | 0.028 |
| length only | 0.032 | 0.041 | 0.045 | 0.026 |
| CLIP only | 0.041 | 0.065 | 0.077 | 0.041 |
| readable | 0.045 | 0.067 | 0.083 | 0.041 |
| full (readable + CLIP) | 0.050 | 0.075 | 0.088 | 0.046 |
| full minus position | 0.048 | 0.073 | 0.087 | 0.044 |
| full minus dialogue | 0.050 | 0.074 | 0.090 | 0.046 |
| full minus scale | 0.051 | 0.076 | 0.093 | 0.047 |
| full minus content | 0.049 | 0.072 | 0.086 | 0.045 |
| full minus editing | 0.047 | 0.070 | 0.083 | 0.045 |
| full minus CLIP | 0.045 | 0.067 | 0.083 | 0.041 |

SHAP mean |value| by group (full model, log-odds): CLIP 0.33, position 0.21, editing 0.18,
content 0.10, dialogue 0.07, scale 0.05. Top single features: `story_pos` 0.16, `shot_len` 0.16,
`dlg_rate` 0.06.

Read: the full model ranks in-trailer shots about 2x better than random (AP 0.050 vs 0.024; 3.3x at
trailer length), but in absolute terms the signal is weak: at the trailer's own shot count, about 9
in 100 picks are right. Readable features get most of the way (0.045 vs 0.050). Dropping CLIP,
editing or position costs the most (AP down 0.0052, 0.0034, 0.0025); dropping dialogue or content
barely matters, and dropping scale slightly helps. Length-adjusted AP holds up (0.046 vs 0.050), and
length alone falls to random once length is controlled (0.026), so the gain is not just "long shots
get sampled". Neighbor-shot alignment errors don't explain the ceiling (see the tolerance eval below).

**Tolerance eval** [`model_eval.json`]: same out-of-fold predictions, a pick counts if a used shot is
within ±k film shots; random gets the same tolerance. Precision and lift are at trailer length, per film.

| model | prec k=0 | lift k=0 | prec ±1 | lift ±1 | prec ±3 | lift ±3 | prec ±5 | lift ±5 |
|---|---|---|---|---|---|---|---|---|
| random | 0.027 | 1.09 | 0.073 | 1.15 | 0.141 | 1.09 | 0.199 | 1.08 |
| position only | 0.045 | 1.82 | 0.107 | 1.68 | 0.196 | 1.53 | 0.260 | 1.42 |
| CLIP only | 0.077 | 3.11 | 0.131 | 2.05 | 0.212 | 1.65 | 0.275 | 1.50 |
| readable | 0.083 | 3.35 | 0.145 | 2.27 | 0.235 | 1.83 | 0.305 | 1.66 |
| full | 0.088 | 3.56 | 0.154 | 2.41 | 0.246 | 1.92 | 0.312 | 1.70 |

Base rate (share of shots within ±k of a used shot): 0.025, 0.064, 0.129, 0.184.

Read: lift is highest at the exact shot (full 3.6x) and falls as tolerance grows (about 1.9x at ±3),
so the misses are different moments, not neighboring shots. Position only has decent precision but
very low recall at ±3 (0.072 vs 0.167 for random, 0.236 for full) because it concentrates its picks in
one stretch of the film.

**Clustering** [`cluster_report.json`]

| features | k | silhouette | bootstrap ARI | genre NMI (null 95%) | decade NMI (null 95%) |
|---|---|---|---|---|---|
| all | 2 | 0.109 | 0.92 | 0.056 (0.022) | 0.027 (0.023) |
| pacing | 2 | 0.310 | 0.90 | 0.021 (0.025) | 0.078 (0.022) |
| audio | 2 | 0.260 | 0.92 | 0.017 (0.025) | 0.013 (0.022) |
| story | 3 | 0.235 | 0.76 | 0.037 (0.037) | 0.043 (0.036) |
| dialogue | 2 | 0.282 | 0.88 | 0.015 (0.025) | 0.018 (0.022) |
| visual | 2 | 0.231 | 0.94 | 0.051 (0.024) | 0.024 (0.021) |

HDBSCAN finds no clusters (all noise) on every set except dialogue (2 clusters, 51% noise).

Read: same picture as on 244. Stable but weakly separated. Genre beats the null only for all
features and visual (no shuffle out of 200 as high), and the NMI is small. Pacing clusters follow decade most clearly.

## 2026-10-28: full run, 244 trailers, CLIP ViT-B/32

**Corpus** [`summary.json`]
- 293 core films, 244 trailers downloadable (49 not: 43 unavailable, 6 other errors incl. 2 age-gated; recovery in progress) [`trailer_status.csv`]
- median trailer 132 s, 74.5 shots, 1.21 s per shot

**Alignment** [`summary.json`, `alignment_threshold.json`]
- 17,929 trailer shots, 1.8% blank
- threshold 0.741 at 5% false-match rate (52,806 impostor scores)
- 54% of trailer shots matched; per trailer median 58%, IQR 39% to 68% [alignment CSVs]
- 15 trailers under 15% matched; Scarface (tt0086250) at 0% [alignment CSVs]

**Story sampling** [`summary.json`]
- 34% of matched shots come from the film's first quarter, 16% from the last quarter
- median trailer takes 14% of its matched shots from the final act

**RQ1 table** [`film_shots.parquet`]
- 389,758 film shots from 244 films, 2.3% used in the trailer
- in-trailer film shots are longer on average: 6.4 s vs 4.6 s

**RQ1 model** [`model_report.json`]
- not run on this set

**Clustering** [`cluster_report.json`]

| features | k | silhouette | bootstrap ARI | genre NMI (null 95%) | decade NMI (null 95%) |
|---|---|---|---|---|---|
| all | 2 | 0.111 | 0.91 | 0.051 (0.027) | 0.026 (0.027) |
| pacing | 2 | 0.301 | 0.92 | 0.027 (0.027) | 0.070 (0.026) |
| audio | 2 | 0.279 | 0.87 | 0.026 (0.031) | 0.014 (0.027) |
| story | 3 | 0.243 | 0.78 | 0.042 (0.042) | 0.051 (0.037) |
| dialogue | 2 | 0.289 | 0.88 | 0.016 (0.027) | 0.014 (0.027) |
| visual | 2 | 0.232 | 0.93 | 0.041 (0.025) | 0.025 (0.027) |

HDBSCAN finds no clusters (all noise) on every set except dialogue (2 clusters, 55% noise).

Read: stable but weakly separated. Genre agreement is small even where it beats the null. Pacing
clusters follow decade (newer trailers cut faster).

## 2026-10-27: first run, 35 trailers

- 56% of trailer shots matched; threshold 0.749
- median trailer 134 s, 47 shots, 1.68 s per shot (older films were overrepresented)
- in-trailer film shots longer on average: 8.2 s vs 6.6 s
- clusters on 35 trailers: ARI 0.62 to 0.94; only all-features and pacing beat the genre null (p 0.015 and 0.01), by a small margin
