# Results log

Numbers as of each run, newest first. Source files in brackets. Old entries stay, so we can see how
numbers moved as the data and methods changed.

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
