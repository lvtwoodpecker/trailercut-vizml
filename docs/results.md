# Results log

Numbers as of each run, newest first. Source files in brackets. Old entries stay, so we can see how
numbers moved as the data and methods changed.

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
