# Results log

Numbers as of each run, newest first. Source files in brackets. Old entries stay, so we can see how
numbers moved as the data and methods changed.

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
- pending (first CV run in progress)

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
