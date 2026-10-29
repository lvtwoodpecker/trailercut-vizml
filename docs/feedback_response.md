# Response to proposal feedback

Longer version of the "Response to Feedback" section in the project update. Numbers come from
`data/features/summary.json`, `alignment_eval.json` and `cluster_report.json`.

## 1. Label quality

Agreed, and the first pipeline run proved the point.

- **No-match option.** Every trailer shot gets a match score, and it only counts as matched above a threshold. The starting threshold comes from an impostor null: the same trailer shot scored against a *different* film. That gives a label-free false-positive rate.
- **What broke first.** At 1% FPR the threshold was 0.9975 and almost nothing matched. The cause was black/fade frames, which match blank frames in any film. We now flag blank shots on both sides (CLIP similarity to solid frames) and never match them.
- **Letterboxing.** Many YouTube trailers are boxed while MovieNet keyframes are not. We crop bars per trailer, using the max over all frames so dark scenes don't get cropped. On *Rebel Without a Cause* this moved matched shots from 32% to 43%, with no change on unboxed trailers.
- **What the impostor null can't see.** Contact sheets showed matches at 0.72 to 0.79 were almost all correct, so 1% FPR was too strict (we use 5% for now). The real error mode is "right scene, wrong shot" (alternate angle, or a shot MovieNet keyframes miss), and an other-film null can't measure that.
- **Hand labels.** So we built `labels/label_tool.html`: 10 trailers sampled across decades, every shot shown with its top-5 film candidates, labeled as one of the candidates, *same scene/other shot*, *not in film*, or *unsure*. Both of us label 2 shared trailers for Cohen's kappa. The final threshold is picked from the precision/recall curve (`label_eval`), not the null.
- **Note.** The 0.9 augmentation target from the proposal is dropped. It tested robustness to crops and grading, not correctness on real trailers.

## 2. Presenting attributions

The model's visual features are now readable by construction:

- **shot scale**: zero-shot CLIP probabilities over MovieNet's 5 scale classes, plus a single "width" score;
- **content**: 20 named concepts (explosion, gun, kiss, crowd, night, text card, ...) scored zero-shot;
- **position** (story position, act, synopsis paragraph), **dialogue** (words, rate, VADER sentiment from subtitles), **editing** (shot length, within-shot visual change).

The raw CLIP embedding is one separate block. In the Trailer View each film shot gets a stacked bar of *grouped* SHAP (position / dialogue / scale / content / editing / "other appearance (CLIP)"), and clicking a group expands it into its named features. The CLIP block never gets broken into dimensions. We report the readable-only model next to readable + CLIP, so the AUPRC gap shows how much of trailerness our vocabulary misses.

Known confound: in-trailer film shots are longer on average (8.2 s vs 6.6 s in the first 35 films), partly just because longer shots give a trailer more footage to sample. We'll control for it (length as a covariate, plus a length-matched evaluation) instead of reporting it as an editorial preference.

## 3. Two projects in one

We agree. **RQ1 + the Trailer View are the core.** RQ2 becomes supporting context:

- one clustering setup (k-means, checked against HDBSCAN) feeding the Overview; per-modality results stay as a table, not separate views;
- story-sampling features from RQ1's alignment (final-act share, order correlation) are the bridge between the two;
- spoiler stretch goals S1/S2 are dropped. "Final-act share" is our only spoiler measure, and we call it a proxy.

## 4. Clustering evaluation

- **Stability:** k-means refit on 50 bootstrap 80% subsamples, ARI against the full-data labels.
- **Genre/era agreement against a baseline:** NMI with primary genre and decade, compared to the 95th percentile of NMI with shuffled labels, so "agrees with genre" has a null.
- **What we'd conclude:** if clusters basically reproduce genre/era, the finding is that trailer style is a genre convention. We'd then regress genre and decade out of the features and re-cluster the residuals to see if any structure is left. If nothing is stable (low ARI), we report continuous axes (PCA loadings) instead of clusters.

## 5. Trailer data

Done. The repo stores YouTube IDs, an availability log (`data/ids/trailer_status.csv`), download scripts and extracted features. Videos, keyframes and embeddings are gitignored and rebuilt by `scripts/run_all.sh`.

## 6. Who does what

- **Cedric:** data pipeline (MovieNet download, trailer scraping and recovery, shot detection, CLIP, alignment, cluster jobs), RQ1 trailerness model and SHAP, Trailer View.
- **Estee:** RQ2 clustering (feature selection, k-means/HDBSCAN, stability and genre/era baselines, residual clustering), cluster Overview in D3 and its link to the Trailer View, case study selection.
- **Both:** hand labeling (shared trailers for agreement), report and presentation.
