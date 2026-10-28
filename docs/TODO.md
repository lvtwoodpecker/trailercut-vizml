# TODO

Owner in brackets. Move items to **Done** with the date when finished.

## Now (Nov 3 to 17)

- [ ] **[Cedric]** Finish dead-link recovery (`trailercut.recover`), rerun align → features → model → cluster → figures on the bigger set
- [ ] **[Cedric]** Check the 15 trailers with under 15% of shots matched (Scarface tt0086250 is at 0%): wrong video, teaser, or re-release cut? Replace through `recover` if wrong
- [ ] **[Cedric]** RQ1 model: read the first CV results, write them into `docs/results.md`
- [ ] **[Cedric]** PR curve and ablation figures for the model
- [ ] **[Both]** Hand-label the 10 sampled trailers in `labels/label_tool.html` (2 shared for kappa), save as `labels/gold_<name>.json`
- [ ] **[Cedric]** Run `label_eval`, pick the final match threshold, rerun everything downstream with it
- [ ] **[Estee]** Residual clustering: regress genre and decade out of the trailer features, re-cluster, same stability/NMI checks
- [ ] **[Estee]** Decide what the Overview shows if structure stays weak (continuous axes vs clusters)
- [ ] **[Cedric]** D3 Trailer View skeleton: export one JSON per trailer, static page with cut timeline, audio curve, shot-to-film links
- [ ] **[Estee]** D3 Overview skeleton, sharing the trailer JSON format and selection events with the Trailer View

## Next (Nov 17 to Dec 1)

- [ ] **[Cedric]** Trailer View: SHAP bars per film shot (grouped, click to expand), keyframe on hover
- [ ] **[Estee]** Overview linked to the Trailer View
- [ ] **[Estee]** Pick case studies (use the model's misses and the cluster extremes)
- [ ] **[Cedric]** ViT-H/14 rerun on the cluster (`hpc/`), compare alignment and model numbers to B/32
- [ ] **[Cedric]** Whisper on trailer audio to replace the subtitle dialogue proxy (cluster GPU)
- [ ] **[Cedric]** Temporal smoothing in alignment (neighboring trailer shots prefer nearby film shots)

## Later (Dec 1 to 15)

- [ ] **[Both]** Final ablations and evaluation tables
- [ ] **[Both]** Report and presentation

## Maybe

- Expand beyond the 293 core films (~800 MovieNet films have keyframes + trailer, but no synopsis alignment)
- Letterboxd list as a filter for case studies
- Film mode in `trailer_prep` for user-owned films outside MovieNet

## Done

- 2026-10-28: full pipeline on all 244 available trailers
- 2026-10-28: RQ1 model module with grouped CV, baselines, ablations, out-of-fold SHAP
- 2026-10-28: dead-link recovery module
- 2026-10-27: Slurm setup for the cluster (`hpc/`)
- 2026-10-27: labeling tool and label evaluation
- 2026-10-27: scraping, shot detection, CLIP embeddings, alignment, initial clustering
