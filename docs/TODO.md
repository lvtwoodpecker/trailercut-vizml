# TODO

Owner in brackets. Move items to **Done** with the date when finished.

## Now (Nov 3 to 17)

- [ ] **[Cedric]** Check the 15 trailers with under 15% of shots matched (still 15 on 293, all original MovieNet links; Scarface tt0086250 is at 0%): wrong video, teaser, or re-release cut? Replace through `recover` if wrong
- [ ] **[Cedric]** Eyeball the recovered trailers with the lowest match share: *Basic* (31%), *Dogma* (33%), *Serenity* (34%)
- [ ] **[Cedric]** PR curve and ablation figures for the model
- [ ] **[Both]** Hand-label the 10 sampled trailers in `labels/label_tool.html` (both label the first 2 for kappa; separate browsers, since progress is one shared `localStorage` slot), save as `labels/gold_<name>.json`
- [ ] **[Cedric]** Run `label_eval`, pick the final match threshold, rerun everything downstream with it
- [ ] **[Estee]** Residual clustering: regress genre and decade out of the trailer features, re-cluster, same stability/NMI checks
- [ ] **[Estee]** Decide what the Overview shows if structure stays weak (continuous axes vs clusters)
- [ ] **[Estee]** D3 Overview skeleton, sharing the trailer JSON format and selection events with the Trailer View

## Next (Nov 17 to Dec 1)

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
