# Course learnings

Notes from the VisML lectures and labs ([course site](https://ctsilva.github.io/2026-VisML-CDS/)), one
section per lecture: what it said, then what it means for this project. Suggestions, not rules; when we
go against one on purpose, the reason is in [design.md](design.md) or [viz_design.md](viz_design.md).

✅ we do it · 🔧 fixed after reading the slides (2026-10-30) · ⬜ still open

---

## Intro to infovis

- Vis = interactive visual representations of abstract data "to amplify cognition"; moves work from
  memory into the eye.
- Expressiveness: ordered data must look ordered, unordered must not.
- Effectiveness: the most important thing gets the best channel.
- Zero baseline for ratio comparisons; fine to drop for intervals.
- One view answers only some questions; interaction covers the rest.

For us
- ✅ RQ1 lifts on one shared position axis; model lift bars start at 0×; dot plots are log scale with a
  1× line instead of bars.
- ✅ Overview → detail: RQ1 page → Trailer View → "now" panel → shots table.

## Perception

- Channel ranking (Cleveland & McGill): position on a common scale > unaligned position > length >
  angle > area > volume > shading/saturation. Don't expect precision from low-rank channels.
- Pop-out needs one unshared feature; a conjunction of two forces a scan.
- Working memory holds ~1 to 3 objects.
- Grouping strength: enclosure > connection > similarity ≈ proximity. Color always claims a group.
- ~5 to 6 distinguishable levels for hue or shape.
- Stevens: area and brightness are compressed, saturation exaggerated.

For us
- ✅ Current shot = outline + ink ring (one feature), not a color mix.
- ✅ 3 hues + grey; no numbers in area or saturation.
- 🔧 "Why this shot" bars cut from 6 groups to the top 3 (the rest behind "more").

## Color

- Quantify with luminance (one hue, light → dark); label with hue at similar luminance. Never cross them.
- Rainbow has no order and invents bands.
- Diverging only around a meaningful midpoint; blue-orange over red-green.
- "No data" goes outside the scale, never as its low end.
- Simulate color blindness before shipping; keep a luminance difference or a second channel.
- Color only works where the eye looks; small blue text is hard to read; test on the projector.
- Lab: contrast 4.5:1 for text, 3:1 for graphics; categorical palettes ≤ ~7.

For us
- 🔧 Match-score scatter used viridis (multi-hue) for a magnitude → single-hue blue, with a colorbar.
- 🔧 Blue (picture) vs violet (dialogue) collapsed under red-green simulation (ΔE 1.9) → dialogue is
  green, and its link is dashed.
- 🔧 "No match" was a faint grey block → left empty.
- 🔧 Small grey text 3.5:1 → 5.0:1.
- ✅ Diverging blue/red around 1×; grey = interval crosses 1×.
- ⬜ Look at both pages and the figures on the projector, in light mode.

## Model assessment

- Accuracy alone hides which errors; always compare with a trivial baseline and say what it is.
- PR chance is a horizontal line at the positive rate, not the diagonal.
- A single number hides curve crossings; summaries rank, curves pick operating points.
- Calibration is a separate property; don't read scores as probabilities without it.
- Keep instances reachable from every aggregate; look at which errors, not how many.
- Evaluate the visualization too.

For us
- ✅ Lift vs random picks; PR curves and the ±k tolerance figure next to the single lift number.
- ✅ Scores are only used as ranks (top-k per film), never shown as a chance.
- 🔧 `model_pr.png` now draws the chance line at the positive rate.
- ⬜ Error slices: which films, genres or decades the model misses, with significance.
- ⬜ RQ1 page rows (e.g. "explosion") don't link to example shots yet.
- ⬜ No user evaluation of the views yet (3 to 5 people, timed questions).

## White-box models (GAM, EBM)

- Interpretable = a person can predict the model's output; predicting is not endorsing.
- Good explanations are contrastive, selective (1 to 3 reasons), and point at the unusual.
- Shape curves are log-odds, centered on the average; pair every curve with an uncertainty band and its
  data density. Wiggles where data is thin are noise.
- Smoothness is a tuning choice; check the story at another setting.
- Interpretable models are often as accurate as boosted trees: fit an EBM early.
- Surrogates must report fidelity; simple-looking ones fail silently.

For us
- ✅ Curves shown as odds multipliers vs the average shot; EBM lift shown next to LightGBM's.
- 🔧 EBM curves had no band and no density → band (spread over the bags) + shots-per-bin strip (shows
  after the EBM rerun).
- ⬜ EBM at a second smoothness; keep only the stories that survive.

## Black-box explanations (SHAP, LIME, PDP)

- A SHAP value is a contribution to this prediction's distance from the average prediction; the
  baseline matters and should be stated.
- Associations, not causes. Don't hand raw SHAP values to end users.
- Correlated features: marginal sampling builds impossible points; credit gets split between partners.
- Check stability (seeds, folds) before trusting an explanation.
- PDP averages can hide opposite effects (ICE shows them).

For us
- 🔧 "+0.69" (log-odds) → "×1.99" (odds vs the average shot); the panel is titled "What moved the
  model's score, vs the average shot", not "Why"; the RQ1 page says "associations, not causes".
- 🔧 Correlated groups (CLIP vs content concepts, story position vs act) noted in limitations.md.
- ⬜ Grouped SHAP across folds: are the top reasons stable?

## Clustering and dimensionality reduction (reference for the clustering part)

- Start from a single-color scatter; show the criterion curve (elbow, silhouette), not just the k.
- Criteria that disagree are a finding; test stability across seeds and settings.
- t-SNE/UMAP: don't read cluster size, density or gaps; compare with PCA and with noise.
- Put method, parameters, seed and preprocessing in the caption.
- Use the projection to ask questions of the data, not as evidence.

## Deep learning and NLP

- Pair a parameter view with its meaning in data space; ask what a picture leaves out.
- 2-D projections of embeddings are lossy; separation there isn't separation in the full space.
- Inspect individual documents next to aggregates.

For us
- ✅ Transcripts shown as captions on the trailer itself; aggregates link to single trailers.

## Project discussion

- Start from a question whose answer could come out the other way.
- Visualize intermediate results ("if you can't, you don't understand them yet").
- A modest, well-evaluated, well-communicated project beats an ambitious one.
- Evaluation: metrics, a baseline, or a 3 to 5 person study.

For us
- ✅ The review pages (labels, swaps, splits, audit) are the intermediate results, visualized.
- ⬜ The small user study (see Model assessment).

## Writing the paper (lab)

- The paper is an argument; each figure is a piece of evidence.
- Abstract with numbers; contributions as specific, checkable bullets at the end of the intro.
- Related work by idea, not one paper per paragraph; background = what you need for *our* work.
- Design goals = what users can do (not features); every view serves at least one.
- Figures stand alone; the caption tells the story; no plain screenshots, annotate them.

For us
- ✅ Every method choice has its "why" in design.md and viz_design.md.
- ⬜ Abstract numbers (lift 3.30× on 1,058 films; alignment P 0.87 / R 0.92), contribution bullets,
  design goals mapped to views, annotated figures (`?t=` links make Trailer View shots reproducible).

---

## Open items

- [ ] Projector check of both pages and the figures
- [ ] Error slices: where the model misses
- [ ] RQ1 page: link each level to example shots in the Trailer View
- [ ] Small user evaluation (3 to 5 people, timed questions)
- [ ] EBM at a second smoothness
- [ ] Grouped SHAP stability across folds
- [ ] Report: numbers in the abstract, contribution bullets, design goals → views, annotated figures
