# Course guidelines

What the course lectures and labs ([VisML course site](https://ctsilva.github.io/2026-VisML-CDS/)) ask
of visualization-for-ML work, and where this project stands on each. These are suggestions, not hard
rules: when we break one on purpose, the reason goes in [design.md](design.md) or
[viz_design.md](viz_design.md). Our own viz design system (tokens, marks, checklist) is in
`viz_design.md`; this file is the outside yardstick.

Status: **ok** = we follow it · **fixed** = fixed on 2026-10-30 after this review · **open** = not yet
(listed again at the end).

## Perception and encoding

| guideline | where it applies | status |
|---|---|---|
| Expressiveness: ordered data must look ordered, unordered data must not | lift levels (ordered bins) on a position scale; concepts sorted by lift, not colored by it | ok |
| Effectiveness: the most important quantity gets the best channel (position on a common scale) | RQ1 lifts on one shared log axis; match position on the timeline | ok |
| Zero baseline for ratio comparisons | model lift bars start at 0×; dot plots are log scale with a 1× reference instead of bars | ok |
| Overview first, details on demand | RQ1 page → Trailer View; timeline → "now" panel → shots table | ok |
| Working memory holds 1 to 3 objects | "now" panel compares two frames; "why" bars cut to the top 3 reasons | fixed |
| Grouping: enclosure > connection > similarity/proximity; color always claims a group | one color per role, never per row; links connect a trailer shot to its film spot | ok |
| A single target pops out only on one feature, not a conjunction | current shot: outline + ink ring, not a color mix | ok |
| Don't expect precision from area, saturation or brightness | no area or saturation encodings of numbers | ok |
| About 5 to 6 levels per channel at most | 3 categorical hues + grey | ok |

## Color

| guideline | where it applies | status |
|---|---|---|
| Hue labels, luminance quantifies; never cross them | match-score scatter used viridis (multi-hue) for a magnitude; now a single-hue blue ramp | fixed |
| A color scale needs a legend or colorbar | match-score scatter had no colorbar | fixed |
| Never red vs green alone; prefer blue-orange-like pairs for diverging | blue/red for direction, checked under colorblind simulation | ok |
| Simulate color vision deficiency before shipping | blue vs violet (picture vs dialogue) collapsed to ΔE 1.9; dialogue is green now | fixed |
| Keep a luminance difference or a second channel | dialogue link dashed vs solid picture links; table under every chart | fixed |
| "No data" sits outside the scale, never as its low end | "no match" is an empty cell, not a faint grey | fixed |
| Contrast: 4.5:1 for normal text, 3:1 for graphics | small grey text was 3.5:1 in light mode, now 5.0:1 | fixed |
| Diverging only around a meaningful midpoint | 1× (as often as the average shot) | ok |
| Avoid small blue text; color only works where the eye looks | status text ("matched", "no match") sits next to the video, not only as a colored cell | ok |
| Test on the projector, not just a laptop | before the presentation, in light mode | open |

## Model assessment

| guideline | where it applies | status |
|---|---|---|
| Always say what the baseline is | lift vs random picks; "vs the average shot" on every explanation | fixed |
| PR chance is the positive rate, not the diagonal | `model_pr.png` now draws the chance line at the positive rate | fixed |
| A single number hides crossings; show the curve | PR curves and the ±k tolerance figure next to the lift | ok |
| Don't present scores as probabilities without calibration | scores are only used as ranks (top-k per film); never shown as a chance | ok |
| Look at which errors, not how many; find failing slices | not done: which films or genres the model misses | open |
| Keep instances reachable from aggregates | Trailer View links shot → frames → film; RQ1 page levels don't link to example shots | open |
| Evaluate the visualization itself | no user study yet (3 to 5 people, timed tasks) | open |

## Interpretable models (EBM, GAM)

| guideline | where it applies | status |
|---|---|---|
| Pair every shape curve with an uncertainty band and its data density | EBM curves had neither; now a band (spread over the bags) and a shots-per-bin strip (shows after the EBM rerun) | fixed |
| Classification curves are log-odds, centered on the average | shown as odds multipliers vs the average shot | ok |
| Wiggles where data is sparse are noise | x cut to the 1st to 99th percentile; the strip shows support | fixed |
| Show the curve at more than one smoothness before telling a story | one setting only | open |
| An interpretable model need not be weak; compare with the black box | EBM lift shown next to LightGBM's | ok |

## Explanations (SHAP and friends)

| guideline | where it applies | status |
|---|---|---|
| Selective: 1 to 3 reasons, more is worse | "why" panel showed all 6 groups; now top 3, the rest behind "more" | fixed |
| Don't hand raw SHAP values to end users | "+0.69" (log-odds) became "×1.99" (odds vs the average shot) | fixed |
| Associations, not causes | the panel is "What moved the model's score", not "Why"; the RQ1 page says so | fixed |
| Explanations are contrastive against a baseline; state it | "vs the average shot" | fixed |
| Correlated features split credit between them | CLIP vs content concepts, story position vs act: say so when reading attributions | open (limitations) |
| Check explanation stability | grouped SHAP not compared across folds or seeds | open |

## Writing the report (paper-writing lab)

| guideline | status |
|---|---|
| Figures stand alone; the caption tells the story; no plain screenshots (annotate them) | open: `?t=` links make annotated Trailer View screenshots reproducible |
| Contributions as specific, checkable bullets at the end of the introduction | open |
| Abstract with numbers (lift 3.30× on 1,058 films, P 0.87 / R 0.92 alignment) | open |
| Design goals as a list of what users can do (not features), and every view tied to at least one | open |
| Related work by idea, not one paper per paragraph | open |
| Methods justify each choice (the "why" is in design.md and viz_design.md) | ok |
| A modest, well-evaluated project beats an ambitious one that doesn't work | ok |

## Projections and clustering (for the clustering part of the project)

Reference only, from the clustering and dimensionality-reduction decks: start from an unlabeled
single-color scatter; show the criterion curve (elbow, silhouette), not just the chosen k; say when
criteria disagree; check stability across seeds and settings; don't read cluster size, density or gaps
from t-SNE/UMAP; compare with PCA and with noise of the same size; report method, parameters, seed and
preprocessing in the caption; use the projection to ask questions of the original data, not as evidence.

## Open items

1. Projector check of both pages and the report figures, in light mode.
2. Error slices: where the model misses (genres, decades, trailer styles), with significance.
3. RQ1 page: link each level (e.g. "explosion") to example shots in the Trailer View (`?id=…&t=…`).
4. A small evaluation of the views: 3 to 5 people, a few timed questions (e.g. "does this trailer
   follow film order?", "which film moment does this line come from?").
5. EBM curves at a second smoothness; keep only the stories that survive.
6. Grouped SHAP across folds: are the top reasons stable?
7. Report: standalone annotated figures, checkable contributions, numbers in the abstract, design
   goals mapped to views.
