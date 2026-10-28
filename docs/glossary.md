# Glossary

| term | meaning here |
|---|---|
| core set | the 293 MovieNet films with keyframes, synopsis-to-shot alignment and a trailer link |
| shot | a continuous run of frames between two cuts. Film shots come from MovieNet; trailer shots from TransNetV2 |
| keyframe | one of the 3 frames sampled per shot (MovieNet's for films; 25/50/75% for trailers) |
| story position | where a film shot falls in the film: shot midpoint over runtime, 0 (start) to 1 (end) |
| act | story position split 1/4, 1/2, 1/4 into setup, middle, ending |
| synopsis position | which paragraph of the MovieNet synopsis covers the shot, scaled 0 to 1 |
| alignment | matching each trailer shot to the film shot it was cut from |
| match / no match | a trailer shot is matched if its best score clears the threshold and it isn't blank |
| impostor null | best scores of trailer shots against *other* films; its 95th percentile is the threshold |
| FPR | false positive rate: share of shots from a different film that would still pass the threshold |
| blank shot | a shot that looks like a solid black, white or grey frame (fades, black frames) |
| margin | best score minus the best score outside ±3 neighboring film shots; low margin = ambiguous |
| trailerness | the RQ1 model's predicted probability that a film shot appears in the trailer |
| `in_trailer` | label: at least one matched trailer shot points at this film shot |
| readable features | features with names people understand (position, dialogue, scale, content, editing) |
| CLIP block | the shot embedding (PCA'd) as one feature group; SHAP shown as one "other appearance" value |
| AP / PR-AUC | average precision, area under the precision/recall curve; the main metric at 2.3% positives |
| precision at trailer length | per film, precision of the top-k shots where k is how many shots the trailer used (R-precision) |
| length-adjusted AP | AP inside shot-length quintiles, averaged; removes the "long shots get sampled more" effect |
| grouped CV | cross-validation where all shots of a film are in the same fold |
| ablation | refitting the model without one feature group to see how much it mattered |
| ARI | adjusted Rand index: agreement between two clusterings, 0 = chance, 1 = identical |
| NMI | normalized mutual information between clusters and a label (genre, decade) |
| permutation null | NMI with shuffled labels; its 95th percentile is the "beats chance" bar |
| final-act share | share of a trailer's matched shots from the last quarter of the film; our spoiler proxy |
| order correlation | rank correlation between trailer order and film order of matched shots |
| match share | fraction of a trailer's non-blank shots that are matched; used to verify recovered trailers |
| source | where a trailer link came from: `movienet` or `search` |
| backbone tag | which CLIP model made the embeddings: `b32`, `l14`, `h14` |
