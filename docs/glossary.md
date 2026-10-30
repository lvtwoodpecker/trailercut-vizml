# Glossary

| term | meaning here |
|---|---|
| core set | the 293 MovieNet films with keyframes, synopsis-to-shot alignment and a trailer link (`core` = 1 in `films_1100.csv`) |
| shot | a continuous run of frames between two cuts. Film shots come from MovieNet; trailer shots from TransNetV2 |
| keyframe | one of the 3 frames sampled per shot (MovieNet's for films; 25/50/75% for trailers) |
| story position | where a film shot falls in the film: shot midpoint over runtime, 0 (start) to 1 (end) |
| act | story position split 1/4, 1/2, 1/4 into setup, middle, ending (used as a feature; no longer labeled in the views) |
| synopsis position | which paragraph of the MovieNet synopsis covers the shot, scaled 0 to 1 |
| alignment | matching each trailer shot to the film shot it was cut from |
| match / no match | a trailer shot is matched if its best score clears the threshold and it isn't blank |
| best-F1 threshold | the match cutoff with the highest F1 (precision vs recall against the hand labels); 0.515, the one `align` uses now (0.535 before the swaps and re-split) |
| impostor null | best scores of trailer shots against *other* films; its 95th percentile was the threshold before hand labels (0.74) |
| smoothing | matching a whole trailer as a sequence (Viterbi) so consecutive shots prefer nearby film shots; tried, not used |
| FPR | false positive rate: share of impostor scores (best match in a different film) at or above the threshold |
| blank shot | a shot that looks like a solid black, white or grey frame (fades, black frames) |
| margin | best score minus the best score outside ±3 neighboring film shots; low margin = ambiguous |
| trailerness | the RQ1 model's predicted probability that a film shot appears in the trailer |
| `in_trailer` | label: at least one matched trailer shot points at this film shot |
| readable features | features with names people understand (position, dialogue, scale, content, editing) |
| CLIP block | the shot embedding (PCA'd) as one feature group; SHAP shown as one "other appearance" value |
| AP / PR-AUC | average precision, area under the precision/recall curve; used because only about 4% of shots are positive |
| precision at trailer length | per film, precision of the top-k shots where k is how many shots the trailer used (R-precision) |
| tolerance (±k shots) | in `model_eval`, a top-ranked film shot counts as a hit if a used shot is within k film shots of it; random gets the same tolerance |
| lift | a rate over the rate you'd get by chance. Model: precision at trailer length over the base rate, at the same tolerance. RQ1 page: pick rate of a kind of shot over the average shot's pick rate (1x = no difference) |
| edge shots | film shots in the first and last 1% of a film (studio logos, credits); left out of the RQ1 model and page |
| context features | `distinct` (unlike the film's average shot), `novelty` (unlike the 2 shots on each side), `recurrence` (the look comes back more than 10 shots away); from CLIP |
| relative features | `rel_*`: a feature's percentile rank within its own film |
| film bootstrap | resampling whole films with replacement to get an interval, so one film can't carry a result |
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
| movie barcode | the film drawn as a strip of its shots, each filled with the average color of its middle keyframe and as wide as its share of runtime (Trailer View) |
| speech segment | one Whisper segment of the trailer transcript, with word timestamps |
| found line | a trailer speech segment matched to the film's subtitles (3+ words, score 90+, margin 10+) |
| sync rate | of a trailer's found lines over a matched picture, the share whose film time is within 30 s of the picture's film shot; high = lines play over their own scene, low = voice-over or montage |
| sound link | in the Trailer View, the curve from the current spoken line to the film moment it comes from; one at a time |
| SSCD | a copy-detection image model (Pizzi et al. 2022); tried as a re-ranker for CLIP's candidates, not used |
| backbone tag | which CLIP model made the embeddings: `b32`, `l14`, `h14` |
| EBM | explainable boosting machine: a glass-box model that learns one curve per feature and adds them; its curves are the explanation (`model_ebm.py`) |
| shape plot | how one feature moves the odds of a shot making the trailer, as a multiplier on a log scale (EBM curve or binned SHAP) |
| trailer audit | rule-based check that a trailer is the original theatrical trailer in the original language (`trailer_audit.csv`) |
| hard / soft flag | audit flags. Hard (`not_trailer`, `ad`, `clip`, `dub`, `short`): swap the trailer. Soft (`rerelease`, `low_res`, `long`): swap only if a better candidate exists |
| scene clip | a YouTube upload of one scene posing as a trailer; it aligns well but only to a sliver of the film |
| span | 10th to 90th percentile of a trailer's matched story positions; under 0.2 means a scene clip |
| swap | replacing a working trailer with a better candidate (`swap.py`, `recover --ids`) |
| swap review | checking replaced trailers by eye on `labels/swap_review.html`, one film at a time: good, bad or unsure (`swap_review.py`, `curate.sh swap-review`) |
| keyframe consistency | the lowest cosine between a shot's 3 keyframes; low = the keyframes show different things |
| missed cut | a trailer shot whose keyframe consistency is below what 99% of MovieNet film shots reach; likely two shots TransNetV2 merged |
| re-split | splitting a likely missed cut once, at TransNetV2's most likely cut frame inside the shot (`resplit.py`) |
| glued shot | a trailer shot still below the keyframe consistency floor after re-split |
| split review | checking a sample of CLIP's proposed cuts inside glued shots by eye on `labels/split_review.html`: real cut, same shot or unsure (`resplit_clip.py`, `curate.sh split-review`) |
| batch apply | `swap apply --batch`: swap checks in rounds, one shot detection and CLIP pass per round over all undecided films |
| expand set | films new to the dataset (not in the 293 core set), handled by `swap --set expand` |
