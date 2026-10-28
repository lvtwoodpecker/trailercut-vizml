# Design choices

Why the pipeline is built the way it is. Each entry: what we do, why, and what we gave up.
Code paths are under `src/trailercut/`. Numbers are from the 244-trailer run with CLIP ViT-B/32.

## Data

**Core set = 293 films.** Films with MovieNet keyframes, synopsis-to-shot alignment and a trailer link.
The synopsis alignment is what lets us place a shot in the story (act, synopsis paragraph), so films
without it are left out even though ~800 have keyframes and a trailer. Expanding to those is possible
on the cluster but loses the story features.

**Store IDs and features, not media.** The repo has YouTube IDs, an availability log
(`data/ids/trailer_status.csv`) and extracted features. Videos, keyframes and embeddings are
gitignored and rebuilt by `scripts/run_all.sh`. Keeps the repo small and avoids redistributing
copyrighted footage. Large derived tables that rebuild in about a minute (`film_shots.parquet`,
model predictions and SHAP) are also gitignored.

**Trailers at 360p.** CLIP sees 224 px and TransNetV2 sees 48x27, so higher resolution only costs disk.

**Dead links are replaced only if they align** (`recover.py`). Search YouTube for "Title (year)
trailer", drop fan edits, reviews, clips and anything outside 45 to 300 s, then accept a candidate
only if at least 25% of its shots match that film's keyframes. Title filtering alone isn't enough:
the 2nd search hit for *The Godfather Part II* is the *Godfather* anniversary trailer. Real trailers
have a median match share of ~58% and a wrong film should sit near the 5% false-match rate, so 25%
separates them with room on both sides. Every replaced row has `source=search`, since searched
trailers can be re-release cuts and we may want to check whether they behave differently.

**Keyframes as one zip per film** (`movienet.py --pack`). ~880 files instead of ~1.3M JPEGs, which
matters for inode quotas on the cluster. Readers accept either a folder or `path.zip::member` refs.

**Year fixes.** MovieNet titles for 7 films (the Star Wars films, *Spider-Man*) have no year; they
are hardcoded in `movienet_io.YEAR_FIX` so the decade analysis doesn't drop them.

## Trailer processing

**TransNetV2 for trailer shots.** Trailers cut fast (median shot 1.2 s) and use flashes and quick
dissolves that simple histogram detectors miss. MovieNet film shots come with the dataset, so only
trailers need detection.

**3 keyframes per shot at 25/50/75%.** Matches MovieNet's 3 keyframes per film shot, so trailer and
film shots are compared the same way. Avoids the first/last frames, which often sit on a transition.

**Per-trailer letterbox crop** (`embed.bar_box`). Many YouTube trailers are boxed and MovieNet
keyframes aren't; the black bars pull similarity down. Bars are found from the per-pixel max over all
of a trailer's frames: real bars stay black in every frame, dark scenes don't. Cropping per frame
was tried first and hurt dark films (*The Godfather* 0.65 to 0.60); per-trailer crop moved *Rebel
Without a Cause* from 32% to 43% matched with no loss elsewhere.

## CLIP

**ViT-B/32 locally, ViT-H/14 on the cluster.** B/32 runs at ~257 images/s on a laptop GPU vs ~15 for
L/14, which made iterating on alignment possible. H/14 is the backbone Argaw et al. use, so final
numbers will be rerun with it (`TRAILERCUT_CLIP=h14`). Caches are keyed by backbone tag so runs never mix.

## Alignment (trailer shot to film shot)

**Shot similarity = mean over trailer keyframes of the max over film keyframes.** Each trailer frame
finds its best match among the film shot's 3 frames; averaging makes one lucky frame not enough.

**Explicit no-match.** A trailer shot only counts as matched above a threshold. Trailers contain
footage that isn't in the film (title cards, alternate takes, VFX shots, other films' logos), so
forcing every shot onto a film shot would create false labels.

**Threshold from an impostor null.** Score each trailer shot against 3 *other* films; the 95th
percentile of those best scores is the threshold (0.741), i.e. ~5% of non-source shots would pass.
This gives a false-match rate without labels. 1% (0.786 to 0.81) was tried first, but contact sheets
showed matches between 0.72 and 0.79 were almost all correct, so 1% threw away real matches.

**Blank masking.** Black frames and fades match blank frames in *any* film and pushed the 1%
threshold to 0.9975. Shots that look like a solid black/white/grey frame (CLIP similarity > 0.9) are
flagged on both sides: blank trailer shots are never matched and blank film shots are never candidates.

**Top-5 and margin are kept.** The null can't see the main real error, "right scene, wrong shot"
(shot/reverse-shot, coverage of the same moment). Top-5 candidates feed the hand-labeling tool, and
the margin to the best shot outside ±3 neighbors is saved as a confidence signal.

**Hand labels pick the final threshold** (`label_tool.py`, `label_eval.py`). 10 trailers sampled across
decades, each shot labeled against its top-5 candidates, with 2 trailers labeled by both of us for
Cohen's kappa. The final cutoff comes from that precision/recall curve, not the null.

## Readable features

**Zero-shot concepts instead of raw embedding dimensions** (`concepts.py`). Shot scale uses MovieNet's
5 classes (ECU, CU, MS, FS, LS) as CLIP text prompts, plus a single `scale_width` summary; content uses
20 named concepts (explosion, gun, kiss, crowd, night, text card, ...). SHAP on these reads as
"close-ups of guns", which SHAP on embedding dimension 317 never can.

**Dialogue from film subtitles.** VADER sentiment and word rate of subtitles overlapping each film
shot. For trailer-level dialogue this is a proxy (it reads what the film says under the matched shots,
not what the trailer plays); Whisper on trailer audio would replace it.

**`visual_change` = 1 - cos(first, last keyframe).** A cheap stand-in for motion without optical flow.

**Act = classic 1/4, 1/2, 1/4 split of story position.** Simple and explainable; synopsis paragraph
position carries the finer story signal.

## RQ1 model (`model.py`)

**Label: `in_trailer` = at least one matched trailer shot points at this film shot.** 2.3% positive.

**LightGBM.** Handles mixed-scale features, missing values (no subtitles, no synopsis paragraph) and
390K rows quickly, and TreeSHAP is exact and fast for it.

**Folds grouped by film.** Shots from the same film share look, cast and score; a random split would
let the model memorize films and inflate every metric.

**Metrics.** Average precision (PR-AUC) instead of accuracy or ROC-AUC because of the 2.3% positive
rate. Pooled AP, mean per-film AP, and precision at trailer length (per film, top-k where k is the
number of shots the trailer used), which is the question a trailer editor actually faces.

**Baselines.** Random, position only, shot length only and CLIP only, so every gain is relative to
something simple.

**Shot length confound.** In-trailer shots are longer on average, partly because a longer shot gives
the editor more footage to sample, not because it's more "trailer-worthy". Reported as the
length-only baseline plus `ap_lenadj`, AP computed within shot-length quintiles and averaged.

**CLIP as one block.** The shot embedding is compressed with PCA (64 components, fit without labels)
and added as a group. Its SHAP values are summed into one "other appearance (CLIP)" value. The
readable-only vs readable+CLIP gap shows how much of trailerness our named features miss.

**Ablations drop one group at a time from the full model** and report the metric drop.

## RQ2 clustering (`cluster.py`)

**Features per trailer in 5 groups:** pacing, audio, story sampling, dialogue, visual. Clustered on
all features and on each group alone, so we can tell which modality drives any structure.

**k-means with k = 2..8 by silhouette, checked against HDBSCAN.** k-means for interpretable centroids;
HDBSCAN as a check on whether density-based structure exists at all.

**Stability:** refit on 50 bootstrap 80% subsamples and compare to the full-data labels with ARI.

**Genre/era with a baseline:** NMI with primary genre and decade, compared against the 95th percentile
of NMI with shuffled labels. Without the null, any NMI looks like agreement.

**Current read:** clusters are stable (ARI 0.78 to 0.93) but weakly separated (silhouette 0.11 to
0.30). Genre NMI is near the null; pacing clusters follow decade. Next step: regress out genre and
decade and re-cluster the residuals; if structure stays weak, report continuous axes instead of types.

## Dropped from the proposal

- **0.9 alignment accuracy on augmented film shots.** It measured robustness to crops and color
  grading, not correctness on real trailers. Replaced by hand labels.
- **Spoiler stretch goals.** Final-act share is the only spoiler measure and is labeled as a proxy.
