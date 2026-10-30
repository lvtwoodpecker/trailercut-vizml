# Design choices

Why the pipeline is built the way it is. Each entry: what we do, why, and what we gave up.
Code paths are under `src/trailercut/`. Numbers are from the 293-trailer run with CLIP ViT-B/32 unless an entry says otherwise.

## Data

**Core set = 293 films.** Films with MovieNet keyframes, synopsis-to-shot alignment and a trailer link.
The synopsis alignment places a shot in the story (synopsis paragraph).

**Film list = all 1,100 keyframe films** (`data/ids/films_1100.csv`, 993 with a MovieNet trailer link;
`paths.IDS`, `TRAILERCUT_FILMS` overrides). Time position in the film covers what `synopsis_pos` adds,
so new films get no synopsis feature (NaN) and we don't build an automatic synopsis alignment for them.
The 293 core films keep a `core` flag. `trailers` only downloads films that have a MovieNet link;
films without one get a trailer through `swap --set expand`. The data funnel figure now counts
keyframe films, then films with a MovieNet trailer link, then films with a working trailer (MovieNet's
or a replacement), then aligned trailers; the synopsis step is gone since it no longer filters films.

**Store IDs and features, not media.** The repo has YouTube IDs, an availability log
(`data/ids/trailer_status.csv`) and extracted features. Videos, keyframes and embeddings are
gitignored and rebuilt by `scripts/run_all.sh`. Keeps the repo small and avoids redistributing
copyrighted footage. Large derived tables that rebuild in about a minute (`film_shots.parquet`,
model predictions and SHAP) are also gitignored.

**Trailers at 360p.** CLIP sees 224 px and TransNetV2 sees 48x27, so higher resolution only costs disk.

**Dead links are replaced only if they align** (`recover.py`). Search YouTube for "Title (year)
official trailer", drop fan edits, reviews, clips, dubs (title in another language), subtitled
uploads (sub, subs, subtitled, untertitel, sottotitol, Korean and Chinese subtitle words), teasers, TV
spots, re-releases and anything outside 45 to 300 s, try official/theatrical titles and 90 to 180 s
videos first, then accept a candidate only if at least 25% of its shots match that film's keyframes. Title filtering alone isn't enough:
the 2nd search hit for *The Godfather Part II* is the *Godfather* anniversary trailer. Real trailers
have a median match share of ~58% and a wrong film should sit near the 5% false-match rate, so 25%
separates them with room on both sides. This needs the 5% impostor cutoff, not the label-picked one
(at 0.515, 83% of impostor scores still clear it), so `align` always saves
`impostor_threshold` next to the cutoff in use and `recover` reads that. Every replaced row has `source=search`, since searched
trailers can be re-release cuts and we may want to check whether they behave differently.
The film's own title is removed before the title filter runs: *Freddy vs. Jason* got zero candidates
because `vs.` tripped it. YouTube is asked for 20 results (it was 10), since the filter throws many out.

**A replacement must also cover the film** (`--min-span 0.2`). A scene clip of the right film aligns
perfectly but only to one stretch of it, so the 25% check passes it. The matched shots' story
positions (10th to 90th percentile) must span at least 20% of the film. The four MovieNet "trailers"
that were scene clips spanned 1% to 3%; every other trailer spans at least 30% (median 71%,
`trailer_audit.csv`), so 20% sits between them.

**`recover --ids` replaces a working trailer.** For a link that plays but is the wrong kind of video
(Scarface's was a VHS-quality TV spot with 0% of shots matched at the impostor cutoff). The old link is
never re-picked, its files are moved to `cache/replaced/{id}/` first, and they are put back if no
candidate passes the 25% check, so a failed search can't leave a film without a trailer.

**Trailer audit: rules only, hard vs soft flags** (`trailer_audit.py`). We want the film's original
theatrical trailer in its original language. Each trailer gets flags from its YouTube metadata
(fetched without downloading), its transcript and its alignment, all from fixed rules so the audit
reruns the same way. Hard flags mean the video isn't what we study and should be swapped: not a
trailer by its title, a home-video ad, a scene clip, a dub, a subtitled or foreign-market upload by
its title (subtitulado, legendado, VOSTFR, Sub.KOR, español, ...), or under 60 s (TV spots, teasers). Soft
flags (re-release title, under 360p, over 240 s) mean swap only if a better candidate exists. A dub
needs Whisper to be confident (p 0.8+) over 15+ s of speech. What we give up: rules can miss what a
person would catch (a re-cut with a clean title). The cutoffs are judgment calls, not tuned: theatrical
trailers mostly run 1.5 to 3 minutes, so 240 s leaves room for long ones while catching extended
previews and featurettes (as a soft flag, since some official trailers do run longer). For dubs,
Whisper's language guess is unreliable on a few words or on music, so a trailer is only flagged when
Whisper is at least 80% sure and has 15+ s of speech to go on. The `subtitled` flag came from
*12 Years a Slave*, whose link was the English trailer with Spanish subtitles: Whisper hears English,
so the `dub` flag can't catch burned-in subtitles. Six core titles had it (12 Years a Slave, Big, The
Usual Suspects, Star Trek, The Avengers, The Ides of March); Twelve Monkeys' is a real Spanish dub.

**Swaps in three steps: search, slow download, apply** (`swap.py`). Swapping 80+ trailers in one
`recover --ids` run hit YouTube's bot check. Splitting it means only the download step talks to
YouTube in bulk, and it can go slowly: one video at a time, a pause between requests (30 s default),
a stop after 3 bot checks in a row, and resume on the next run. Unavailable videos (removed, private,
region-locked) are written into the candidate list with the reason so later runs skip them without
asking YouTube again (`--recheck` retries them, e.g. from another region). Download walks down each
film's ranked candidates until one works. `apply` checks candidates in rank order (`swap.check`) and
keeps the old trailer if none passes.

**Swap check: span, then share or beats other films** (`swap.check`). A candidate passes if its
matched shots span 20% of the film and either 25% of its shots clear the impostor cutoff or it beats
other films on at least 45% of its non-blank shots (`beats_others`: share of shots that match this
film by more than 0.02 better than any of 5 random other films). The 25% rule alone rejected some
real trailers that match weakly overall (dark, grainy, heavily graded): the current *Bourne Identity*
link has 10% of shots at the impostor cutoff, and 17 of 293 current core trailers are under 25%
(in-session). Such trailers still match their own film far better than other films; another film's
trailer rarely does. On the core, 5% of real trailers score under 0.48 on `beats_others` and 1% of
wrong pairings over 0.58, so 0.45 sits below nearly all real trailers. What we give up: a wrong film
with a similar look (same series, same era) could beat 5 random films; the span rule and the review
page are the backstop. `TODO(why):` the 0.02 margin and 5 other films were not tuned. `apply --batch
--retry` judges candidates rejected in earlier runs again, so films rejected under the old rule get a
second look.

**Batch apply for hundreds of films** (`swap apply --batch`). Plain `apply` runs `trailer_prep` and
`embed trailer` per candidate, which reloads TransNetV2 and CLIP every time; fine for 80 films, slow
for the ~800 new ones. `--batch` works in rounds: every undecided film's next downloaded candidate
goes in place, shot detection and CLIP run once over all of them, then each film is aligned and
decided with the same checks. A rejected film tries its next candidate in the next round. Same
decisions, fewer model loads. A film whose check crashes (e.g. failed shot detection) counts as
rejected.

**Long jobs save per item and resume.** `trailers`, `recover`, `swap prepare` and both `swap apply`
modes write their logs after every film, and reruns skip what is already decided (from the status
and recovery logs, or the candidate list). These jobs can run for hours against YouTube; before
this the logs were only written at the end, so an interrupted run lost every decision it had made.
The backup of a film's original trailer in `cache/replaced/{id}/` is written once: a resumed run that
finds one doesn't overwrite it with a half-tried candidate. It is removed after a restore, so the next swap of that film starts clean.

**New films: metadata first** (`swap prepare --set expand`). For films new to the dataset, MovieNet's
own link stays the first choice when its YouTube metadata looks like a clean trailer (alive, title
passes the filter, 60 to 240 s); only dead, flagged or missing links get a search. Nothing is
downloaded in `prepare`, and `download` fetches one pick per film by default. This keeps requests to
YouTube low (metadata requests count toward the rate limit too, so they are paced) and leaves no pile
of unused candidate videos on disk.

**Swaps are reviewed on a page, one film at a time** (`swap_review.py`, `curate.sh swap-review`).
Rules and alignment checks can miss what a person would catch (a re-cut with a clean title), so every
replaced trailer (`source=search`) also gets a look by eye. The first tool was contact sheets (`swap
sheet`: several films per image, 4 matched shot pairs each), but they were hard to read. The page shows
one film at a time: the old title struck through, the new title, link and channel, the new trailer
playable from the local file, and 8 of its best-matching shots (spread out, no two within 2 shots of
each other) next to the film keyframes they matched. Verdicts are `g`/`b`/`u` per film, kept in the
browser and exported as JSON per reviewer, like the label tool. Images are film frames, so the page
and `labels/swap_img/` are gitignored.

**Low-match trailers were checked by eye** (`data/ids/trailer_review.csv`). The 15 lowest-match
trailers and the 3 weakest search replacements are all the right film. Low match comes from foreign
versions, TV spots, dark films and logos or credits, so they stay in the corpus.

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

**Re-split likely missed cuts** (`resplit.py`). A trailer shot whose 3 keyframes disagree more than
99% of MovieNet film shots do (lowest keyframe cosine below `film_floor_1pct` in
`alignment_diagnostics.json`) likely spans a cut TransNetV2 missed. For those shots only, we take
TransNetV2's per-frame cut probabilities and split the shot once, at its most likely frame, if that
probability is at least 0.1 (TransNetV2's own cutoff is 0.5) and both halves are at least 6 frames.
Only suspect shots are touched, so the lower cutoff can't add cuts elsewhere.
0.1 is well below the default so the soft cuts TransNetV2 hesitates on (dissolves, flashes) still count, which is fine because only shots already flagged by their keyframes are considered; 6 frames (about a quarter second) keeps flash frames from becoming shots of their own. Neither was tuned. What we give up: at most one split per shot, so a shot hiding two
missed cuts stays partly merged. The shot table and keyframes are rewritten and the trailer's CLIP
embedding is removed, so `embed trailer` and everything after see the new shots. New keyframes and the
shot table are built in a side folder and swapped in, so an interrupted run never leaves keyframes
that don't match the shot table. The log records every
trailer checked (`shot_idx` -1 when nothing split) so reruns skip them.

**Second split pass with CLIP, checked by eye before applying** (`resplit_clip.py`). Shots still
below the film floor after `resplit` get every frame embedded with CLIP. The shot is cut where two
neighboring frames are least alike, and each piece is checked and cut again the same way, up to 3
rounds, with pieces at least 6 frames long. This handles shots hiding more than one missed cut, which
`resplit` can't. It is split into propose, review and apply because CLIP both flags the glued shots
and picks the cut, so applying it would lower the below-floor share (3.4%) partly by construction;
that number can't tell us whether the cuts are real. A by-eye check of a sample (spread evenly over
`sim`, the similarity of the two frames at the cut, shown shuffled so the order doesn't hint at the
answer) is the independent measure, and the `sim` cutoff for `apply` is picked from it. Real shots
that change a lot (fast pans, flashes) also get cut, which is why a cutoff is needed at all. What we
give up: cuts above the cutoff stay unapplied, and hand-labeled trailers are skipped as in `resplit`.
The check found no cutoff to pick: of 48 sampled cuts with a verdict, 77% were real, and the share
stays flat around 80% at every `sim` cutoff (83% at 0.5, 79% at 0.7, 80% at 0.9;
`labels/split_review_ced.json`). `sim` doesn't separate real cuts from fake ones, so every proposal is
applied (`--max-sim 1.0`). What we give up: about 1 in 5 applied cuts splits a real shot in two,
and both halves likely match the same film shot.
`TODO(why):` MAX_DEPTH 3 and the 6-frame minimum were not tuned.

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

**Impostor null (no longer sets the threshold).** Score each trailer shot against 3 *other* films;
the 95th percentile of those best scores (0.740) was the threshold before hand labels existed, i.e.
~5% of non-source shots would pass. 1% (0.786 to 0.81) was tried first, but contact sheets showed
matches between 0.72 and 0.79 were almost all correct, so 1% threw away real matches. `align` without
`--threshold` still uses the 5% rule.

**Blank masking.** Black frames and fades match blank frames in *any* film and pushed the 1%
threshold to 0.9975. Shots that look like a solid black/white/grey frame (CLIP similarity > 0.9) are
flagged on both sides: blank trailer shots are never matched and blank film shots are never candidates.

**Top-5 and margin are kept.** The null can't see the main real error, "right scene, wrong shot"
(shot/reverse-shot, coverage of the same moment). Top-5 candidates feed the hand-labeling tool, and
the margin to the best shot outside ±3 neighbors is saved as a confidence signal.

**Hand labels pick the threshold: best F1, now 0.515** (`label_tool.py`, `label_eval.py`). 10 trailers
sampled across decades, each shot labeled against its top-5 candidates (710 usable shots at first, one
labeler so far; 2 trailers are meant to be labeled by both of us for Cohen's kappa). The 0.74 impostor
cutoff had precision 0.90 but recall 0.60: it threw away 40% of real matches, which become false
negatives in the RQ1 labels. The best-F1 cutoff was 0.535 (precision 0.85, recall 0.91). After the
trailer swaps and the re-split, 639 labels are left and the best-F1 cutoff is 0.515 (precision 0.87,
recall 0.92, `alignment_eval.json`). The `align` stage of
`run_all.sh` re-runs `align --threshold` with `best_f1.threshold` from `alignment_eval.json` once labels exist. What we
give up: 83% of impostor scores clear 0.515, so the null no longer limits matches against the wrong
film, and the labels show most "same scene" and many "not in film" shots get matched too (see
`limitations.md`). The null's pass rate at `T` is still saved as `fpr`.

**Sequence smoothing tried and dropped** (`align.smooth`, `align_tune.py`). Idea: shots cut from the
same scene should land on nearby film shots, so match the whole trailer with Viterbi. A first version
penalized jumps between distant film shots and dropped real matches (32 to 23 on *The Shining*),
since trailers cut between scenes all the time. The kept version only rewards consecutive matches
that land within `tau` film shots and never penalizes a jump. Against the hand labels its best F1
(0.846) is still below raw best-match (0.879), so `align` doesn't use it. The function stays for
reference.

**Labels belong to one trailer** (`labels/label_trailers.csv`, `label_eval.load_gold`). Labels are
keyed `imdb_id:shot_idx`, so they only mean something for the trailer that was labeled. After a swap,
the same key points at a different shot of a different video. `label_trailers.csv` records the
`youtube_id` each labeled film had, and `load_gold` drops a film's labels when the status log now
shows another trailer. Miami Vice's trailer was swapped, so its 71 labels drop out (639 labels are
used now, `alignment_eval.json`);
every step that reads labels (`label_eval`, `align_tune`, `align_diagnostics`, `av_sync`, `rerank`) goes
through `load_gold`. What we give up: those labels, unless that film is labeled again. `resplit` skips
hand-labeled trailers for the same reason: splitting a shot renumbers the ones after it.

**Label tool copies its own images** (`label_tool.copy_images`). The page used to load images by
relative path from `data/`, which broke with per-film keyframe zips (`--pack`) and with
`TRAILERCUT_DATA` pointing elsewhere. It now copies the trailer keyframes and re-saves the film
candidates into gitignored `labels/img/`, so the page works from any data layout and the folder can be
handed to a labeler as is.

## Sound alignment (`trailer_asr.py`, `dialogue_align.py`, `av_sync.py`)

**Whisper on the trailer audio, word timestamps.** Trailer dialogue is cut from the film's soundtrack,
so a transcribed line can be looked up in the film's subtitles and placed in the film without looking
at pixels. faster-whisper `small` with voice activity detection, which skips music-only stretches; it
runs locally (~13 s per trailer), so it isn't on the cluster. Transcripts are copyrighted text and stay
in the gitignored cache; the committed `dialogue_alignment/*.csv` hold times and scores only.

**Fuzzy match to subtitle lines and runs of 2 to 3 lines.** Subtitles split sentences, so runs of
lines are candidates too; fuzzy scoring keeps small transcription errors from breaking a match. A
segment counts as found only with at least 3 words (short lines like "Get in here!" occur all over a
film), a score of at least 90, and a margin of at least 10 over the best match more than 30 s away
from it. These were set by reading matches on a few trailers, not tuned: real lines score 90 or more, while a line like "That's right, there's more" scored 84 against an unrelated subtitle; narration scores around 55 to 65.

**Sync rate.** For each found line over a matched picture, is the picture's film shot within 30 s of
the line's film time? Yes means the line plays over its own scene; no means it was laid over other
footage (voice-over, montage). Lines over unmatched pictures (title cards, new footage) aren't
counted. Per trailer: `speech_share`, `words_per_min`, `lines_found`, `sync_rate`. 30 s stands in for "the same scene"; it was not tuned.

**Dialogue is not fused into shot matching.** The idea was to use a line's film time to fix or fill
in picture matches. Against the hand labels, a found line is within 30 s of the labeled source shot
only 59% of the time (93 labeled shots under a found line), far below the picture matcher's precision
(0.85). About half of trailer dialogue plays over a different scene, so the sound says little about
where the picture comes from. Dropped; dialogue stays a separate signal (sync rate, the sound track in
the Trailer View).

**Copy detection (SSCD) as a re-ranker: dropped** (`rerank.py`). CLIP matches what a frame shows, so
a look-alike shot elsewhere in the film can outscore the true source. SSCD is trained to find the same
image after crops, color edits and overlays, so it re-scored CLIP's top 20. On the hand labels it is
worse than CLIP alone (top pick right 71% vs 93%, best F1 0.71 vs 0.88), a CLIP + SSCD blend is too
(0.74), and SSCD as an accept/reject gate gave no F1 gain. With only 3 keyframes per film shot,
stricter visual matching doesn't help.

## Readable features

**Zero-shot concepts instead of raw embedding dimensions** (`concepts.py`). Shot scale uses MovieNet's
5 classes (ECU, CU, MS, FS, LS) as CLIP text prompts, plus a single `scale_width` summary; content uses
20 named concepts (explosion, gun, kiss, crowd, night, text card, ...). SHAP on these reads as
"close-ups of guns", which SHAP on embedding dimension 317 never can.

**Dialogue from film subtitles.** VADER sentiment and word rate of subtitles overlapping each film
shot. For trailer-level dialogue this is a proxy (it reads what the film says under the matched shots,
not what the trailer plays). The RQ2 "speech" group (below) is measured on the trailer's own audio;
the film-side dialogue features stay for RQ1, where the shot is a film shot.

**`visual_change` = 1 - cos(first, last keyframe).** A cheap stand-in for motion without optical flow.

**Act = classic 1/4, 1/2, 1/4 split of story position.** Simple and explainable; synopsis paragraph
position carries the finer story signal.

## RQ1 model (`model.py`)

**Label: `in_trailer` = at least one matched trailer shot points at this film shot.** 4.1% positive
at threshold 0.515 (3.8% at 0.535 before the swaps and re-split, 2.5% at 0.74).

**Logo and credit shots are dropped** (`model.EDGE`). Film shots in the first and last 1% of each film
are left out of training and scoring. The film's first shot counted as "used" in 23.5% of films vs a
2.5% base rate (measured in-session), because studio logos in the film match the same logos in the
trailer. Left in, they teach the model "the start of the film gets picked". The RQ1 page uses the same
filter. We give up any real picks from the first and last 1%. The Trailer View shows no score for
these shots (`viewer_export` joins model scores by shot).

**LightGBM.** Handles mixed-scale features, missing values (no subtitles, no synopsis paragraph) and
464K rows quickly, and TreeSHAP is exact and fast for it.

**Folds grouped by film.** Shots from the same film share look, cast and score; a random split would
let the model memorize films and inflate every metric.

**Metrics.** Average precision (PR-AUC) instead of accuracy or ROC-AUC because of the ~4% positive
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

**Context and relative features exist but aren't in the default model.** `context` = how a shot looks
next to its film (`distinct`: distance from the film's mean embedding; `novelty`: distance from the 2
shots on each side; `recurrence`: best match more than 10 shots away). `relative` = within-film
percentile ranks of the editing, dialogue, scale, context and content features, since a "long" shot
means different things in different films. `model_tune.py` tried them, plus a lambdarank ranker and two
LightGBM settings, on the same folds: all within 0.003 precision at trailer length of the plain full
model, so `model.py` keeps the original groups. The RQ1 page still uses the relative ranks to bin shots.

**Tolerance eval (`model_eval.py`).** The aligner can land one or two shots off the true source
(shot/reverse-shot, continuous coverage), which at shot level is a full miss. `model_eval` rescores the
saved out-of-fold predictions counting a pick as a hit if a used shot is within ±k film shots, to see
whether these near misses hide a better model. Random gets the same tolerance, because a wider window
also makes random hits more likely (at ±3, 21% of shots are near a used shot vs 4.1% exact); lift over
that base rate is the fair comparison. Result: lift drops as k grows, so near misses are not the story.

**EBM for the report's feature curves** (`model_ebm.py`). An explainable boosting machine learns one
curve per feature and adds them up, so the curves are the model, not an approximation of it after the
fact like SHAP on LightGBM. No pairwise interactions, so each effect is one curve. Same readable
features, folds and metrics as LightGBM: lift 2.74 vs 2.82 (`model_ebm.json`), close enough that we
use its curves to explain trailerness. LightGBM stays the main model (it also takes the CLIP block);
`figures/model_shapes.png` shows its binned SHAP for the same features. Curves
are drawn as odds multipliers on one shared log scale, clipped to where 98% of shots fall, since the
outermost bins hold a handful of shots and swing. It takes ~30 min, so it is an opt-in stage.

**Headline metric: per-film precision at trailer length, reported as lift over random.** Pooled AP
mixes films with different base rates (trailer length vs film shot count), so films with many used
shots dominate it. Per-film precision at trailer length asks each film the editor's question, and lift
makes it comparable across tolerances. Pooled AP stays in the tables as a secondary number.

## RQ2 clustering (`cluster.py`)

**Features per trailer in 6 groups:** pacing, audio, story sampling, dialogue, visual, speech (from
`av_sync.csv`; dropped when that file is missing). Clustered on
all features and on each group alone, so we can tell which modality drives any structure.

**k-means with k = 2..8 by silhouette, checked against HDBSCAN.** k-means for interpretable centroids;
HDBSCAN as a check on whether density-based structure exists at all.

**Stability:** refit on 50 bootstrap 80% subsamples and compare to the full-data labels with ARI.

**Genre/era with a baseline:** NMI with primary genre and decade, compared against the 95th percentile
of NMI with shuffled labels. Without the null, any NMI looks like agreement.

**Current read** (threshold 0.535): clusters are stable (ARI 0.79 to 0.94) but weakly separated
(silhouette 0.10 to 0.31). Genre NMI beats the null for all, story, dialogue, visual and (barely)
speech, and stays small (largest 0.103, dialogue); pacing and speech clusters follow decade. The
speech group's k = 8 sits at the edge of the search range with a flat silhouette, so it isn't a real k. Next step: regress out genre and
decade and re-cluster the residuals; if structure stays weak, report continuous axes instead of types.

## Trailer View (`viz/index.html`, `viewer_export.py`)

**One JSON per trailer.** The page loads only the trailer it shows plus a small index for the picker.
Named-feature detail (value + SHAP per feature) is only exported for in-trailer shots and the top 60 by
model score, the shots someone will actually hover; the rest get grouped SHAP only, which keeps files
small.

**Movie barcode from average color.** The film is drawn as a strip of its shots, each filled with the
average color of its middle keyframe and as wide as its share of runtime. It gives the film axis a
recognizable look (night scenes, color grading, act changes) without showing any frames, so it works
in the committed page too. Average color is cheap (JPEG decoded at reduced size, ~1 ms per keyframe)
and cached per film in `cache/shot_colors/`. We give up any notion of shot content; the hover frames
cover that.

**Links plus an order scatter.** Curves from each trailer shot down to its film position show where
footage comes from, but with 70+ shots they cross into a tangle that's hard to read for order. The
scatter (trailer position vs film position, diagonal = film order) answers "does the trailer follow the
film?" at a glance, so both are shown.

**No trailerness chart.** We first drew the model's score as a bar per film shot, then replaced it
with a "Would the model have picked the same shots?" panel (top-k thumbnails, rank of each used shot,
hits vs random). Neither told a clear story (~700 thin bars per film were hard to read), so both were dropped. The model score now shows in the
film-shot tooltip and in the "Why this shot?" SHAP panel, filled by hovering or clicking a film shot on
the barcode. We give up a view of the score profile along the film.

**Hover previews, click locks.** Hover shows a tooltip and highlights the shot and its match, but it
disappears as soon as the mouse moves. Judging a match means looking at both frames for a while, so a
click locks the shot: the highlight stays and the trailer frame and matched film frame show side by
side, large enough to compare. Arrow keys step through matched shots and Esc unlocks, so you can walk a whole trailer from the keyboard.

**Thumbnails are local only.** Frame images load from the viewer's own `data/movienet/frames/` and
`data/cache/trailer_frames/`. Frames are copyrighted, so they are never committed, exported into
`viz/data/` or hosted; on a machine without the data the page still works, just without pictures.
The paths are fixed to `data/` and loose JPEGs, so they ignore `TRAILERCUT_DATA` and `--pack` zips
(unlike the label tool, which copies its images).

**Playable trailer with a custom player.** The trailer plays above the timeline and the current shot
drives the same highlight as hover. No native control bar: a play/pause button, the time, a CC
toggle, click the video to play/pause, and drag the trailer strip or sound track to scrub. Trailers
are AV1, VP9 or H.264 in MP4, so Safari can't play all of them; Chrome and Firefox can.
The native controls look different in every browser and duplicate the timeline; with our own controls the trailer strip itself is the seek bar.

**Captions from the transcript, on by default** (`viz/data/{id}.vtt`). Whisper can merge ~30 s of
speech into one segment, so `viewer_export` splits cues at about 6 s, 12 words or a pause over 1.5 s,
using the word timestamps.

**One sound track.** Loudness as an area, rescaled to its 10th percentile so quiet ends don't flatten
it, tinted violet where a line was found in the film's dialogue. One sound link at a time runs from the
current or hovered line to its film moment; picture links are drawn only in the main chart. A "Sound
vs picture" panel plots each found line's film time against its picture's film time and counts lines
within 30 s. Drawing every sound link next to every picture link was too crowded to read, so sound shows on demand and the sync panel carries the overview.

**Terse labels, no act labels** (also on the RQ1 page): "Keyframe 249", "Shot 7 · 0:39", short
subtitles, no setup/middle/ending marks. The design language is to label things, not explain them; explanations live in these docs.

## RQ1 page (`viz/rq1.html`, `rq1_export.py`)

**One page across all films**, linked from the Trailer View header. The Trailer View answers "what did
this trailer take"; this page answers "what kind of shot gets picked" over the whole corpus.

**Everything is a lift**: pick rate of a kind of shot over the pick rate of the average shot. One
unit for story position, shot scale, within-film shot length, distinctiveness, dialogue and 20
content concepts, so panels compare directly. Shot length and distinctiveness are binned by
within-film percentile (`rel_*`), and a concept counts as present in the film's top 10% for it, since
raw CLIP scores and shot lengths shift from film to film.

**Shared log scale, 0.5x to 2x** (clamped). Log so 0.5x and 2x sit the same distance from 1x; shared
so a long bar in one panel means the same as in another.

**95% intervals from resampling films, not shots.** Shots within a film aren't independent; resampling
films keeps one film from carrying a result.

**Grey when the interval crosses 1x**, blue above, red below. Color then only marks effects that hold up.

**Subtitles are written from the data.** Each panel's line quotes only the strongest effect above and
below 1x whose interval excludes 1x, or says there is no clear difference. Text can't drift from the
numbers after a rerun.

**Model lift bars** reuse `model_eval.json` (exact-shot lift) so the page shows how far the features
get together, next to the single-feature lifts.

**Effect of each feature: EBM curves** (from `model_ebm.json`). The lifts above look at one feature at
a time; the EBM curves show each feature's effect with the others held fixed. The page shows the 6
curves with the highest importance, as odds multipliers on one shared log scale (same reason as the
lift bars), cut to the feature's 1st to 99th percentile since the outer bins hold few shots. The
section is hidden when `model_ebm.json` doesn't exist.

## Pipeline structure

**Staged rebuild vs curation.** `scripts/run_all.sh` only rebuilds outputs from the current inputs, in
named stages, so a change downstream reruns just the stages after it. Steps that change the inputs (dead-link recovery, the trailer audit and swaps,
label pages, variant comparisons against the labels) live in `scripts/curate.sh` and are run on purpose:
they change what every later number is computed from, and the trailer steps are slow under YouTube's
rate limits, so a full rebuild never triggers them.

**`run_all.sh` is a menu, not a run-everything script.** With no arguments it lists the stages and
runs nothing; you name the ones to run. Stages are run on purpose and several take hours (downloads,
shot detection, embeddings, the model), so a bare call shouldn't start all of them. `rq2` (trailer
features, clustering, case studies) and `ebm` are only run when named: RQ2 is a separate part of the
project, and the EBM is slow. Two things moved so the core stages don't need `rq2`: `figures` (data
funnel, alignment and story figures, `summary.json`) is now in `rq1`, and `viewer_export` computes the
Trailer View's loudness and cut curves itself instead of reading `trailer_curves.json` (cached per
trailer in `cache/curves/`, recomputed when the trailer's shot table changes, so a swap or re-split
can't leave stale curves). The script keeps the name `run_all.sh` for now.

## Dataset release (`release.py`)

**Links and annotations only.** `data/release/` holds one table per film (the trailer link in use,
its YouTube title, channel and length, and how well it matches the film: trailer shots, match share,
film span), trailer shot boundaries,
the picture and dialogue alignments, sync features and the hand labels, plus a dataset card. Left
out: videos, frames, audio and transcripts, which are copyrighted (transcripts are the film's
dialogue), and MovieNet's own keyframes, subtitles and annotations, which are research-only under
MovieNet's terms; film shot indices point into MovieNet's shot annotations instead. Labeler names are
replaced by `labeler_1`, `labeler_2`, ... so a public file carries no personal names. The folder is
gitignored and rebuilt from committed files (YouTube titles, channels and lengths come from the
metadata cache and the candidate logs). The annotations are released under CC BY-NC 4.0: non-commercial, to match MovieNet's research-only terms that the alignment depends on.

**The release table has no swap history.** `trailers.csv` lists only the link in use; MovieNet's old
link, whether and why it was replaced, audit flags and review notes stay in `data/ids/` and are not
released. An earlier version had the before and after link per film with reasons; the release only
needs the trailer each film's annotations were made on, and how we got there is our own provenance.

## Dropped from the proposal

- **0.9 alignment accuracy on augmented film shots.** It measured robustness to crops and color
  grading, not correctness on real trailers. Replaced by hand labels.
- **Spoiler stretch goals.** Final-act share is the only spoiler measure and is labeled as a proxy.
