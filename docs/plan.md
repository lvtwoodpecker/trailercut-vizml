# Execution plan: audiovisual alignment and the playable Trailer View

Working plan for the current block. Check items off as they land; anything that slips moves to
`TODO.md`. Every step ends with a commit and a docs pass.

## Where we are

- Visual alignment: CLIP, cutoff 0.535 picked from 710 hand-labeled shots (P 0.85, R 0.91).
- Tried and dropped: sequence smoothing (lower F1), SSCD copy-detection re-ranking (top pick right
  71% vs CLIP's 93%) and SSCD as an accept/reject gate (no F1 gain). With only 3 keyframes per film
  shot, stricter visual matching doesn't help; the next signal has to come from audio.
- Audio side started: `trailer_asr` (Whisper, word timestamps) and `dialogue_align` (trailer lines
  to film subtitles) ran on all 293 trailers; 29% of speech segments found in the film.

## A. Audio alignment

1. [x] Finish `trailer_asr` on all trailers; list failures and non-English trailers.
2. [x] Run `dialogue_align` on all trailers: share of speech segments matched, trailers with no match,
   and why (narrator-only trailers, missing subtitles, other languages; Rear Window matched 0).
3. [x] Check dialogue anchors against the hand labels: for labeled trailer shots under a matched line,
   how often is the labeled film shot inside the line's film time window?
4. [x] Sound/picture sync per trailer: share of matched lines whose picture comes from the same film
   moment as the sound vs from somewhere else (voice-over, montage). New trailer features for RQ2,
   one figure.
5. [x] Fusion: use dialogue time windows to rescue shots the visual matcher missed or got wrong
   (prefer CLIP candidates inside the line's film window). Score against the hand labels; adopt only
   if F1 improves. Mind the label bias: labels were chosen from CLIP's top 5.
   Dropped: a found line is within 30 s of the labeled shot only 59% of the time.
6. [x] Trailer-level dialogue features from the transcript (speech share, words per minute, share of
   lines found in the film vs narration) to replace the subtitle proxy.

## B. Playable Trailer View

1. [x] Export: trailer speech segments (times, film times, matched) per trailer; transcript text only
   in the local, gitignored viewer data.
2. [x] Video player above the timeline (trailer video with its audio, local only). Playhead on the
   trailer strip; the current shot drives the same highlight as hover (link, film outline, scatter
   dot); the matched film keyframe shows live next to the video.
3. [x] Click the trailer strip to seek; space to play/pause; arrows keep stepping through shots.
4. [x] Audio row: speech segments under the trailer strip with their own links to the film moment
   they come from, in a separate color from picture links; current line shown while playing.
5. [x] Browser note in the README: trailers are AV1/VP9/H.264 in MP4; Chrome and Firefox play all of
   them. Optional later: transcode to H.264 for Safari.

## C. Vision-language judge (blocked on cluster access)

1. [ ] Slurm job serving an open VLM (e.g. Qwen2.5-VL) with vLLM; prompt: trailer frame + top-5 film
   candidates, "which is the same shot, or none?"
2. [ ] Run only on uncertain shots (score near the cutoff, or picture and dialogue disagree).
3. [ ] Score against the hand labels before adopting. No frames go to external APIs.

## D. Housekeeping

- [ ] Doc-keeper pass after each of A, B, C (A and B done); small commits; push.
- [ ] A second labeler labels the first 2 sample trailers so we can report agreement (kappa).
- [ ] Refresh the numbers in the project update draft (293 trailers, label-picked cutoff) before it
  is submitted.

## E. Better alignment and trailerness: diagnostics and explanations

1. [x] Shape plots: how each readable feature moves the pick chance (binned SHAP, as an odds multiplier), small multiples; `figures/model_shapes.png`
2. [x] EBM (glass-box model, readable features) with the same folds and metrics; adopt its curves for the report if lift is close to LightGBM's
3. [x] Case finder: trailers ranked by order (chronological vs shuffled), sound/picture sync, final-act share, and per-film model hits; `data/features/case_studies.json`
4. [x] Alignment diagnostics from the labels: score distributions by label type; match rate by trailer language (Whisper); `figures/alignment_diagnostics.png`
5. [x] Shot detection check: trailer shots whose 3 keyframes disagree (likely missed cuts); count and examples

Findings so far: wrong (look-alike) and same-scene matches score as high as right ones, so the
cutoff can't remove them; trailer language makes no difference at 0.535; 6.7% of trailer shots have
keyframes less consistent than 99% of MovieNet shots (likely missed cuts). Four MovieNet "trailers"
were scene clips (Moonstruck, Commando, Ferris Bueller, Phone Booth) and were replaced.

6. [~] Re-split likely missed cuts (keyframe consistency below the film floor; `resplit`, running; labeled trailers skipped): split each at TransNetV2's most likely cut frame (probability at least 0.1), check the effect on matching
7. [ ] Full rerun after the replacements and swaps (transcripts, alignment, features, model, EBM, exports)

## F. Representative trailers (original theatrical, original language; rules only)

1. [x] `trailer_audit`: flags from YouTube metadata, transcript and alignment; hard flags (not a trailer, ad, clip, dub, under 60 s) mean swap, soft ones (re-release, low resolution, over 240 s) don't
2. [x] `recover` search: "official trailer" query, rejects dubs/teasers/TV spots/re-releases by title, official titles and 90 to 180 s first
3. [x] Swap the hard-flagged trailers: bulk `recover --ids` was blocked by YouTube; the three-step `swap` (search, slow download, apply): 77 swapped, 6 kept
4. [~] Contact sheet of every swap to skim (`swap sheet`, built; run after the rebuild)
5. [ ] Transcribe the new trailers, re-run the audit: dubs can only be confirmed after Whisper
6. [ ] Re-split likely missed cuts (section E6), then one full rerun (E7), then rerun the EBM

## G. Dataset release and expansion

1. [x] `release` module (local, `data/release/`): one consolidated table per film (MovieNet link vs link in use, source, why it changed, channel, title, duration, audit flags, match share/span, date checked) plus our annotations (trailer shot boundaries, trailer-to-film alignment with scores, dialogue alignment times, sync, hand labels); links and annotations only, no videos, frames or transcripts
2. [x] Dataset card (annotations under CC BY-NC 4.0): provenance (MovieNet + YouTube + our pipeline), licensing (MovieNet is research-only; videos stay on YouTube), how each column was made, known limitations
3. [ ] Publish to Hugging Face (needs an HF account/token)
4. [ ] Expansion: the ~800 other MovieNet films with keyframes and a trailer link (no synopsis annotation, so no synopsis-position feature); keyframes (~100 GB) on the cluster; trailers through audit + swap

## H. All 1,100 MovieNet films (time position replaces the synopsis requirement)

1. [x] Film list `data/ids/films_1100.csv`: every film with keyframes, with MovieNet's trailer link where it exists (993 do, 107 don't)
2. [ ] Keyframes for the ~807 new films (~135 GB, network bound, many hours; resumable)
3. [x] Trailers, metadata first (`swap prepare --set expand`): MovieNet's link stays first choice if its metadata is clean (alive, trailer-like title, 60 to 240 s); dead/flagged/missing links get a search pick (798 films: 349 MovieNet links, 449 search picks)
3b. [~] Download one pick per film (`swap download --set expand`, running), falling through unavailable ones; then `swap apply --set expand --batch` (alignment, coverage) and the post-download audit (Whisper language)
4. [x] Switch the pipeline's film list to all films; new films have no synopsis feature (NaN)
5. [ ] Rebuild: prep, align, audio, rq1, rq2, viz (and the EBM)
