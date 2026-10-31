# Execution plan: audiovisual alignment and the playable Trailer View

Working plan for the current block. Check items off as they land; anything that slips moves to
`TODO.md`. Every step ends with a commit and a docs pass.

## Where we are

- Visual alignment: CLIP, cutoff 0.515 picked from 639 hand-labeled shots (P 0.87, R 0.92), after the
  trailer swaps and the re-split (0.535 from 710 before).
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

6. [x] Re-split likely missed cuts (keyframe consistency below the film floor; `resplit`; labeled trailers skipped): split each at TransNetV2's most likely cut frame (probability at least 0.1). Core: 1,225 shots split in 242 of 259 checked trailers; shots below the floor 6.7% to 3.4% after the rebuild
7. [x] Full rerun after the replacements and swaps: transcripts, alignment, model, figures, diagnostics, EBM and RQ1 page curves done for the core
8. [x] Second split pass with CLIP for shots still below the floor (`resplit_clip`): proposed on the core (919 cuts) and a sample of 48 checked by eye: 77% real, flat at about 80% for every `sim` cutoff, so no cutoff; applied all (`--max-sim 1.0`) on every trailer with proposals, then rebuilt: glued shots 3.4% to 1.1% (968 trailers)

## F. Representative trailers (original theatrical, original language; rules only)

1. [x] `trailer_audit`: flags from YouTube metadata, transcript and alignment; hard flags (not a trailer, ad, clip, dub, under 60 s) mean swap, soft ones (re-release, low resolution, over 240 s) don't
2. [x] `recover` search: "official trailer" query, rejects dubs/teasers/TV spots/re-releases by title, official titles and 90 to 180 s first
3. [x] Swap the hard-flagged trailers: bulk `recover --ids` was blocked by YouTube; the three-step `swap` (search, slow download, apply): 77 swapped, 6 kept
4. [~] Check every swap by eye: contact sheets (`swap sheet`) were hard to read, replaced by the review page (`curate.sh swap-review`, built: all core swaps plus 30 sampled new films, weakest first); review pending
5. [~] Transcribe the new trailers (done), re-run the audit and check its flags on the audit review page (`curate.sh audit-review`, built): dubs can only be confirmed after Whisper
7. [x] `subtitled` hard flag (title says burned-in subtitles or a foreign-market upload) and a looser swap check (pass weak matches that beat other films); retry on 12 core films with new candidates (`swap apply --batch --retry`). The Wizard of Oz stays: a swap would drop its ~70 hand labels
6. [x] Re-split likely missed cuts (section E6), one full rerun (E7, core) and the EBM rerun

## G. Dataset release and expansion

1. [x] `release` module (local, `data/release/`): one table per film with only the link in use (film, year, link, YouTube title, channel, duration, trailer shots, match share, film span, date checked; which links replaced MovieNet's stays in `data/ids/`) plus our annotations (trailer shot boundaries, trailer-to-film alignment with scores, dialogue alignment times, sync, hand labels); links and annotations only, no videos, frames or transcripts
2. [x] Dataset card (annotations under CC BY-NC 4.0): provenance (MovieNet + YouTube + our pipeline), licensing (MovieNet is research-only; videos stay on YouTube), how each column was made, known limitations
3. [~] Publish to Hugging Face: card numbers filled from the outputs, viewer configs, `check()`, `release --push USER/NAME` (private by default) are in; built locally (1,058 films, ~6 MB); needs a go and a repo name
4. [~] (see section H) Expansion: the ~800 other MovieNet films with keyframes and a trailer link (no synopsis annotation, so no synopsis-position feature); keyframes (~100 GB) on the cluster; trailers through audit + swap

## H. All 1,100 MovieNet films (time position replaces the synopsis requirement)

1. [x] Film list `data/ids/films_1100.csv`: every film with keyframes, with MovieNet's trailer link where it exists (993 do, 107 don't)
2. [x] Keyframes for the ~807 new films (~135 GB, network bound, many hours; resumable)
3. [x] Trailers, metadata first (`swap prepare --set expand`): MovieNet's link stays first choice if its metadata is clean (alive, trailer-like title, 60 to 240 s); dead/flagged/missing links get a search pick (798 films: 349 MovieNet links, 449 search picks)
3b. [x] Download one pick per film (`swap download --set expand`), falling through unavailable ones: all 798 have one (Rio Bravo and The Theory of Everything after a wider re-search, 20 results)
3c. [x] `swap apply --set expand --batch`: 673 of 790 accepted on the first pass (old 25% rule); the 117 rejected re-judged under the beats-other-films rule (56 more), up to 2 more downloads each (34 more), 2 late downloads: 765 of 807 new films have a trailer
3d. [ ] The 42 still without one: 26 ran out of candidates (a wider search found 175 new ones for 21 of them, not downloaded yet), 16 have untried candidates (downloads failed in a DNS blip)
3e. [ ] Post-download audit (Whisper language) and the audit review page
4. [x] Switch the pipeline's film list to all films; new films have no synopsis feature (NaN)
5. [~] Rebuild for the new films: Whisper, `resplit`, CLIP splits, `embed trailer`, align, audio done (968, then 1,058 trailers); rq1, viz and the EBM running
