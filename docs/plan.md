# Execution plan: audiovisual alignment and the playable Trailer View

Working plan for the current block. Check items off as they land; anything that slips moves to
`TODO.md`. Every step ends with a commit and a docs pass.

## Where we are

- Visual alignment: CLIP, cutoff 0.535 picked from 710 hand-labeled shots (P 0.85, R 0.91).
- Tried and dropped: sequence smoothing (lower F1), SSCD copy-detection re-ranking (top pick right
  71% vs CLIP's 93%) and SSCD as an accept/reject gate (no F1 gain). With only 3 keyframes per film
  shot, stricter visual matching doesn't help; the next signal has to come from audio.
- Audio side started: `trailer_asr` (Whisper, word timestamps) and `dialogue_align` (trailer lines
  to film subtitles) exist; transcription of all trailers is running.

## A. Audio alignment

1. [ ] Finish `trailer_asr` on all trailers; list failures and non-English trailers.
2. [ ] Run `dialogue_align` on all trailers: share of speech segments matched, trailers with no match,
   and why (narrator-only trailers, missing subtitles, other languages; Rear Window matched 0).
3. [ ] Check dialogue anchors against the hand labels: for labeled trailer shots under a matched line,
   how often is the labeled film shot inside the line's film time window?
4. [ ] Sound/picture sync per trailer: share of matched lines whose picture comes from the same film
   moment as the sound vs from somewhere else (voice-over, montage). New trailer features for RQ2,
   one figure.
5. [ ] Fusion: use dialogue time windows to rescue shots the visual matcher missed or got wrong
   (prefer CLIP candidates inside the line's film window). Score against the hand labels; adopt only
   if F1 improves. Mind the label bias: labels were chosen from CLIP's top 5.
6. [ ] Trailer-level dialogue features from the transcript (speech share, words per minute, share of
   lines found in the film vs narration) to replace the subtitle proxy.

## B. Playable Trailer View

1. [ ] Export: trailer speech segments (times, film times, matched) per trailer; transcript text only
   in the local, gitignored viewer data.
2. [ ] Video player above the timeline (trailer video with its audio, local only). Playhead on the
   trailer strip; the current shot drives the same highlight as hover (link, film outline, scatter
   dot); the matched film keyframe shows live next to the video.
3. [ ] Click the trailer strip to seek; space to play/pause; arrows keep stepping through shots.
4. [ ] Audio row: speech segments under the trailer strip with their own links to the film moment
   they come from, in a separate color from picture links; current line shown while playing.
5. [ ] Browser note in the README: trailers are AV1/VP9/H.264 in MP4; Chrome and Firefox play all of
   them. Optional later: transcode to H.264 for Safari.

## C. Vision-language judge (blocked on cluster access)

1. [ ] Slurm job serving an open VLM (e.g. Qwen2.5-VL) with vLLM; prompt: trailer frame + top-5 film
   candidates, "which is the same shot, or none?"
2. [ ] Run only on uncertain shots (score near the cutoff, or picture and dialogue disagree).
3. [ ] Score against the hand labels before adopting. No frames go to external APIs.

## D. Housekeeping

- [ ] Doc-keeper pass after each of A, B, C; small commits; push.
- [ ] Estee labels the first 2 sample trailers so we can report agreement (kappa).
- [ ] Refresh the numbers in the project update draft (293 trailers, label-picked cutoff) before it
  is submitted.
