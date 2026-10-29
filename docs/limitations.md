# Known limitations

What the current data and methods can't tell us, so we don't overclaim in the report.

## Data

- **Replaced trailers.** 49 of the 293 MovieNet trailer links were dead and were replaced by YouTube
  search hits that align to the film. 45 of the 49 are from the 1990s on, so newer films lean more on
  replacements. A replacement may be a re-release, anniversary or international cut, not the original
  theatrical trailer (tracked in `source`). The 3 weakest replacements and the 15 lowest-match trailers
  were checked by eye and are all the right film (`trailer_review.csv`). Scarface's working TV spot was
  also replaced, by an HD trailer.
- **One trailer per film.** Films often have a teaser, several trailers and TV spots. We see one,
  usually whichever MovieNet's authors found, so "what the trailer used" means "what this trailer used".
- **MovieNet keyframes are 3 per shot.** Trailer footage from the middle of a long shot may not
  look like any of the 3 keyframes and goes unmatched. This also limits strict visual matching: SSCD
  copy detection did worse than CLIP (top pick right 71% vs 93%).
- **Foreign and dubbed trailers.** Some trailers are German, Spanish, French or other versions
  (`trailer_review.csv`). Dubbed audio can't match the film's subtitles, so their `lines_found` and
  `sync_rate` read as zero or missing, not as "no film dialogue".
- **Core set bias.** Films with synopsis alignment are mostly popular English-language films. Results
  don't say much about other cinemas.

## Labels

- **`in_trailer` comes from our aligner, not ground truth.** Missed matches become false negatives
  in the model's training data; wrong matches become false positives. Hand labels measure this on 10
  trailers only (precision 0.85, recall 0.91 at the 0.535 cutoff).
- **Hand labels: one labeler, possible anchoring.** All 710 labels are from one person, so there is
  no kappa yet. The tool shows candidates sorted by score, and 94% of picks were candidate #1, so
  labels may lean toward the aligner's own choice and overstate its accuracy.
- **The 0.535 cutoff lets through shots that aren't in the film.** 77% of impostor scores (best match
  in a *different* film) clear it, so the null no longer bounds false matches. In the labels, 37 of 38
  "same scene" and 27 of 71 "not in film" trailer shots still get matched (in-session count). Trailer
  stats that count matches (match share, story sampling) include some of these.
- **Look-alike shots, not neighbors.** Of the 36 labeled shots matched to the wrong film shot at 0.535,
  only 5 land within ±3 shots of the true one (in-session count); the true shot is usually rank 2
  somewhere else in the film. Unlike a neighbor, that can put the shot at the wrong story position,
  and the ±k tolerance eval can't catch it. Lift over random still falls as the tolerance widens (full
  3.22x exact, 1.71x at ±3), so model misses are different moments too.
- **Shot length.** Longer film shots give a trailer more footage to sample, so they are more likely to
  be labeled in-trailer regardless of content. Controlled with a length-only baseline and
  length-adjusted AP, not removed.
- **Studio logos and credits.** Logo and credit shots match the trailer's own logos and cards. The
  RQ1 model and page drop the first and last 1% of each film for this; logos or cards elsewhere in a
  film (and any real picks inside that 1%) are not handled.

## Features

- **Zero-shot concepts are CLIP's idea of a concept.** "gun" fires on dark metal objects; there is no
  validation set for the 20 concepts. Treat them as soft descriptors.
- **Subtitle dialogue is a proxy.** The `dialogue` group reads the film's subtitles under matched
  shots, not what the trailer plays. The `speech` group uses the trailer's own audio instead.
- **Whisper errors.** It can merge up to ~30 s of speech into one segment and adds stray words. A
  merged segment is matched as one line; only 29% of speech segments are found in the subtitles.
- **Sync uses a 30 s window.** A line counts as playing over its own scene if its film time is within
  30 s of the picture's film shot, so nearby scenes count as in sync and long scenes can count as out
  of sync. The dialogue anchor is within 30 s of the hand-labeled shot only 59% of the time, so
  per-trailer sync rates are noisy.
- **`visual_change` is not motion.** Cosine between first and last keyframe; a static shot with a
  lighting change scores high.

## Analysis

- **The RQ1 signal is weak.** The full model's AP is 0.076 against 0.038 for random, and at trailer
  length about 12 in 100 top-ranked shots are in the trailer (lift 3.22x). It ranks better than chance,
  not well enough to say which shots an editor will pick. Readable features carry most of what the
  model finds (AP 0.069 vs 0.076 with CLIP). Allowing ±k shots of slack doesn't rescue it (lift 1.71x
  at ±3), and extra features or a ranker don't move it (`model_tune.json`).
- **Clusters are weakly separated** (silhouette 0.10 to 0.31). We report them as reproducible splits,
  not as distinct trailer types.
- **Decade confounds style.** Pacing follows decade, and decade correlates with almost everything.
  Residual clustering is planned.
- **Spoilers.** Final-act share is a proxy for spoilers, not a measure: a final-act shot can be harmless
  and an early shot can reveal a twist.
