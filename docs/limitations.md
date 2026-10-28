# Known limitations

What the current data and methods can't tell us, so we don't overclaim in the report.

## Data

- **Replaced trailers.** 49 of the 293 MovieNet trailer links were dead and were replaced by YouTube
  search hits that align to the film. 45 of the 49 are from the 1990s on, so newer films lean more on
  replacements. A replacement may be a re-release, anniversary or international cut, not the original
  theatrical trailer (tracked in `source`); the lowest accepted match shares (*Basic* 31%, *Dogma* 33%,
  *Serenity* 34%) haven't been checked by eye.
- **One trailer per film.** Films often have a teaser, several trailers and TV spots. We see one,
  usually whichever MovieNet's authors found, so "what the trailer used" means "what this trailer used".
- **MovieNet keyframes are 3 per shot.** Trailer footage from the middle of a long shot may not
  look like any of the 3 keyframes and goes unmatched.
- **Core set bias.** Films with synopsis alignment are mostly popular English-language films. Results
  don't say much about other cinemas.

## Labels

- **`in_trailer` comes from our aligner, not ground truth.** Missed matches become false negatives
  in the model's training data; wrong matches become false positives. Hand labels measure this on 10
  trailers only.
- **Right scene, wrong shot.** The error we expect most (not measured until the hand labels are in)
  lands on a neighboring film shot. For story position this barely matters; for shot-level features
  (scale, content) it can. The model is also scored at shot level, so the true source shot then counts
  as a negative and the neighbor as a positive; exact-shot metrics understate the model.
- **Shot length.** Longer film shots give a trailer more footage to sample, so they are more likely to
  be labeled in-trailer regardless of content. Controlled with a length-only baseline and
  length-adjusted AP, not removed.

## Features

- **Zero-shot concepts are CLIP's idea of a concept.** "gun" fires on dark metal objects; there is no
  validation set for the 20 concepts. Treat them as soft descriptors.
- **Dialogue is a proxy.** Trailer-level dialogue reads the film's subtitles under matched shots, not
  what the trailer actually plays (trailers recut and remix audio). Whisper on trailer audio fixes this.
- **`visual_change` is not motion.** Cosine between first and last keyframe; a static shot with a
  lighting change scores high.

## Analysis

- **The RQ1 signal is weak.** The full model's AP is 0.050 against 0.024 for random, and at trailer
  length about 9 in 100 top-ranked shots are in the trailer. It ranks better than chance, not well
  enough to say which shots an editor will pick. Readable features carry most of what the model finds
  (AP 0.045 vs 0.050 with CLIP).
- **Clusters are weakly separated** (silhouette 0.11 to 0.31). We report them as reproducible splits,
  not as distinct trailer types.
- **Decade confounds style.** Pacing follows decade, and decade correlates with almost everything.
  Residual clustering is planned.
- **Spoilers.** Final-act share is a proxy for spoilers, not a measure: a final-act shot can be harmless
  and an early shot can reveal a twist.
