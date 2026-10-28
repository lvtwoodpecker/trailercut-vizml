# Known limitations

What the current data and methods can't tell us, so we don't overclaim in the report.

## Data

- **Survivor sample.** 244 of 293 trailers are still on YouTube. Dead links skew toward newer films:
  45 of the 49 are from the 1990s on (about 20% of those decades lost), vs 4 of 65 older films. Search replacements may be re-release or anniversary cuts, not the
  original theatrical trailer (tracked in `source`).
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
  lands on a neighboring film shot. For
  story position this barely matters; for shot-level features (scale, content) it can.
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

- **Clusters are weakly separated** (silhouette 0.11 to 0.30). We report them as reproducible splits,
  not as distinct trailer types.
- **Decade confounds style.** Pacing follows decade, and decade correlates with almost everything.
  Residual clustering is planned.
- **Spoilers.** Final-act share is a proxy for spoilers, not a measure: a final-act shot can be harmless
  and an early shot can reveal a twist.
