# Known limitations

What the current data and methods can't tell us, so we don't overclaim in the report.

## Data

- **Replaced trailers.** 49 of the 293 MovieNet trailer links were dead and were replaced by YouTube
  search hits that align to the film. 45 of the 49 are from the 1990s on, so newer films lean more on
  replacements. A replacement may be a re-release, anniversary or international cut, not the original
  theatrical trailer (tracked in `source`). The 3 weakest replacements and the 15 lowest-match trailers
  were checked by eye and are all the right film (`trailer_review.csv`). Six working links were also
  replaced: Scarface's TV spot, four scene clips and Mulholland Drive's German dub. Then 77 trailers
  the audit flagged were swapped for official theatrical ones (6 flagged films kept theirs), and 12
  more in a retry, so 144 of 293 core trailers are search picks now (`trailer_status.csv`). Of the 765
  new films with a trailer, 434 are search picks. Swaps were checked by alignment only; the eye check
  on the review page (`curate.sh swap-review`: all core swaps, 30 sampled new films) is pending, and
  the new films' trailers haven't been through the audit yet.
- **42 of the 1,100 films have no trailer.** All new films (none of the core): 26 ran out of
  candidates and 16 have untried ones. They lean toward films whose trailer is hard to find on
  YouTube, so the 1,058 are not a random sample of MovieNet.
- **YouTube rate limits and region locks.** Bulk downloads trip YouTube's bot check, so swaps run one
  video at a time with pauses and can take several sessions. Some candidates are unavailable in one
  region and fine in another, so which trailer a film ends up with can depend on where the download
  ran (`unavailable` in the candidate lists records the reason).
- **One trailer per film.** Films often have a teaser, several trailers and TV spots. We see one,
  usually whichever MovieNet's authors found, so "what the trailer used" means "what this trailer used".
- **Missed cuts in trailer shots.** TransNetV2 misses some cuts (fast flashes, dissolves). After
  `resplit` and the CLIP pass, 1.1% of 84,288 trailer shots still have keyframes less alike than 99%
  of MovieNet film shots (6.7% on the core before any split, 3.4% after `resplit`; `alignment_diagnostics.json`), so they likely span two shots; such a shot gets one match for two pieces of footage. `resplit`
  splits each such shot at most once, so a shot hiding two missed cuts stays partly merged, and
  hand-labeled trailers are not re-split. The second pass with CLIP (`resplit_clip`) was applied in
  full; the below-floor share drops partly by construction (CLIP both
  flags and cuts). By eye, 77% of 48 sampled cuts are real at every `sim` cutoff, so about 1 in 5
  applied cuts splits a real shot.
- **MovieNet keyframes are 3 per shot.** Trailer footage from the middle of a long shot may not
  look like any of the 3 keyframes and goes unmatched. This also limits strict visual matching: SSCD
  copy detection did worse than CLIP (top pick right 71% vs 93%).
- **Foreign and dubbed trailers.** Some trailers are German, Spanish, French or other versions
  (`trailer_review.csv`). Dubbed audio can't match the film's subtitles, so their `lines_found` and
  `sync_rate` read as zero or missing, not as "no film dialogue".
- **Subtitles are caught by title only.** Burned-in subtitles over English audio pass the `dub` check,
  so the `subtitled` flag reads the YouTube title. An upload with subtitles and a clean English title
  goes unflagged.
- **Dub detection needs the audio.** Titles often don't say a trailer is dubbed, so the audit's `dub`
  flag comes from Whisper's language guess. A new trailer can only be checked for a dub after it is
  downloaded and transcribed, and the guess compares against the language of the film's first
  production country, which is wrong for some co-productions.
- **Core set bias.** Films with synopsis alignment are mostly popular English-language films. Results
  don't say much about other cinemas.

## Labels

- **`in_trailer` comes from our aligner, not ground truth.** Missed matches become false negatives
  in the model's training data; wrong matches become false positives. Hand labels measure this on few
  trailers: 10 were labeled, and since Miami Vice's trailer was swapped its labels no longer count, so
  9 trailers (639 labels) are left; precision 0.87, recall 0.92 at the 0.515 cutoff.
- **Hand labels: one labeler, possible anchoring.** All labels are from one person, so there is
  no kappa yet. The tool shows candidates sorted by score, and 94% of picks were candidate #1, so
  labels may lean toward the aligner's own choice and overstate its accuracy.
- **The 0.515 cutoff lets through shots that aren't in the film.** 83% of impostor scores (best match
  in a *different* film) clear it, so the null no longer bounds false matches. In the labels, 96% of 27
  "same scene" and 43% of 54 "not in film" trailer shots score above it (`alignment_diagnostics.json`). Trailer
  stats that count matches (match share, story sampling) include some of these.
- **Look-alike shots, not neighbors.** Of the 36 labeled shots matched to the wrong film shot at 0.535,
  only 5 land within ±3 shots of the true one (in-session count); the true shot is usually rank 2
  somewhere else in the film. Unlike a neighbor, that can put the shot at the wrong story position,
  and the ±k tolerance eval can't catch it. Lift over random still falls as the tolerance widens (full
  3.15x exact, 1.70x at ±3), so model misses are different moments too.
- **Shot length.** Longer film shots give a trailer more footage to sample, so they are more likely to
  be labeled in-trailer regardless of content. Controlled with a length-only baseline and
  length-adjusted AP, not removed.
- **Studio logos and credits.** Logo and credit shots match the trailer's own logos and cards. The
  RQ1 model and page drop the first and last 1% of each film for this; logos or cards elsewhere in a
  film (and any real picks inside that 1%) are not handled. Trailer title cards and logos that match
  weakly (0.515 to 0.744) are wrong about 1 time in 3 in the hand labels (9 of 28 vs 37 of 200 for other
  shots in that range), mostly landing on the film's credits or signs, so the "text card" and "logo"
  concept lifts (1.42x, 1.33x) run a bit high. Dropping those matches threw out more right ones than
  wrong ones (results.md 2026-11-11).

## Features

- **Zero-shot concepts are CLIP's idea of a concept.** "gun" fires on dark metal objects; there is no
  validation set for the 20 concepts. Treat them as soft descriptors.
- **Subtitle dialogue is a proxy.** The `dialogue` group reads the film's subtitles under matched
  shots, not what the trailer plays. The `speech` group uses the trailer's own audio instead.
- **285 films have no MovieNet subtitles.** Their `dialogue` features and sync rates are missing, not
  zero; the models treat them as missing, and the RQ1 dialogue row uses only the 797 films with subtitles.
- **Whisper errors.** It can merge up to ~30 s of speech into one segment and adds stray words. A
  merged segment is matched as one line; only 29% of speech segments are found in the subtitles.
- **Sync uses a 30 s window.** A line counts as playing over its own scene if its film time is within
  30 s of the picture's film shot, so nearby scenes count as in sync and long scenes can count as out
  of sync. The dialogue anchor is within 30 s of the hand-labeled shot only 59% of the time, so
  per-trailer sync rates are noisy.
- **`visual_change` is not motion.** Cosine between first and last keyframe; a static shot with a
  lighting change scores high.

## Analysis

- **The RQ1 signal is weak.** The full model's AP is 0.099 against 0.046 for random, and at trailer
  length about 15 in 100 top-ranked shots are in the trailer (lift 2.75x over random). It ranks better
  than chance, not well enough to say which shots an editor will pick. Readable features carry most of
  what the model finds (lift 2.59x). Allowing ±k shots of slack doesn't rescue it (lift 1.56x at ±3), and extra features or a ranker don't move it (`model_tune.json`).
- **The third reason is shaky.** Retrained on different films, the model keeps a shot's top reason
  76% of the time but its top-3 set only 55% (`shap_stability.json`); read the first reason, not the order of the rest.
- **Uneven across films.** Lift is 2.2x on the quarter of films with the fewest shots and 3.7x on the
  most, but about half of that is arithmetic (lower positive rates in long films); on per-film AUC
  the gap is 0.68 vs 0.64. Comedy stays below at equal film length (AUC 0.014 lower), and so do films
  from before 1990 (0.035 lower); romance and action don't differ clearly once length is controlled (`error_slices.json`, results.md 2026-11-06).
- **Slice tests mix length and quality.** `error_slices` lift is a ratio of means, and its permutation
  test rejects when a slice's films differ in length; read it with the length-adjusted numbers.
- **Clusters are weakly separated** (silhouette 0.10 to 0.31). We report them as reproducible splits,
  not as distinct trailer types.
- **Correlated features share credit.** CLIP and the content concepts (both from the same embedding), and story position and act, overlap, so SHAP and EBM split their effect between them; read a group's attribution together with its partner, and as association, not cause.
- **Decade confounds style.** Pacing follows decade, and decade correlates with almost everything.
  Residual clustering is planned.
- **Spoilers.** Final-act share is a proxy for spoilers, not a measure: a final-act shot can be harmless
  and an early shot can reveal a twist.
