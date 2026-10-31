# Data dictionary

Every file the pipeline writes that someone downstream reads, with its columns. "C" = committed
(not gitignored), "G" = gitignored (rebuilt by the pipeline). Files marked "not written yet" come from
steps that haven't produced a committed output so far.

## IDs and logs (`data/ids/`)

**`core_films_293.csv`** (C): the 293 core films (the pipeline's film list before `films_1100.csv`; still used for keyframes in `run_all.sh data`).
`imdb_id`, `youtube_id` (MovieNet's original trailer link).

**`films_1100.csv`** (C): all 1,100 MovieNet films with keyframes. `imdb_id`, `youtube_id` (MovieNet's
trailer link; empty for the 107 films without one), `core` (1 for the 293 core films, else 0). The
pipeline's film list (`paths.IDS`; `TRAILERCUT_FILMS` overrides).

**`trailer_status.csv`** (C): one row per film with a trailer in use (1,058: the 293 core films and
765 new ones, added by `swap apply --set expand`).
`imdb_id`, `youtube_id` (link actually used), `status` (`ok` / `unavailable` / `error`), `duration` (s),
`title` (YouTube title), `error` (first line of the yt-dlp error, up to 200 chars), `source` (`movienet`
or `search`). `duration` and `title` are empty when the video was already on disk. All rows are `ok`.
Core: 149 `movienet`, 144 `search` (49 dead links; Scarface, four scene clips and Mulholland Drive's dub
replaced with `recover --ids`; 77 flagged trailers swapped with `swap apply`, then 12 more in the
retry). New films: 331 `movienet`, 434 `search`. Saves re-read the file and replace only the films
they decided, so two jobs can run at once.

**`trailer_recovery.csv`** (C): every candidate tried by `recover` (dead links, `--ids`) or `swap apply`, appended across runs and after every film (1,411 rows for 940 films). `swap apply` reads it to skip films already decided.
`imdb_id`, `rank` (search order among filtered results, 0-based), `youtube_id`, `title`, `channel`,
`duration` (s, from search results), `match_share` (fraction of non-blank shots matched to the film,
empty if download or alignment failed; `recover` sets it to 0 when the matched shots span under
`--min-span`, default 0.2, of the film), `accepted` (`match_share` at least `--min-match`, default
0.25, and for `swap apply` a span of at least 0.2).

**`trailer_review.csv`** (C): eye check of low-match and recovered trailers. `imdb_id`, `verdict`
(`right film` or `replaced`), `reason` (free text: foreign version, TV spot, dark film, ...; may be empty).

**`trailer_audit.csv`** (C, `trailer_audit`): one row per trailer in use, most flags first.
`imdb_id`, `film` (MovieNet title), `youtube_id`, `source`, `yt_title` (YouTube title, or the title
logged at download if YouTube refused the metadata request), `channel`, `duration` (s), `height`
(best available picture height, px), `language` (Whisper's guess), `expected_language` (from the
film's first production country; empty if not in the lookup), `span` (10th to 90th percentile of
matched shots' `film_story_pos`), `flags` (space-separated, below), `n_flags`, `swap` (any hard flag).
Hard flags: `not_trailer` (title says TV spot, teaser, clip, scene, featurette, Blu-ray, review, ...),
`ad` (transcript sells a home-video edition), `clip` (span under 0.2), `dub` (confident spoken language,
p at least 0.8 with over 15 s of speech, differs from the expected one), `subtitled` (title says
burned-in subtitles or a foreign-market upload: subtitulado, legendado, VOSTFR, Sub.KOR, español, ...),
`short` (under 60 s). Soft
flags: `rerelease` (anniversary, restoration, 4K, ... or a year in the title 3+ years from the film's),
`low_res` (under 360p), `long` (over 240 s). The committed file was written before the alignment was
rerun for the four replaced scene clips, so their rows still show the old `span` and `clip` flag.

**`swap_candidates.csv`** (C, `swap search`): ranked replacement candidates for the audit's `swap` films
whose link is still MovieNet's (overridden by `labels/audit_review_*.json`). A new search appends after a
film's existing candidates, ranks continuing, and skips ones already listed. `imdb_id`, `rank` (0 = first choice: official/theatrical titles,
then 90 to 180 s, then search order), `youtube_id`, `title`, `channel`, `duration` (s, from search),
`unavailable` (YouTube's reason when a download showed the video is gone, private or region-locked;
empty otherwise; `download --recheck` retries these), `source` (`search`; only on rows added since search
appends). The committed list has 122 candidates for 12 films: the core retry search, run when a search
still replaced the list (the first 551 candidates for 83 films were overwritten).

**`swap_downloads.csv`** (C, `swap download`): one row per video attempted. `imdb_id`, `youtube_id`,
`status` (`ok` / `error`), `error` (first line of the yt-dlp error). Bot-check errors are retried on
the next run. The committed log has 83 films downloaded.

**`expand_candidates.csv`** (C, `swap prepare --set expand`): first-choice trailers for films new to the
dataset, same columns as `swap_candidates.csv` plus `source` (`movienet` for MovieNet's own link kept
as rank 0, `search` for search picks ranked from 1). Appended per film. The committed file has 798
films: 349 keep MovieNet's link, 449 have a search pick. Rio Bravo and The Theory of Everything, whose picks
all failed to download, got a wider re-search (20 results) appended.
`unavailable` is filled once downloads have run.

**`expand_downloads.csv`** (C, `swap download --set expand`): same columns as `swap_downloads.csv`.
The committed log has a downloaded pick for all 798 new films.

## Trailer shots (`data/features/trailer_shots/{imdb_id}.csv`, C)

One row per TransNetV2 shot, renumbered after `resplit` splits a shot.
`shot_idx`, `frame_start`, `frame_end` (inclusive), `t_start`, `t_end` (s), `shot_len` (s),
`trailer_pos` (shot midpoint as a fraction of video length, 0 to 1).

## Alignment (`data/features/alignment/{imdb_id}.csv`, C)

One row per trailer shot.

| column | meaning |
|---|---|
| `shot_idx`, `t_start`, `t_end`, `shot_len`, `trailer_pos` | copied from the trailer shot table |
| `film_shot` | best-matching MovieNet film shot index |
| `score` | similarity to that shot (mean over trailer keyframes of max cosine over film keyframes) |
| `margin` | `score` minus the best score outside ±3 film shots of the best |
| `blank` | mean over keyframes of the best CLIP similarity to a solid black/white/grey frame is > 0.9; never matched |
| `matched` | `score >= threshold` and not blank |
| `film_story_pos` | story position (0 to 1, film shot midpoint over film duration) of the matched film shot; empty if not matched |
| `top5`, `top5_scores` | space-separated top-5 film shot indices and scores |

**`alignment_threshold.json`** (C): `fpr`, `threshold`, `source`, `impostor_threshold` (5% impostor cutoff, always saved; used by `recover`), `n_null`, `null_quantiles` (impostor null at
0.5, 0.9, 0.99, 0.999). `source` is `impostor_null` (threshold = null quantile at `fpr`) or `manual`
(`align --threshold`; `fpr` is then the share of impostor scores at or above that threshold). A file
without `source` means `impostor_null`. The committed file is `manual`, threshold 0.535, `fpr` 0.77.

**`alignment_eval.json`** (C): hand-label scores from `label_eval`; keys listed in `labeling_guide.md`.

**`align_tune.json`** (C): raw vs smoothed matching scored against the hand labels (`align_tune`).
`n` (labeled shots, `u` dropped), `trailers`, `current` (raw at the threshold in
`alignment_threshold.json` when it ran: the label-picked cutoff if `align --threshold` ran last), `best`
(`raw`, `smooth`: the best-F1 row of each), `grid` (every row). Each row: `method` (`raw` / `smooth`),
`threshold`, `lam`, `tau` (smooth only), `precision`, `recall`, `f1`, `near_precision` (a match within
±3 film shots of the labeled source counts), `n_pred` (shots predicted matched).

## Sound (`data/features/`)

**`dialogue_alignment/{imdb_id}.csv`** (C): one row per Whisper speech segment of the trailer, no text.
`seg_idx` (segment index in the transcript), `t_start`, `t_end` (trailer s), `n_words`, `film_t_start`,
`film_t_end` (s, the best-matching subtitle line or run of 2 to 3 lines), `score` (fuzzy match, 0 to
100), `margin` (score minus the best score more than 30 s away), `matched` (at least 3 words, score
at least 90, margin at least 10). Film times and scores are empty for segments under 2 words or films
without subtitles.

**`av_sync.csv`** (C): one row per trailer. `imdb_id`, `speech_share` (transcribed speech time over
trailer length), `words_per_min`, `lines_found` (share of segments `matched`), `sync_rate` (of found
lines over a matched picture, share whose film time is within 30 s of the picture's film shot; empty
when there are none), `n_lines_paired` (those lines; `sync_rate`'s denominator).

**`av_sync.json`** (C): `trailers`, `trailers_with_found_lines`, `lines_paired`, `pooled_sync_rate`
(weighted by `n_lines_paired`), `median_sync_rate_3plus` and `trailers_3plus` (trailers with 3+ paired
lines), `median_speech_share`, `median_words_per_min`, `sync_seconds` (30), `labels`
(`labeled_shots_under_found_line`, `line_within_sync_of_labeled_shot`: share of those where the line's
film time is within 30 s of the labeled source shot).

**`alignment_diagnostics.json`** (C, `align_diagnostics`): `threshold` (cutoff in use), `labels` (per
hand-label kind `right shot`, `wrong shot`, `same scene, other shot`, `not in film`: `n`,
`median_score`, `above_cutoff` share), `language` (per trailer group `English`, `other language`,
`no speech` from Whisper: `trailers`, `median_match` rate), `cuts` (`film_floor_1pct`: keyframe
consistency, the lowest cosine between a shot's 3 keyframes, that only 1% of MovieNet film shots fall
below; `trailer_shots`; `trailer_below_floor`: share of trailer shots below it, likely missed cuts).

**`resplit_log.csv`** (C, `resplit`): appended per trailer. `imdb_id`, `shot_idx`
(the shot that was split, numbered as before the split; -1 for a trailer checked with nothing to
split), `split_frame` (first frame of the second half), `prob` (TransNetV2's cut probability there;
empty for -1 rows).

**`split_proposals.csv`** (C, `resplit_clip propose`): appended per trailer, one row per proposed cut
inside a shot still below the film floor. `imdb_id`, `youtube_id` (trailer the proposal was made on;
rows for an older trailer are ignored), `shot_idx`, `frame_start`, `frame_end` (the shot as it was),
`cut_frame` (first frame of the new piece), `sim` (cosine between the CLIP embeddings of the frames
just before and at the cut; lower = more likely a real cut), `depth` (0 for the first cut in a shot,
up to 2 for cuts inside its pieces). A `shot_idx` -1 row marks a trailer checked with nothing to cut.
**`split_applied.csv`** (C, `resplit_clip apply`): one row per trailer cut: `imdb_id`, `youtube_id`,
`cuts` (number applied), `max_sim` (cutoff used).

**`rerank_eval.json`** (C): `clip`, `sscd`, `blend` (CLIP + SSCD), each with `best_f1` (`threshold`,
`precision`, `recall`, `f1`, `near_precision`, `n_pred`, as in `align_tune.json`) and
`recall_no_threshold` (share of labeled sources that are the top pick).

## RQ1: film shots (`data/features/film_shots.parquet`, G, ~65 MB)

One row per film shot of every film with an aligned trailer.

| group | columns |
|---|---|
| ids | `imdb_id`, `shot_idx`, `t_start`, `t_end` |
| position | `story_pos` (shot midpoint over film duration, 0 to 1), `act` (0/1/2 from a 1/4, 1/2, 1/4 split of `story_pos`), `synopsis_pos` (synopsis paragraph index over paragraphs minus 1, 0 to 1, empty for shots no paragraph covers) |
| dialogue | `dlg_words` (words in subtitles overlapping the shot), `dlg_rate` (words / max(shot_len, 0.5 s)), `dlg_has` (0/1), `dlg_sentiment` (VADER compound, empty when no subtitle overlaps) |
| scale | `scale_ecu`, `scale_cu`, `scale_ms`, `scale_fs`, `scale_ls` (softmax over the 5 scale prompts, CLIP logits ×100; sums to 1), `scale_width` (expected scale index, 0 = ECU to 4 = LS) |
| content | raw CLIP cosine to each prompt, mean over keyframes: `c_explosion`, `c_gun`, `c_fight`, `c_chase`, `c_running`, `c_kiss`, `c_crying`, `c_scream`, `c_laugh`, `c_crowd`, `c_night`, `c_day_outdoor`, `c_creature`, `c_space`, `c_water`, `c_city`, `c_interior`, `c_text_card`, `c_logo`, `c_blood` |
| editing | `shot_len` (s), `visual_change` (1 minus cosine of first and last keyframe) |
| label | `trailer_hits` (matched trailer shots pointing here), `in_trailer` (hits > 0) |

## RQ1 model outputs

`model`, `model_tune` and `rq1_export` leave out film shots with `story_pos` below 0.01 or above 0.99
(studio logos and credits, `model.EDGE`), so their row counts are lower than `film_shots.parquet`.
`model.load` also adds, in memory only, `distinct`, `novelty`, `recurrence` (group `context`) and
`rel_<feature>` within-film percentile ranks (group `relative`) for `shot_len`, `visual_change`,
`dlg_words`, `scale_width`, the 3 context features and the 20 concepts.

**`model_report.json`** (C): `n_shots`, `n_films`, `positive_rate`, `folds`, `clip_pcs`, `groups`,
`models` (per model: `ap`, `ap_film`, `r_prec`, `ap_lenadj`), `ablation_drop` (full minus each
ablation, per metric), `shap_mean_abs_by_group` (mean |summed SHAP| per group), `shap_mean_abs_top`
(15 largest mean |SHAP| columns). `groups` lists each group's columns except `clip`.

**`model_preds.parquet`** (G): `imdb_id`, `shot_idx`, `shot_len`, `in_trailer`, then one out-of-fold
score column per model (`random`, `position_only`, `length_only`, `clip_only`, `readable`, `full`,
`full_minus_<group>` for the 5 readable groups and `clip`).

**`model_eval.json`** (C): keys `k0`, `k1`, `k3`, `k5` (tolerance in film shots), then one entry per
model (`random`, `position_only`, `length_only`, `clip_only`, `readable`, `full`) with `base_rate`
(share of shots within ±k of a used shot; what random scores as precision), `ap` (pooled AP against
the ±k label), `prec_at_k` (per film, share of the top-m shots within ±k of a used shot, m = shots the
trailer used, averaged over films), `recall_at_k` (per film, share of used shots with a top-m shot
within ±k, averaged) and `lift` (`prec_at_k / base_rate`). `k0` is the exact-shot score.

**`model_tune.json`** (C): `n_shots`, `n_films`, `positive_rate`, `variants` (per variant: `ap`,
`ap_film`, `r_prec`, `ap_lenadj`, and `lift` = `r_prec / positive_rate`). Variants: `random`, `full`,
`full+context`, `full+relative`, `full+context+relative`, `rank: full+context+relative`, `rank, slower
learning`, `rank, smaller trees`. The committed file was computed at threshold 0.74, before the
re-alignment at 0.535.

**`model_ebm.json`** (C, `model_ebm`): `metrics` (out of fold, same folds as `model.py`: `ap`, `ap_film`,
`r_prec`, `ap_lenadj`, `lift` = `r_prec` over the positive rate), `lightgbm_readable_lift` (the
`readable` lift at k0 from `model_eval.json`), `curves` (per readable feature, from a fit on all films:
`x` bin edges, `score` per bin in log-odds, `importance` = mean absolute score).

**`model_shap.parquet`** (G): `imdb_id`, `shot_idx`, one out-of-fold SHAP column per readable
feature, `clip` (sum over the CLIP PCs), `bias`. Values are in log-odds; a row sums to the full
model's log-odds for that shot.

## RQ2: trailers (`data/features/trailers.parquet`, C)

One row per trailer. Ids: `imdb_id`, `title`, `year`, `decade`, `genres`, `primary_genre`.

| group | columns |
|---|---|
| pacing | `n_shots`, `duration` (end of last shot, s), `cuts_per_min` (shots per minute), `shot_len_median`, `shot_len_cv` (std / mean), `pace_q1..q4` (shots per minute in each quarter of the trailer, a shot counted in the quarter holding its midpoint), `pace_accel` (slope of log shot length against `trailer_pos`; negative = shots get shorter toward the end; empty under 3 shots) |
| audio | from the mono WAV, RMS in dB relative to the trailer's loudest frame (so 0 = loudest): `rms_mean_db`, `rms_dynamic_range` (95th minus 5th percentile dB), `quiet_frac` (share of frames below -40 dB), `onset_rate` (onsets per minute), `centroid_mean` (spectral centroid, Hz), `rms_q1..q4` (mean dB per quarter) |
| story | `match_rate` (matched shots over all shots, blank ones included), then over matched shots' `film_story_pos`: `story_mean`, `story_median`, `story_max`, `story_std`, `first_act_share` (< 0.25), `final_act_share` (> 0.75), `order_corr` (Spearman of trailer position vs story position; empty under 3 matched shots) |
| dialogue | over each distinct matched film shot, from the film subtitles overlapping it: `dlg_words_per_shot` (mean words), `dlg_sentiment_mean`, `dlg_sentiment_std` (VADER compound, shots with text only), `dlg_neg_share` (share with compound < -0.05) |
| visual | means over trailer shots: `scale_width_mean`, `c_<concept>_mean` for the 20 concepts (e.g. `c_gun_mean`) |
| speech | copied from `av_sync.csv`: `speech_share`, `words_per_min`, `lines_found`, `sync_rate` (only when that file exists) |

**`trailer_curves.json`** (C, `trailer_features`; the views no longer read it, see `cache/curves/`): object keyed by IMDb ID; each has `rms_db` (100 values, dB relative to
the loudest frame, interpolated over trailer time) and `cuts` (100 counts of shots whose midpoint falls
in each bin).

**`case_studies.json`** (C, `case_studies`): one object per angle: `order` (`order_corr`), `ending`
(`final_act_share`), `sync` (`sync_rate`), `model` (`model_precision`: per-film precision of the
`full` model at trailer length). Each has `column`, `n_trailers` (trailers with enough data: 15+
matched shots, 5+ paired lines, or 10+ used shots for `model`) and two lists of the top and bottom
`--n` (default 5) trailers, each `imdb_id`, `title`, `genre`, `value`.

## Clusters

**`clusters.csv`** (C): one row per trailer. `imdb_id`, `title`, `year`, `primary_genre`, k-means
label `cluster_<group>` for `all`, `pacing`, `audio`, `story`, `dialogue`, `visual`, `speech`, and `umap_x`,
`umap_y` (UMAP of all features, standardized, medians imputed).

**`cluster_report.json`** (C): a JSON list, one object per feature set (`all` first, then the 6
groups, `speech` last). Keys: `modality`, `n_features`, `best_k`, `silhouette` (k-means silhouette for k = 2 to 8,
keyed by k as a string), `stability_ari_mean`, `stability_ari_std` (50 refits on 80% subsamples),
`hdbscan_n_clusters`, `hdbscan_noise_frac`, and `genre`, `decade`, each `{nmi, null_p95, p}` (`p` =
share of 200 shuffles with NMI at least as high).

## Summary and labels

**`summary.json`** (C): `funnel` (film counts per step), `corpus` (`trailers`, `trailer_len_median`,
`shots_per_trailer_median`, `shot_len_median`), `story_depth` (`pooled_first_quarter`,
`pooled_final_quarter`, `median_trailer_final_act_share`), `alignment` (`trailer_shots`, `blank`,
`matched`, `match_rate_per_trailer_median`). The numbers behind the figures.

**`labels/label_sample.json`** (C): list of the sampled IMDb IDs. **`labels/label_trailers.csv`** (C,
kept by hand): `imdb_id`, `youtube_id` of the trailer each labeled film had when it was labeled.
`label_eval.load_gold` drops a film's labels when its trailer in `trailer_status.csv` differs (so far
Miami Vice's 71 labels, after its swap). Add a row when labeling a new film. **`labels/gold_<name>.json`** (C,
`gold_ced.json` so far): `{labeler, labels}`, labels keyed `imdb_id:shot_idx` with a film shot index, `s`, `n` or
`u`. **`labels/label_tool.html`** (G): the page, with each shot's top-5 candidates embedded.
**`labels/img/{id}/`** (G): images the page shows, copied by `label_tool`: `t_shot_XXXX_img_{0,1,2}.jpg`
(trailer keyframes) and `f_shot_XXXX_img_1.jpg` (film candidates, re-saved as JPEG).
**`labels/swap_review.html`** (G, `swap_review`): the swap review page, with every replaced trailer
(`source=search`) embedded: `id`, `film`, `old` (MovieNet link's YouTube title, "dead link" or "no
MovieNet link"), `old_url`, `new`, `channel`, `url`, and 8 `pairs` (`t`, `f` image paths, `time` in
the trailer in s, `pos` story position of the film shot). **`labels/swap_img/{id}/`** (G): `t_XXXX.jpg`
(trailer keyframe of shot XXXX) and `f_XXXX.jpg` (matched film keyframe). **`labels/swap_review_<name>.json`**
(Export from the page): `{reviewer, verdicts}`, verdicts keyed by IMDb ID: `g` good, `b` bad, `u` unsure.
The page holds every replaced core trailer and a random sample of new films (`--new`, default 30; their
`film` ends in "(new film)"), each with `share` (share of non-blank shots above the impostor cutoff),
sorted by `share`, lowest first.
**`labels/audit_review.html`** (G, `audit_review`): the audit review page, every row of
`trailer_audit.csv` with a flag embedded: `id`, `film`, `title`, `channel`, `flags`, `rule` (the audit's
`swap`), `lang`, `want` (expected language), `span`, `url`; plays the trailer from `../data/trailers/`.
**`labels/audit_review_<name>.json`** (C, Export from the page): `{reviewer, verdicts}`,
verdicts keyed by IMDb ID: `swap` or `keep`. `audit_review.decisions()` applies them over the audit's
`swap` column, later files winning.
**`labels/split_review.html`** (G, `resplit_clip review`): the split review page, a sample of
proposed cuts with the trailer clip and 4 frames each (shot start, just before, just after, shot end)
from **`labels/split_img/`** (G). **`labels/split_review_<name>.json`** (Export from the page):
`{reviewer, verdicts}`, verdicts keyed `imdb_id:cut_frame`: `y` real cut, `n` same shot (key `x`), `u` unsure.

## Trailer View data (`viz/data/`, G)

Written by `viewer_export`, read by `viz/index.html`. Compact JSON, NaN written as `null`.

**`index.json`**: list, one entry per trailer: `id`, `title`, `year`, `genre` (first genre), `shots`,
`match_rate`.

**`{imdb_id}.json`**: one trailer.

| key | meaning |
|---|---|
| `id`, `title`, `year`, `genres`, `duration` | ids; `duration` = end of the last trailer shot (s) |
| `speech` | one list per transcript segment: `[t_start, t_end, matched (0/1), film position of the matched line (0 to 1) or null, text, words as [start, end, word]]`; empty without a transcript |
| `curves` | `rms_db`, `cuts` as in `trailer_curves.json`, computed by `viewer_export` (cached in `cache/curves/`) |
| `shots` | one list per trailer shot: `[t_start, t_end, matched (0/1), film_shot or -1, score, blank (0/1)]` |
| `shot_colors` | average color (`#rrggbb`) of each trailer shot's middle keyframe |
| `film.runtime` | film runtime (s, end of the last film shot) |
| `film.n`, `film.story_pos`, `film.shot_len`, `film.in_trailer` | per film shot, from `film_shots.parquet` |
| `film.start`, `film.end` | film shot extent as a fraction of runtime (for the barcode) |
| `film.color` | average color of each film shot's middle keyframe; `null` if the image can't be read |
| `film.trailerness`, `film.trailerness_readable` | out-of-fold scores of the `full` and `readable` models (only if `model` was run); `null` for logo/credit edge shots the model skips |
| `film.shap_groups`, `film.shap` | group order and per film shot SHAP summed per group (`position`, `dialogue`, `scale`, `content`, `editing`, `clip`) |
| `film.detail_features`, `film.detail` | readable feature names, and for in-trailer shots plus the top `--detail` (default 60) by `full` score, `[value, SHAP]` per feature, keyed by shot index |

**`{imdb_id}.vtt`**: WebVTT captions from the Whisper transcript, cues split at about 6 s, 12 words
or a pause over 1.5 s. Written only when the transcript exists.

**`rq1.json`** (from `rq1_export`, read by `viz/rq1.html`): `n_shots`, `n_films`, `positive_rate`
(edge shots excluded), then lift tables `story` (20 bins of story position, `level` = bin center),
`scale` (dominant scale class), `length` and `distinct` (within-film quintiles of `rel_shot_len`,
`rel_distinct`), `dialogue` (has / no dialogue), `content` (per concept, shots in the film's top 10%
for it). Each row: `level`, `lift` (pick rate over the overall pick rate), `lo`, `hi` (95% interval,
films resampled, 300 draws by default), `n` (shots). `models`: `name`, `lift` from `model_eval.json`
`k0`. Only if `model_ebm.json` exists: `ebm`, the 6 EBM curves with the highest importance, each `name`
(readable label), `x` (bin edges), `y` (odds multiplier, exp of the score), cut to the feature's 1st to
99th percentile; `ebm_lift` (the EBM's lift).

## Caches (G, under `$TRAILERCUT_DATA/cache/`)

`clip/<tag>/{film,trailer}/{id}.npy` (float16, [shots, 3, d], L2-normalized),
`trailer_frames/{id}/shot_XXXX_img_{0,1,2}.jpg` (320x180), `transnet/{id}_{video bytes}.npy` (`trailer_prep`:
TransNetV2's per-frame cut probability, float32; read by `resplit`), `trailer_audio/{id}.wav` (mono 22,050 Hz),
`trailer_asr/{id}.json` (`trailer_asr`: `language`, `language_p`, `segments` with `start`, `end`,
`text`, `words` [{`start`, `end`, `word`, `p`}]), `models/` (SSCD weights for `rerank`),
`replaced/{id}/` (a trailer's old files while `recover --ids` or `swap apply` replaces it, plus `_paths.json`; written once, removed after a restore),
`yt_meta/{youtube_id}.json` (`trailer_audit`: `title`, `channel`, `duration`, `height`; failed
requests aren't cached),
`concept_text_<tag>.npy`, `blank_emb_<tag>.npy`, `alignment_null_<tag>.npy`,
`shot_colors/{id}.json` (`film`, `trailer`: lists of `#rrggbb` per shot, from `viewer_export`; reused
while the shot counts match), `curves/{id}.json` (`viewer_export`: `rms_db`, `cuts` as in
`trailer_curves.json`, plus `stamp`, the trailer shot table's modification time; recomputed when it
changes), `sheets/swaps_NN.jpg` (`swap sheet`, no longer run by `curate.sh`: per swapped film, old
and new YouTube title and the new trailer's 4 best-matching shots next to the film keyframes they
matched; film frames, so never committed).

Swap downloads go to `$TRAILERCUT_DATA/trailers/candidates/{id}__{youtube_id}.mp4` (G).

## Dataset release (`data/release/`, G, `release`)

Links and annotations only, to upload as a dataset. Not written to the repo.

| file | what |
|---|---|
| `trailers.csv` | one row per film with a working trailer (`status=ok`), only the link in use: `imdb_id`, `film` (MovieNet title), `year`, `url` (YouTube link), `title`, `channel`, `duration_s` (YouTube metadata, from `cache/yt_meta/` and the candidate logs, falling back to the status log), `trailer_shots`, `match_share` (non-blank shots matched), `film_span` (10th to 90th percentile of matched shots' story position), `checked` (build date). Which links replaced MovieNet's, and why, stays in `data/ids/` and is not released |
| `trailer_shots.parquet`, `alignment.parquet`, `dialogue_alignment.parquet` | the per-trailer CSVs stacked, with an `imdb_id` column |
| `av_sync.csv` | copy of `data/features/av_sync.csv` |
| `hand_labels.csv` | `imdb_id`, `trailer_shot`, `label`, `labeler` (`labeler_1`, ... by file order, names removed) |
| `README.md` | dataset card: files, how it was made, limitations, terms (annotations under CC BY-NC 4.0); numbers (films, trailer shots, labels, precision/recall, cutoff, glued-shot share) filled from the tables, `alignment_eval.json` and `alignment_diagnostics.json`; YAML header with one Hugging Face viewer `configs` entry per table |
