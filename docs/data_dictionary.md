# Data dictionary

Every file the pipeline writes that someone downstream reads, with its columns. "C" = committed
(not gitignored), "G" = gitignored (rebuilt by the pipeline). Files marked "not written yet" come from
steps that haven't produced a committed output so far.

## IDs and logs (`data/ids/`)

**`core_films_293.csv`** (C): the 293 core films.
`imdb_id`, `youtube_id` (MovieNet's original trailer link).

**`trailer_status.csv`** (C): one row per core film.
`imdb_id`, `youtube_id` (link actually used), `status` (`ok` / `unavailable` / `error`), `duration` (s),
`title` (YouTube title), `error` (first line of the yt-dlp error, up to 200 chars), `source` (`movienet`
or `search`). `duration` and `title` are empty when the video was already on disk. All 293 rows are `ok`
(244 `movienet`, 49 `search`).

**`trailer_recovery.csv`** (C): every search candidate tried for a dead link, appended across runs (55 rows for 49 films, 49 accepted).
`imdb_id`, `rank` (search order among filtered results, 0-based), `youtube_id`, `title`, `channel`,
`duration` (s, from search results), `match_share` (fraction of non-blank shots matched to the film,
empty if download or alignment failed), `accepted` (`match_share` at least `--min-match`, default 0.25).

## Trailer shots (`data/features/trailer_shots/{imdb_id}.csv`, C)

One row per TransNetV2 shot.
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

**`alignment_threshold.json`** (C): `fpr`, `threshold`, `source`, `n_null`, `null_quantiles` (impostor null at
0.5, 0.9, 0.99, 0.999). `source` is `impostor_null` (threshold = null quantile at `fpr`) or `manual`
(`align --threshold`; `fpr` is then the null's false-match rate at that threshold). The committed file
was written without `source`, which means `impostor_null`.

**`alignment_eval.json`** (C, not written yet; needs `labels/gold_*.json`): hand-label scores from
`label_eval`; keys listed in `labeling_guide.md`.

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

**`trailer_curves.json`** (C): object keyed by IMDb ID; each has `rms_db` (100 values, dB relative to
the loudest frame, interpolated over trailer time) and `cuts` (100 counts of shots whose midpoint falls
in each bin).

## Clusters

**`clusters.csv`** (C): one row per trailer. `imdb_id`, `title`, `year`, `primary_genre`, k-means
label `cluster_<group>` for `all`, `pacing`, `audio`, `story`, `dialogue`, `visual`, and `umap_x`,
`umap_y` (UMAP of all features, standardized, medians imputed).

**`cluster_report.json`** (C): a JSON list, one object per feature set (`all` first, then the 5
groups). Keys: `modality`, `n_features`, `best_k`, `silhouette` (k-means silhouette for k = 2 to 8,
keyed by k as a string), `stability_ari_mean`, `stability_ari_std` (50 refits on 80% subsamples),
`hdbscan_n_clusters`, `hdbscan_noise_frac`, and `genre`, `decade`, each `{nmi, null_p95, p}` (`p` =
share of 200 shuffles with NMI at least as high).

## Summary and labels

**`summary.json`** (C): `funnel` (film counts per step), `corpus` (`trailers`, `trailer_len_median`,
`shots_per_trailer_median`, `shot_len_median`), `story_depth` (`pooled_first_quarter`,
`pooled_final_quarter`, `median_trailer_final_act_share`), `alignment` (`trailer_shots`, `blank`,
`matched`, `match_rate_per_trailer_median`). The numbers behind the figures.

**`labels/label_sample.json`** (C): list of the sampled IMDb IDs. **`labels/gold_<name>.json`** (C,
none yet): `{labeler, labels}`, labels keyed `imdb_id:shot_idx` with a film shot index, `s`, `n` or
`u`. **`labels/label_tool.html`** (G): the page, with each shot's top-5 candidates embedded.
**`labels/img/{id}/`** (G): images the page shows, copied by `label_tool`: `t_shot_XXXX_img_{0,1,2}.jpg`
(trailer keyframes) and `f_shot_XXXX_img_1.jpg` (film candidates, re-saved as JPEG).

## Trailer View data (`viz/data/`, G)

Written by `viewer_export`, read by `viz/index.html`. Compact JSON, NaN written as `null`.

**`index.json`**: list, one entry per trailer: `id`, `title`, `year`, `genre` (first genre), `shots`,
`match_rate`.

**`{imdb_id}.json`**: one trailer.

| key | meaning |
|---|---|
| `id`, `title`, `year`, `genres`, `duration` | ids; `duration` = end of the last trailer shot (s) |
| `curves` | that trailer's entry from `trailer_curves.json` (`rms_db`, `cuts`) |
| `shots` | one list per trailer shot: `[t_start, t_end, matched (0/1), film_shot or -1, score, blank (0/1)]` |
| `shot_colors` | average color (`#rrggbb`) of each trailer shot's middle keyframe |
| `film.n`, `film.story_pos`, `film.shot_len`, `film.in_trailer` | per film shot, from `film_shots.parquet` |
| `film.start`, `film.end` | film shot extent as a fraction of runtime (for the barcode) |
| `film.color` | average color of each film shot's middle keyframe; `null` if the image can't be read |
| `film.trailerness`, `film.trailerness_readable` | out-of-fold scores of the `full` and `readable` models (only if `model` was run) |
| `film.shap_groups`, `film.shap` | group order and per film shot SHAP summed per group (`position`, `dialogue`, `scale`, `content`, `editing`, `clip`) |
| `film.detail_features`, `film.detail` | readable feature names, and for in-trailer shots plus the top `--detail` (default 60) by `full` score, `[value, SHAP]` per feature, keyed by shot index |

## Caches (G, under `$TRAILERCUT_DATA/cache/`)

`clip/<tag>/{film,trailer}/{id}.npy` (float16, [shots, 3, d], L2-normalized),
`trailer_frames/{id}/shot_XXXX_img_{0,1,2}.jpg` (320x180), `trailer_audio/{id}.wav` (mono 22,050 Hz),
`concept_text_<tag>.npy`, `blank_emb_<tag>.npy`, `alignment_null_<tag>.npy`,
`shot_colors/{id}.json` (`film`, `trailer`: lists of `#rrggbb` per shot, from `viewer_export`; reused
while the shot counts match).
