# Data dictionary

Every file the pipeline writes that someone downstream reads, with its columns. "C" = committed,
"G" = gitignored (rebuilt by the pipeline).

## IDs and logs (`data/ids/`)

**`core_films_293.csv`** (C): the 293 core films.
`imdb_id`, `youtube_id` (MovieNet's original trailer link).

**`trailer_status.csv`** (C): one row per core film.
`imdb_id`, `youtube_id` (link actually used), `status` (`ok` / `unavailable` / `error`), `duration` (s),
`title` (YouTube title), `error` (first line of the yt-dlp error), `source` (`movienet` or `search`).

**`trailer_recovery.csv`** (C): every search candidate tried for a dead link.
`imdb_id`, `rank` (search order among filtered results, 0-based), `youtube_id`, `title`, `channel`,
`duration`, `match_share` (fraction of non-blank shots matched to the film, empty if download failed),
`accepted`.

## Trailer shots (`data/features/trailer_shots/{imdb_id}.csv`, C)

One row per TransNetV2 shot.
`shot_idx`, `frame_start`, `frame_end` (inclusive), `t_start`, `t_end` (s), `shot_len` (s),
`trailer_pos` (shot midpoint as a fraction of trailer length, 0 to 1).

## Alignment (`data/features/alignment/{imdb_id}.csv`, C)

One row per trailer shot.

| column | meaning |
|---|---|
| `shot_idx`, `t_start`, `t_end`, `shot_len`, `trailer_pos` | copied from the trailer shot table |
| `film_shot` | best-matching MovieNet film shot index |
| `score` | similarity to that shot (mean over trailer keyframes of max cosine over film keyframes) |
| `margin` | `score` minus the best score outside ±3 film shots of the best |
| `blank` | shot looks like a solid black/white/grey frame; never matched |
| `matched` | `score >= threshold` and not blank |
| `film_story_pos` | story position (0 to 1) of the matched film shot; empty if not matched |
| `top5`, `top5_scores` | space-separated top-5 film shot indices and scores |

**`alignment_threshold.json`** (C): `fpr`, `threshold`, `n_null`, `null_quantiles` (impostor null).

## RQ1: film shots (`data/features/film_shots.parquet`, G, ~65 MB)

One row per film shot of every film with an aligned trailer.

| group | columns |
|---|---|
| ids | `imdb_id`, `shot_idx`, `t_start`, `t_end` |
| position | `story_pos` (0 to 1), `act` (0/1/2 from a 1/4, 1/2, 1/4 split), `synopsis_pos` (synopsis paragraph position, 0 to 1, empty outside the story annotation) |
| dialogue | `dlg_words`, `dlg_rate` (words/s), `dlg_has`, `dlg_sentiment` (VADER compound, empty without subtitles) |
| scale | `scale_ecu`, `scale_cu`, `scale_ms`, `scale_fs`, `scale_ls` (zero-shot probabilities ×100), `scale_width` |
| content | `c_explosion`, `c_gun`, `c_fight`, `c_chase`, `c_running`, `c_kiss`, `c_crying`, `c_scream`, `c_laugh`, `c_crowd`, `c_night`, `c_day_outdoor`, `c_creature`, `c_space`, `c_water`, `c_city`, `c_interior`, `c_text_card`, `c_logo`, `c_blood` |
| editing | `shot_len` (s), `visual_change` (1 minus cosine of first and last keyframe) |
| label | `trailer_hits` (matched trailer shots pointing here), `in_trailer` (hits > 0) |

## RQ1 model outputs

**`model_report.json`** (C): `n_shots`, `n_films`, `positive_rate`, `folds`, `clip_pcs`, `groups`,
`models` (per model: `ap`, `ap_film`, `r_prec`, `ap_lenadj`), `ablation_drop` (full minus each
ablation, per metric), `shap_mean_abs_by_group`, `shap_mean_abs_top`.

**`model_preds.parquet`** (G): `imdb_id`, `shot_idx`, `shot_len`, `in_trailer`, then one out-of-fold
score column per model (`random`, `position_only`, `length_only`, `clip_only`, `readable`, `full`,
`full_minus_<group>`).

**`model_shap.parquet`** (G): `imdb_id`, `shot_idx`, one out-of-fold SHAP column per readable
feature, `clip` (sum over the CLIP PCs), `bias`. Values are in log-odds; a row sums to the full
model's log-odds for that shot.

## RQ2: trailers (`data/features/trailers.parquet`, C)

One row per trailer. Ids: `imdb_id`, `title`, `year`, `decade`, `genres`, `primary_genre`.

| group | columns |
|---|---|
| pacing | `n_shots`, `duration`, `cuts_per_min`, `shot_len_median`, `shot_len_cv`, `pace_q1..q4` (cuts/min per trailer quarter), `pace_accel` |
| audio | `rms_mean_db`, `rms_dynamic_range`, `quiet_frac`, `onset_rate`, `centroid_mean`, `rms_q1..q4` |
| story | `match_rate`, `story_mean`, `story_median`, `story_max`, `story_std`, `first_act_share`, `final_act_share`, `order_corr` (trailer order vs film order) |
| dialogue | `dlg_words_per_shot`, `dlg_sentiment_mean`, `dlg_sentiment_std`, `dlg_neg_share` (film subtitles under matched shots) |
| visual | `scale_width_mean`, `c_<concept>_mean` for the 20 concepts |

**`trailer_curves.json`** (C): per trailer, `rms_db` and `cuts` on a 100-bin grid over trailer time.

## Clusters

**`clusters.csv`** (C): `imdb_id`, `title`, `year`, `primary_genre`, `cluster_<group>` for `all`,
`pacing`, `audio`, `story`, `dialogue`, `visual`, and `umap_x`, `umap_y` (UMAP of all features).

**`cluster_report.json`** (C): one entry per feature group with the chosen k, silhouette, bootstrap
ARI, HDBSCAN result and genre/decade NMI with their permutation null.

## Summary and labels

**`summary.json`** (C): `funnel`, `corpus`, `story_depth`, `alignment` (the numbers behind the figures).

**`labels/label_sample.json`** (C): trailers and shots in the labeling sample with their top-5
candidates. **`labels/gold_<name>.json`** (C): one labeler's answers. **`label_tool.html`** (G).

## Caches (G, under `$TRAILERCUT_DATA/cache/`)

`clip/<tag>/{film,trailer}/{id}.npy` (float16, [shots, 3, d], L2-normalized),
`trailer_frames/{id}/shot_XXXX_img_{0,1,2}.jpg` (320x180), `trailer_audio/{id}.wav` (mono 22,050 Hz),
`concept_text_<tag>.npy`, `blank_emb_<tag>.npy`, `alignment_null_<tag>.npy`.
