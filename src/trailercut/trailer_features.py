
"""Per-trailer features: pacing, audio, story sampling, dialogue and visual concepts.

Full collection:
  python -m trailercut.trailer_features

Incremental:
  python -m trailercut.trailer_features --ids tt0032138

Incremental mode updates only the requested rows in trailers.parquet
and only the requested entries in trailer_curves.json.
"""
import argparse
import json
import os

import librosa
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from . import concepts
from . import movienet_io as mn
from .film_features import VADER
from .paths import CACHE, CLIP, FEATURES

AUDIO = CACHE / "trailer_audio"
GRID = 100

PARQUET = FEATURES / "trailers.parquet"
CURVES_JSON = FEATURES / "trailer_curves.json"
ALIGNMENT = FEATURES / "alignment"
SHOTS = FEATURES / "trailer_shots"
CLIP_TRAILER = CLIP / "trailer"

GROUPS = {
    "pacing": ["n_shots", "duration", "cuts_per_min", "shot_len_median", "shot_len_cv",
               "pace_q1", "pace_q2", "pace_q3", "pace_q4", "pace_accel"],
    "audio": ["rms_mean_db", "rms_dynamic_range", "quiet_frac", "onset_rate", "centroid_mean",
              "rms_q1", "rms_q2", "rms_q3", "rms_q4"],
    "story": ["match_rate", "story_mean", "story_median", "story_max", "story_std",
              "first_act_share", "final_act_share", "order_corr"],
    "dialogue": ["dlg_words_per_shot", "dlg_sentiment_mean", "dlg_sentiment_std", "dlg_neg_share"],
    "visual": ["scale_width_mean"] + [f"{c}_mean" for c in concepts.CONTENT],
    "speech": ["speech_share", "words_per_min", "lines_found", "sync_rate"],
}


def quarters(pos, values, agg=np.mean):
    q = np.minimum((pos * 4).astype(int), 3)
    return [
        agg(values[q == i]) if (q == i).any() else np.nan
        for i in range(4)
    ]


def pacing(sh):
    dur = sh.t_end.max()
    pos = sh.trailer_pos.values
    lens = sh.shot_len.values

    if not np.isfinite(dur) or dur <= 0 or len(sh) == 0:
        raise ValueError("Invalid or empty shot table")

    cut_rate = [
        ((pos >= i / 4) & (pos < (i + 1) / 4)).sum()
        / (dur / 4) * 60
        for i in range(4)
    ]

    return dict(
        n_shots=len(sh),
        duration=dur,
        cuts_per_min=len(sh) / dur * 60,
        shot_len_median=np.median(lens),
        shot_len_cv=lens.std() / lens.mean(),
        pace_q1=cut_rate[0],
        pace_q2=cut_rate[1],
        pace_q3=cut_rate[2],
        pace_q4=cut_rate[3],
        pace_accel=(
            np.polyfit(pos, np.log(lens), 1)[0]
            if len(sh) > 2 and (lens > 0).all()
            else np.nan
        ),
    )


def audio(imdb_id, dur):
    wav = AUDIO / f"{imdb_id}.wav"

    if not wav.exists():
        raise FileNotFoundError(f"Trailer audio missing: {wav}")

    y, sr = librosa.load(wav, sr=22050)

    if len(y) == 0:
        raise ValueError(f"Trailer audio is empty: {wav}")

    rms = librosa.feature.rms(y=y)[0]
    db = librosa.amplitude_to_db(rms + 1e-6, ref=np.max)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    pos = np.linspace(0, 1, len(db), endpoint=False)
    q = quarters(pos, db)
    curve = np.interp(np.linspace(0, 1, GRID), pos, db)

    feats = dict(
        rms_mean_db=db.mean(),
        rms_dynamic_range=np.percentile(db, 95) - np.percentile(db, 5),
        quiet_frac=(db < -40).mean(),
        onset_rate=len(onsets) / dur * 60,
        centroid_mean=cent.mean(),
        rms_q1=q[0],
        rms_q2=q[1],
        rms_q3=q[2],
        rms_q4=q[3],
    )

    return feats, curve


def story_and_dialogue(imdb_id):
    al = pd.read_csv(ALIGNMENT / f"{imdb_id}.csv")

    m = al[al.matched]
    sp = m.film_story_pos.values

    out = dict(
        match_rate=al.matched.mean(),
        story_mean=np.mean(sp) if len(sp) else np.nan,
        story_median=np.median(sp) if len(sp) else np.nan,
        story_max=sp.max() if len(sp) else np.nan,
        story_std=sp.std() if len(sp) else np.nan,
        first_act_share=(sp < 0.25).mean() if len(sp) else np.nan,
        final_act_share=(sp > 0.75).mean() if len(sp) else np.nan,
        order_corr=(
            spearmanr(m.trailer_pos, sp)[0]
            if len(sp) > 2 else np.nan
        ),
    )

    shots, subs = mn.shots(imdb_id), mn.subtitles(imdb_id)
    words, sents = [], []

    for fs in m.film_shot.unique():
        a, b = shots.t_start[fs], shots.t_end[fs]

        hit = (
            subs[(subs.t_start < b) & (subs.t_end > a)]
            if len(subs) else subs
        )

        text = " ".join(hit.text) if len(hit) else ""
        words.append(len(text.split()))

        if text:
            sents.append(VADER.polarity_scores(text)["compound"])

    sents = np.array(sents)

    out.update(
        dlg_words_per_shot=np.mean(words) if words else 0.0,
        dlg_sentiment_mean=sents.mean() if len(sents) else np.nan,
        dlg_sentiment_std=sents.std() if len(sents) else np.nan,
        dlg_neg_share=(sents < -0.05).mean() if len(sents) else np.nan,
    )

    return out


def visual(imdb_id):
    path = CLIP_TRAILER / f"{imdb_id}.npy"

    if not path.exists():
        raise FileNotFoundError(
            f"Trailer CLIP embedding missing: {path}. "
            "Run the trailer embedding step first."
        )

    emb = np.load(path)
    scores = concepts.scores(emb)

    return {
        "scale_width_mean": scores["scale_width"].mean(),
        **{
            f"{c}_mean": scores[c].mean()
            for c in concepts.CONTENT
        },
    }


def feature_row(imdb_id):
    """Calculate one complete trailer feature row and its curve."""
    shots_path = SHOTS / f"{imdb_id}.csv"
    alignment_path = ALIGNMENT / f"{imdb_id}.csv"

    for path in (shots_path, alignment_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")

    sh = pd.read_csv(shots_path)
    m = mn.meta(imdb_id)

    p = pacing(sh)
    a, rms_curve = audio(imdb_id, p["duration"])

    cut_curve = np.histogram(
        sh.trailer_pos,
        bins=GRID,
        range=(0, 1),
    )[0]

    row = dict(
        imdb_id=imdb_id,
        title=m["title"],
        year=m["year"],
        genres="|".join(m["genres"]),
        primary_genre=m["genres"][0] if m["genres"] else None,
        **p,
        **a,
        **story_and_dialogue(imdb_id),
        **visual(imdb_id),
    )

    row["decade"] = (
        (row["year"] // 10 * 10)
        if pd.notna(row["year"])
        else pd.NA
    )

    curve = dict(
        rms_db=np.round(rms_curve, 2).tolist(),
        cuts=cut_curve.tolist(),
    )

    return row, curve


def atomic_parquet(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.parquet")

    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.json")

    try:
        tmp.write_text(json.dumps(data, allow_nan=False))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def update_outputs(rows, new_curves, selected_ids, full_run=False):
    """Update collection outputs without discarding unrelated trailers."""
    new_df = pd.DataFrame(rows)

    if full_run:
        df = new_df
        curves = new_curves
    else:
        if PARQUET.exists():
            old_df = pd.read_parquet(PARQUET)
            old_df = old_df[
                ~old_df.imdb_id.isin(selected_ids)
            ]
            df = pd.concat(
                [old_df, new_df],
                ignore_index=True,
                sort=False,
            )
        else:
            df = new_df

        df = df.sort_values("imdb_id").reset_index(drop=True)

        if CURVES_JSON.exists():
            curves = json.loads(CURVES_JSON.read_text())
        else:
            curves = {}

        curves.update(new_curves)

    atomic_parquet(df, PARQUET)
    atomic_json(curves, CURVES_JSON)

    print(
        f"Updated {len(new_df)} trailer(s). "
        f"Collection now contains {len(df)} rows."
    )



def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None,
                    help="only these trailers (default: all)")
    a = ap.parse_args()

    all_ids = sorted(
        p.stem for p in (FEATURES / "alignment").glob("*.csv")
    )
    ids = all_ids if a.ids is None else [
        i for i in a.ids if i in all_ids
    ]

    if not ids:
        print("No matching trailers found")
        return

    output = FEATURES / "trailers.parquet"
    curves_path = FEATURES / "trailer_curves.json"

    # Load existing outputs so unrequested trailers remain untouched.
    if output.exists():
        existing = pd.read_parquet(output)
        existing = existing[~existing.imdb_id.isin(ids)]
    else:
        existing = pd.DataFrame()

    curves = (
        json.loads(curves_path.read_text())
        if curves_path.exists() else {}
    )

    rows = []

    for imdb_id in tqdm(ids):
        shots_path = FEATURES / "trailer_shots" / f"{imdb_id}.csv"
        audio_path = AUDIO / f"{imdb_id}.wav"
        embedding_path = CLIP / "trailer" / f"{imdb_id}.npy"

        for path in (shots_path, audio_path, embedding_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{imdb_id}: missing prerequisite {path}"
                )

        sh = pd.read_csv(shots_path)
        m = mn.meta(imdb_id)
        p = pacing(sh)
        a, rms_curve = audio(imdb_id, p["duration"])

        cut_curve = np.histogram(
            sh.trailer_pos, bins=GRID, range=(0, 1)
        )[0]

        curves[imdb_id] = dict(
            rms_db=np.round(rms_curve, 2).tolist(),
            cuts=cut_curve.tolist()
        )

        rows.append(dict(
            imdb_id=imdb_id,
            title=m["title"],
            year=m["year"],
            genres="|".join(m["genres"]),
            primary_genre=m["genres"][0] if m["genres"] else None,
            **p,
            **a,
            **story_and_dialogue(imdb_id),
            **visual(imdb_id)
        ))

    df = pd.DataFrame(rows)
    df["decade"] = (df.year // 10 * 10).astype("Int64")

    av = FEATURES / "av_sync.csv"
    if av.exists():
        df = df.merge(
            pd.read_csv(av)[["imdb_id", *GROUPS["speech"]]],
            on="imdb_id", how="left"
        )
    else:
        df = df.drop(
            columns=[c for c in GROUPS["speech"] if c in df.columns]
        )

    combined = pd.concat([existing, df], ignore_index=True)
    combined = combined.sort_values("imdb_id")

    combined.to_parquet(output, index=False)
    curves_path.write_text(json.dumps(curves))

    print(f"Updated {len(df)} trailers; total: {len(combined)}")

if __name__ == "__main__":
    main()
