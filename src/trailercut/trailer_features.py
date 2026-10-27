"""One row per trailer: pacing, audio, story sampling, dialogue, visual concepts (RQ2 table).

Dialogue here is a proxy: sentiment of the *film* subtitles under the matched source shots.
Trailers re-cut dialogue, so a Whisper transcript of the trailer audio is the planned upgrade.

Also writes per-trailer curves for the Trailer View (RMS energy and cut rate on a 100-bin grid).

  python -m trailercut.trailer_features
Writes data/features/trailers.parquet and data/features/trailer_curves.json (both committed).
"""
import json

import librosa
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from . import concepts
from . import movienet_io as mn
from .film_features import VADER
from .paths import CACHE, FEATURES

AUDIO = CACHE / "trailer_audio"
GRID = 100

GROUPS = {
    "pacing": ["n_shots", "duration", "cuts_per_min", "shot_len_median", "shot_len_cv",
               "pace_q1", "pace_q2", "pace_q3", "pace_q4", "pace_accel"],
    "audio": ["rms_mean_db", "rms_dynamic_range", "quiet_frac", "onset_rate", "centroid_mean",
              "rms_q1", "rms_q2", "rms_q3", "rms_q4"],
    "story": ["match_rate", "story_mean", "story_median", "story_max", "story_std",
              "first_act_share", "final_act_share", "order_corr"],
    "dialogue": ["dlg_words_per_shot", "dlg_sentiment_mean", "dlg_sentiment_std", "dlg_neg_share"],
    "visual": ["scale_width_mean"] + [f"{c}_mean" for c in concepts.CONTENT],
}


def quarters(pos, values, agg=np.mean):
    q = np.minimum((pos * 4).astype(int), 3)
    return [agg(values[q == i]) if (q == i).any() else np.nan for i in range(4)]


def pacing(sh):
    dur = sh.t_end.max()
    pos = sh.trailer_pos.values
    lens = sh.shot_len.values
    cut_rate = [((pos >= i / 4) & (pos < (i + 1) / 4)).sum() / (dur / 4) * 60 for i in range(4)]
    return dict(n_shots=len(sh), duration=dur, cuts_per_min=len(sh) / dur * 60,
                shot_len_median=np.median(lens), shot_len_cv=lens.std() / lens.mean(),
                pace_q1=cut_rate[0], pace_q2=cut_rate[1], pace_q3=cut_rate[2], pace_q4=cut_rate[3],
                # negative = shots get shorter toward the end (the classic trailer build)
                pace_accel=np.polyfit(pos, np.log(lens), 1)[0] if len(sh) > 2 else np.nan)


def audio(imdb_id, dur):
    y, sr = librosa.load(AUDIO / f"{imdb_id}.wav", sr=22050)
    rms = librosa.feature.rms(y=y)[0]
    db = librosa.amplitude_to_db(rms + 1e-6, ref=np.max)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    pos = np.linspace(0, 1, len(db), endpoint=False)
    q = quarters(pos, db)
    curve = np.interp(np.linspace(0, 1, GRID), pos, db)
    feats = dict(rms_mean_db=db.mean(), rms_dynamic_range=np.percentile(db, 95) - np.percentile(db, 5),
                 quiet_frac=(db < -40).mean(), onset_rate=len(onsets) / dur * 60, centroid_mean=cent.mean(),
                 rms_q1=q[0], rms_q2=q[1], rms_q3=q[2], rms_q4=q[3])
    return feats, curve


def story_and_dialogue(imdb_id):
    al = pd.read_csv(FEATURES / "alignment" / f"{imdb_id}.csv")
    m = al[al.matched]
    sp = m.film_story_pos.values
    out = dict(match_rate=al.matched.mean(), story_mean=np.mean(sp) if len(sp) else np.nan,
               story_median=np.median(sp) if len(sp) else np.nan, story_max=sp.max() if len(sp) else np.nan,
               story_std=sp.std() if len(sp) else np.nan,
               first_act_share=(sp < 0.25).mean() if len(sp) else np.nan,
               final_act_share=(sp > 0.75).mean() if len(sp) else np.nan,
               order_corr=spearmanr(m.trailer_pos, sp)[0] if len(sp) > 2 else np.nan)
    shots, subs = mn.shots(imdb_id), mn.subtitles(imdb_id)
    words, sents = [], []
    for fs in m.film_shot.unique():
        a, b = shots.t_start[fs], shots.t_end[fs]
        hit = subs[(subs.t_start < b) & (subs.t_end > a)] if len(subs) else subs
        text = " ".join(hit.text) if len(hit) else ""
        words.append(len(text.split()))
        if text:
            sents.append(VADER.polarity_scores(text)["compound"])
    sents = np.array(sents)
    out.update(dlg_words_per_shot=np.mean(words) if words else 0.0,
               dlg_sentiment_mean=sents.mean() if len(sents) else np.nan,
               dlg_sentiment_std=sents.std() if len(sents) else np.nan,
               dlg_neg_share=(sents < -0.05).mean() if len(sents) else np.nan)
    return out


def visual(imdb_id):
    emb = np.load(CACHE / "clip" / "trailer" / f"{imdb_id}.npy")
    s = concepts.scores(emb)
    return {"scale_width_mean": s["scale_width"].mean(), **{f"{c}_mean": s[c].mean() for c in concepts.CONTENT}}


def main():
    ids = sorted(p.stem for p in (FEATURES / "alignment").glob("*.csv"))
    rows, curves = [], {}
    for imdb_id in tqdm(ids):
        sh = pd.read_csv(FEATURES / "trailer_shots" / f"{imdb_id}.csv")
        m = mn.meta(imdb_id)
        p = pacing(sh)
        a, rms_curve = audio(imdb_id, p["duration"])
        cut_curve = np.histogram(sh.trailer_pos, bins=GRID, range=(0, 1))[0]
        curves[imdb_id] = dict(rms_db=np.round(rms_curve, 2).tolist(), cuts=cut_curve.tolist())
        rows.append(dict(imdb_id=imdb_id, title=m["title"], year=m["year"],
                         genres="|".join(m["genres"]), primary_genre=m["genres"][0] if m["genres"] else None,
                         **p, **a, **story_and_dialogue(imdb_id), **visual(imdb_id)))
    df = pd.DataFrame(rows)
    df["decade"] = (df.year // 10 * 10).astype("Int64")
    df.to_parquet(FEATURES / "trailers.parquet", index=False)
    (FEATURES / "trailer_curves.json").write_text(json.dumps(curves))
    print(df.shape)


if __name__ == "__main__":
    main()
