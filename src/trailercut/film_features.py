"""One row per film shot: readable features + trailerness label (RQ1 table).

Feature groups (dropped one at a time in the ablation):
  position  story_pos, act, synopsis paragraph position
  dialogue  words, speech rate, sentiment of overlapping subtitles
  scale     zero-shot shot-scale probabilities (concepts.py)
  content   zero-shot content concepts (concepts.py)
  editing   shot length, within-shot visual change
  clip      raw embedding (kept separate; not in this table, loaded from cache)

Label: in_trailer = some trailer shot matched to this film shot above the impostor threshold.

  python -m trailercut.film_features
Writes data/features/film_shots.parquet (committed).
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from . import concepts
from . import movienet_io as mn
from .paths import CLIP, FEATURES

VADER = SentimentIntensityAnalyzer()


def dialogue_features(shots, subs):
    words = np.zeros(len(shots))
    sent = np.full(len(shots), np.nan)
    if len(subs):
        s0, s1 = subs.t_start.values, subs.t_end.values
        for i, (a, b) in enumerate(zip(shots.t_start.values, shots.t_end.values)):
            hit = (s0 < b) & (s1 > a)
            if hit.any():
                text = " ".join(subs.text.values[hit])
                words[i] = len(text.split())
                sent[i] = VADER.polarity_scores(text)["compound"]
    return dict(dlg_words=words, dlg_rate=words / shots.shot_len.clip(lower=0.5).values,
                dlg_has=(words > 0).astype(float), dlg_sentiment=sent)


def story_features(shots, story):
    act = np.digitize(shots.story_pos.values, [0.25, 0.75])  # classic 1/4, 1/2, 1/4 split
    para = np.full(len(shots), np.nan)
    if len(story):
        for p in story.itertuples():
            para[p.shot_lo: p.shot_hi + 1] = p.para_idx / max(len(story) - 1, 1)
    return dict(act=act, synopsis_pos=para)


def labels(imdb_id, n):
    path = FEATURES / "alignment" / f"{imdb_id}.csv"
    if not path.exists():
        return None
    al = pd.read_csv(path)
    cnt = np.bincount(al.loc[al.matched, "film_shot"], minlength=n)[:n]
    return cnt


def film_table(imdb_id):
    shots = mn.shots(imdb_id)
    cnt = labels(imdb_id, len(shots))
    if cnt is None:
        return None
    emb = np.load(CLIP / "film" / f"{imdb_id}.npy").astype(np.float32)
    df = shots[["shot_idx", "t_start", "t_end", "shot_len", "story_pos"]].copy()
    df.insert(0, "imdb_id", imdb_id)
    for k, v in {**story_features(shots, mn.story(imdb_id)),
                 **dialogue_features(shots, mn.subtitles(imdb_id)),
                 **concepts.scores(emb)}.items():
        df[k] = v
    # 1 - cos(first keyframe, last keyframe): cheap proxy for camera/subject motion
    df["visual_change"] = 1 - (emb[:, 0] * emb[:, 2]).sum(-1)
    df["trailer_hits"] = cnt
    df["in_trailer"] = (cnt > 0).astype(int)
    return df


def main():
    ids = sorted(p.stem for p in (FEATURES / "alignment").glob("*.csv"))
    tables = [t for t in (film_table(i) for i in tqdm(ids)) if t is not None]
    df = pd.concat(tables, ignore_index=True)
    for c in df.select_dtypes("float64"):
        df[c] = df[c].astype("float32")
    df.to_parquet(FEATURES / "film_shots.parquet", index=False)
    print(df.shape, "positive rate", round(df.in_trailer.mean(), 4))


if __name__ == "__main__":
    main()
