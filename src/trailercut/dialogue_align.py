"""Align trailer dialogue to the film through its subtitles (the audio side of alignment).

Each Whisper segment of the trailer is compared with every subtitle line of the film and with runs of
2 to 3 consecutive lines (subtitles split sentences). Text is normalized and scored with fuzzy
matching, so small transcription errors don't break a match. A segment counts as matched only if
  - it has enough words to be specific (short lines like "Get in here!" occur all over a film),
  - its best score clears MIN_SCORE, and
  - it beats the best match elsewhere in the film (more than AWAY seconds from it) by MIN_MARGIN.
The film time of the matched subtitle lines then says which moment of the film this piece of the
trailer's soundtrack comes from, without looking at pixels.

Outputs (committed; times and scores only, no dialogue text)
  data/features/dialogue_alignment/{imdb_id}.csv  one row per trailer speech segment:
      seg_idx, t_start, t_end (trailer), n_words, film_t_start, film_t_end, score, margin, matched

  python -m trailercut.dialogue_align
"""
import json
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from . import movienet_io as mn
from .paths import CACHE, FEATURES

ASR = CACHE / "trailer_asr"
OUT = FEATURES / "dialogue_alignment"
MIN_WORDS, MIN_SCORE, MIN_MARGIN, AWAY = 3, 90, 10, 30


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", s.lower())).strip()


def windows(subs):
    """Subtitle lines and runs of 2 to 3 consecutive lines: (text, t_start, t_end)."""
    txt = [norm(t) for t in subs.text]
    out = []
    for n in (1, 2, 3):
        for i in range(len(txt) - n + 1):
            out.append((" ".join(txt[i:i + n]), subs.t_start.iloc[i], subs.t_end.iloc[i + n - 1]))
    return out


def match(seg, wins):
    q = norm(seg)
    scores = np.array([fuzz.ratio(q, w) if len(w) < 1.6 * len(q) else fuzz.partial_ratio(q, w) for w, _, _ in wins])
    best = int(scores.argmax())
    t0 = wins[best][1]
    far = np.array([abs(w[1] - t0) > AWAY for w in wins])
    second = scores[far].max() if far.any() else 0
    return best, float(scores[best]), float(scores[best] - second)


def align_film(imdb_id):
    asr = json.loads((ASR / f"{imdb_id}.json").read_text())
    subs = mn.subtitles(imdb_id)
    rows = []
    wins = windows(subs) if len(subs) else []
    for k, s in enumerate(asr["segments"]):
        n = len(norm(s["text"]).split())
        row = dict(seg_idx=k, t_start=s["start"], t_end=s["end"], n_words=n,
                   film_t_start=np.nan, film_t_end=np.nan, score=np.nan, margin=np.nan, matched=False)
        if wins and n >= 2:
            b, sc, mg = match(s["text"], wins)
            row.update(film_t_start=wins[b][1], film_t_end=wins[b][2], score=sc, margin=mg,
                       matched=n >= MIN_WORDS and sc >= MIN_SCORE and mg >= MIN_MARGIN)
        rows.append(row)
    cols = ["seg_idx", "t_start", "t_end", "n_words", "film_t_start", "film_t_end", "score", "margin", "matched"]
    return pd.DataFrame(rows, columns=cols), asr["language"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in ASR.glob("*.json"))
    stats = []
    for i in tqdm(ids):
        df, lang = align_film(i)
        df.round(3).to_csv(OUT / f"{i}.csv", index=False)
        stats.append(dict(imdb_id=i, language=lang, segments=len(df), matched=int(df.matched.sum())))
    s = pd.DataFrame(stats)
    print(f"{len(s)} trailers, {s.segments.sum()} speech segments, {s.matched.sum()} matched to subtitles "
          f"({s.matched.sum() / max(s.segments.sum(), 1):.0%}); trailers with any match: {(s.matched > 0).sum()}")


if __name__ == "__main__":
    main()
