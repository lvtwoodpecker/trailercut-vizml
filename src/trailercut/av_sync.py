"""Sound vs picture: per-trailer speech features, the sync rate, and a check of the dialogue anchors.

A spoken line found in the film's subtitles (dialogue_align) places the trailer's sound at a film
moment. The picture alignment places the shot on screen at a film moment too. If the two are within
SYNC seconds, the line plays over its own scene; otherwise it is laid over footage from elsewhere
(voice-over, montage). Lines over unmatched pictures (title cards, new footage) are counted apart.

Per trailer (data/features/av_sync.csv, committed; also read by trailer_features as the "speech" group)
  speech_share     share of the trailer with transcribed speech
  words_per_min    transcribed words per minute of trailer
  lines_found      share of speech segments found in the film's subtitles
  sync_rate        of found lines over a matched picture, share within SYNC seconds
  n_lines_paired   found lines over a matched picture (sync_rate's denominator)

Check against the hand labels (data/features/av_sync.json): for labeled shots with a source film shot
under a found line, how often the labeled shot's film time is within SYNC seconds of the line.

  python -m trailercut.av_sync
Writes figures/av_sync.png too.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import movienet_io as mn
from .label_eval import load_gold
from .paths import CACHE, FEATURES, FIGURES

SYNC = 30


def trailer(imdb_id):
    al = pd.read_csv(FEATURES / "alignment" / f"{imdb_id}.csv")
    dl = pd.read_csv(FEATURES / "dialogue_alignment" / f"{imdb_id}.csv")
    asr = json.loads((CACHE / "trailer_asr" / f"{imdb_id}.json").read_text())
    dur = al.t_end.max()
    shots = mn.shots(imdb_id)
    film_t = ((shots.t_start + shots.t_end) / 2).values
    words = sum(len(s["words"]) for s in asr["segments"])
    paired = sync = 0
    for d in dl[dl.matched].itertuples():
        mid = (d.t_start + d.t_end) / 2
        row = al[(al.t_start <= mid) & (al.t_end > mid)]
        if len(row) and row.matched.iloc[0]:
            paired += 1
            sync += abs(film_t[int(row.film_shot.iloc[0])] - (d.film_t_start + d.film_t_end) / 2) <= SYNC
    return dict(imdb_id=imdb_id, speech_share=float((dl.t_end - dl.t_start).sum() / dur) if len(dl) else 0.0,
                words_per_min=words / dur * 60, lines_found=float(dl.matched.mean()) if len(dl) else 0.0,
                sync_rate=sync / paired if paired else np.nan, n_lines_paired=paired)


def label_check():
    gold = load_gold().drop_duplicates("key")
    gold = gold[gold.label.str.isdigit()]
    hits = n = 0
    for key, lab in zip(gold.key, gold.label):
        i, s = key.split(":")
        dl = pd.read_csv(FEATURES / "dialogue_alignment" / f"{i}.csv")
        al = pd.read_csv(FEATURES / "alignment" / f"{i}.csv")
        sh = al.iloc[int(s)]
        mid = (sh.t_start + sh.t_end) / 2
        line = dl[dl.matched & (dl.t_start <= mid) & (dl.t_end > mid)]
        if not len(line):
            continue
        fs = mn.shots(i).iloc[int(lab)]
        n += 1
        hits += abs((fs.t_start + fs.t_end) / 2 - (line.film_t_start.iloc[0] + line.film_t_end.iloc[0]) / 2) <= SYNC
    return dict(labeled_shots_under_found_line=n, line_within_sync_of_labeled_shot=hits / n if n else None)


def main():
    ids = sorted(p.stem for p in (FEATURES / "dialogue_alignment").glob("*.csv"))
    df = pd.DataFrame([trailer(i) for i in ids])
    df.round(4).to_csv(FEATURES / "av_sync.csv", index=False)
    paired = df[df.n_lines_paired >= 3]
    summary = dict(trailers=len(df), trailers_with_found_lines=int((df.lines_found > 0).sum()),
                   lines_paired=int(df.n_lines_paired.sum()),
                   pooled_sync_rate=float((df.sync_rate * df.n_lines_paired).sum() / df.n_lines_paired.sum()),
                   median_sync_rate_3plus=float(paired.sync_rate.median()), trailers_3plus=len(paired),
                   median_speech_share=float(df.speech_share.median()), median_words_per_min=float(df.words_per_min.median()),
                   sync_seconds=SYNC, labels=label_check())
    (FEATURES / "av_sync.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.hist(paired.sync_rate, bins=np.linspace(0, 1, 11), color="#6250d6", rwidth=0.9)
    ax.set_xlabel("share of film lines that play over their own scene")
    ax.set_ylabel("trailers")
    ax.set_title(f"Sound vs picture ({len(paired)} trailers with 3+ lines)", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "av_sync.png", dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
