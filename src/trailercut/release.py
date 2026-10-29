"""Bundle the trailer links and our annotations as a dataset release (e.g. for Hugging Face).

Links and annotations only: no videos, frames, audio or transcripts (copyrighted; MovieNet itself is
research-only). Written to data/release/ (gitignored; upload that folder).

  trailers.csv                one row per film: MovieNet's trailer link vs the link we use, why it
                              changed, YouTube title/channel/duration, audit flags, match share, span
  trailer_shots.parquet       TransNetV2 shot boundaries per trailer
  alignment.parquet           trailer shot -> film shot (score, matched, top-5), at the cutoff in use
  dialogue_alignment.parquet  trailer speech segment -> film subtitle time (times and scores, no text)
  av_sync.csv                 per-trailer speech features and sound/picture sync rate
  hand_labels.csv             hand-checked alignments (trailer shot -> film shot, or s / n / u)
  README.md                   dataset card

  python -m trailercut.release
"""
import json
from datetime import date

import pandas as pd

from . import movienet_io as mn
from .paths import FEATURES, IDS, LABELS, ROOT

OUT = ROOT / "data" / "release"


def stack(folder):
    parts = [pd.read_csv(p).assign(imdb_id=p.stem) for p in sorted((FEATURES / folder).glob("*.csv"))]
    df = pd.concat(parts, ignore_index=True)
    return df[["imdb_id", *[c for c in df.columns if c != "imdb_id"]]]


def trailers():
    ids = IDS.parent
    status = pd.read_csv(ids / "trailer_status.csv", dtype=str)
    films = pd.read_csv(ids / "films_1100.csv", dtype=str).rename(columns={"youtube_id": "movienet_youtube_id"})
    audit = pd.read_csv(ids / "trailer_audit.csv", dtype=str)[["imdb_id", "flags", "span", "channel", "height"]]
    review = pd.read_csv(ids / "trailer_review.csv", dtype=str).rename(columns={"reason": "review_note"})
    df = status.merge(films[["imdb_id", "movienet_youtube_id", "core"]], on="imdb_id", how="left") \
               .merge(audit, on="imdb_id", how="left").merge(review[["imdb_id", "review_note"]], on="imdb_id", how="left")
    df["title_film"] = df.imdb_id.map(lambda i: mn.meta(i)["title"])
    df["changed"] = df.youtube_id != df.movienet_youtube_id
    share = {}
    for p in (FEATURES / "alignment").glob("*.csv"):
        a = pd.read_csv(p)
        share[p.stem] = round(float(a.matched[~a.blank].mean()), 3)
    df["match_share"] = df.imdb_id.map(share)
    df["checked"] = date.today().isoformat()
    cols = ["imdb_id", "title_film", "core", "movienet_youtube_id", "youtube_id", "changed", "source", "status",
            "title", "channel", "duration", "height", "flags", "match_share", "span", "review_note", "checked"]
    return df.rename(columns={"title": "youtube_title"})[[c if c != "title" else "youtube_title" for c in cols]]


CARD = """---
pretty_name: Trailer-to-film alignment for MovieNet
license: cc-by-nc-4.0
task_categories: [video-classification]
tags: [movie-trailers, movienet, shot-alignment, video]
---

# Trailer-to-film alignment for MovieNet

Which shots of a film end up in its trailer. For {n} MovieNet films: a working YouTube link to a
trailer (checked {checked}), shot boundaries for each trailer, an alignment of every trailer shot to
the film shot it was cut from (or "no match"), an alignment of trailer dialogue to the film's
subtitles, and {n_labels} hand-checked alignments.

## Files

| file | what |
|---|---|
| `trailers.csv` | one row per film: MovieNet's trailer link, the link used here, why it changed, YouTube metadata, audit flags, match share |
| `trailer_shots.parquet` | trailer shot boundaries (TransNetV2) |
| `alignment.parquet` | trailer shot -> MovieNet film shot index, similarity score, matched flag, top-5 candidates |
| `dialogue_alignment.parquet` | trailer speech segment -> film subtitle time; times and scores only |
| `av_sync.csv` | per-trailer speech share, words per minute, share of lines found in the film, sound/picture sync rate |
| `hand_labels.csv` | hand labels: the source film shot, `s` (same scene, other shot), `n` (not in film) or `u` (unsure) |

Film shot indices refer to MovieNet's shot annotations (`movie1K.shot_detection`).

## How it was made

- Trailer links start from MovieNet's list. Dead links, scene clips, dubs, teasers, TV spots and home-video
  ads were replaced by searching YouTube for the original theatrical trailer; a replacement is kept only if
  at least 25% of its shots match the film and the matches span at least 20% of it. `source` says
  whether a link is MovieNet's or a replacement.
- Shots: TransNetV2. Alignment: CLIP ViT-B/32 on 3 keyframes per shot; a shot counts as matched above
  a similarity cutoff picked from the hand labels (best F1: precision 0.85, recall 0.91).
- Dialogue: Whisper (small) transcripts matched to MovieNet subtitles with fuzzy text matching.

## Limitations

- One trailer per film; for older films the original theatrical trailer is not always on YouTube.
- About 15% of matched shots are wrong (look-alike shots, or footage not in the film).
- Some trailer shots likely span a cut the shot detector missed.
- Hand labels come from one labeler, choosing among the aligner's top 5.

## Terms

Annotations only. Videos stay on YouTube and are not redistributed; MovieNet's keyframes, subtitles
and annotations are under MovieNet's own terms (research use). These annotations: CC BY-NC 4.0.
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = trailers()
    t.to_csv(OUT / "trailers.csv", index=False)
    stack("trailer_shots").to_parquet(OUT / "trailer_shots.parquet", index=False)
    stack("alignment").to_parquet(OUT / "alignment.parquet", index=False)
    stack("dialogue_alignment").to_parquet(OUT / "dialogue_alignment.parquet", index=False)
    pd.read_csv(FEATURES / "av_sync.csv").to_csv(OUT / "av_sync.csv", index=False)
    labels = []
    for n, p in enumerate(sorted(LABELS.glob("gold_*.json")), start=1):  # labelers anonymized
        g = json.loads(p.read_text())
        labels += [dict(imdb_id=k.split(":")[0], trailer_shot=int(k.split(":")[1]), label=v, labeler=f"labeler_{n}")
                   for k, v in g["labels"].items()]
    pd.DataFrame(labels).to_csv(OUT / "hand_labels.csv", index=False)
    (OUT / "README.md").write_text(CARD.format(n=len(t), checked=t.checked.iloc[0], n_labels=len(labels)))
    print(f"{len(t)} films, {int(t.changed.sum())} links changed from MovieNet's -> {OUT}")


if __name__ == "__main__":
    main()
