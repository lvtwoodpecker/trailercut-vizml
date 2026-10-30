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
from .paths import CACHE, FEATURES, IDS, IDS_DIR, LABELS, ROOT

OUT = ROOT / "data" / "release"


def stack(folder):
    parts = [pd.read_csv(p).assign(imdb_id=p.stem) for p in sorted((FEATURES / folder).glob("*.csv"))]
    df = pd.concat(parts, ignore_index=True)
    return df[["imdb_id", *[c for c in df.columns if c != "imdb_id"]]]


def youtube_meta():
    """Title / channel / duration per YouTube id, from the metadata cache and every candidate log."""
    out = {}
    for f in ["trailer_recovery.csv", "swap_candidates.csv", "expand_candidates.csv"]:
        if (IDS_DIR / f).exists():
            for r in pd.read_csv(IDS_DIR / f, dtype=str).itertuples():
                out[r.youtube_id] = dict(title=r.title, channel=r.channel, duration=r.duration)
    for p in (CACHE / "yt_meta").glob("*.json"):
        m = json.loads(p.read_text())
        if "error" not in m:
            out[p.stem] = dict(title=m.get("title"), channel=m.get("channel"), duration=m.get("duration"))
    return out


def why(i, old, now, review, audit, meta):
    if now == old:
        return ""
    if i in review.index and isinstance(review.reason.get(i), str):
        return review.reason.get(i)
    if not old:
        return "no MovieNet link"
    if old not in meta:
        return "MovieNet link dead or unavailable"
    flags = audit["flags"].get(i)
    return f"audit: {flags}" if isinstance(flags, str) else "replaced"


def trailers():
    """One row per film: the film, MovieNet's original trailer (before), the trailer in use (after), and
    how well the trailer in use matches the film."""
    status = pd.read_csv(IDS_DIR / "trailer_status.csv", dtype=str)
    films = pd.read_csv(IDS_DIR / "films_1100.csv", dtype=str).set_index("imdb_id")
    audit = pd.read_csv(IDS_DIR / "trailer_audit.csv", dtype=str).set_index("imdb_id")
    review = pd.read_csv(IDS_DIR / "trailer_review.csv", dtype=str).drop_duplicates("imdb_id", keep="last").set_index("imdb_id")
    meta = youtube_meta()
    url = lambda y: f"https://www.youtube.com/watch?v={y}" if isinstance(y, str) and y else ""
    rows = []
    for r in status[status.status == "ok"].itertuples():
        i, old = r.imdb_id, films.youtube_id.get(r.imdb_id)
        old = old if isinstance(old, str) else ""
        a = pd.read_csv(FEATURES / "alignment" / f"{i}.csv") if (FEATURES / "alignment" / f"{i}.csv").exists() else None
        pos = a.film_story_pos.dropna() if a is not None else pd.Series(dtype=float)
        m = mn.meta(i)
        now = meta.get(r.youtube_id, {})
        rows.append(dict(
            imdb_id=i, film=m["title"], year=m["year"], core=films.core.get(i),
            movienet_url=url(old), movienet_title=meta.get(old, {}).get("title", "") if old else "",
            movienet_flags=audit["flags"].get(i, "") if audit["youtube_id"].get(i) == old else "",
            url=url(r.youtube_id), title=now.get("title") or r.title, channel=now.get("channel", ""),
            duration_s=now.get("duration") or r.duration, source=r.source, changed=r.youtube_id != old,
            why_changed=why(i, old, r.youtube_id, review, audit, meta),
            trailer_shots=len(a) if a is not None else None,
            match_share=round(float(a.matched[~a.blank].mean()), 3) if a is not None else None,
            film_span=round(float(pos.quantile(0.9) - pos.quantile(0.1)), 3) if len(pos) else None,
            checked=date.today().isoformat()))
    return pd.DataFrame(rows)


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
| `trailers.csv` | one row per film: MovieNet's original trailer (link, YouTube title, audit flags), the trailer used here (link, title, channel, length, source), why it changed, and how well it matches the film (shots, match share, span of the film covered) |
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
    from .label_eval import load_gold  # drops labels of trailers that were swapped since
    gold = load_gold()
    names = {n: f"labeler_{k}" for k, n in enumerate(sorted(gold.labeler.unique()), start=1)}  # anonymized
    labels = [dict(imdb_id=k.split(":")[0], trailer_shot=int(k.split(":")[1]), label=v, labeler=names[l])
              for k, v, l in zip(gold.key, gold.label, gold.labeler)]
    pd.DataFrame(labels).to_csv(OUT / "hand_labels.csv", index=False)
    (OUT / "README.md").write_text(CARD.format(n=len(t), checked=t.checked.iloc[0], n_labels=len(labels)))
    print(f"{len(t)} films, {int(t.changed.sum())} links changed from MovieNet's -> {OUT}")


if __name__ == "__main__":
    main()
