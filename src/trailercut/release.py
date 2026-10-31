"""Bundle the trailer links and our annotations as a dataset release (e.g. for Hugging Face).

Links and annotations only: no videos, frames, audio or transcripts (copyrighted; MovieNet itself is
research-only). Written to data/release/ (gitignored; upload that folder).

  trailers.csv                one row per film: the trailer link in use, YouTube title/channel/duration,
                              trailer shots, match share, film span (swap history stays in data/ids/)
  trailer_shots.parquet       TransNetV2 shot boundaries per trailer
  alignment.parquet           trailer shot -> film shot (score, matched, top-5), at the cutoff in use
  dialogue_alignment.parquet  trailer speech segment -> film subtitle time (times and scores, no text)
  av_sync.csv                 per-trailer speech features and sound/picture sync rate
  hand_labels.csv             hand-checked alignments (trailer shot -> film shot, or s / n / u)
  README.md                   dataset card (numbers filled in from the pipeline's outputs)

  python -m trailercut.release                      build data/release/ and check it
  python -m trailercut.release --push USER/NAME     also upload it to a Hugging Face dataset repo (private
                                                    unless --public; uses the cached `huggingface-cli login`)
"""
import argparse
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


def trailers():
    """One row per film: the film, the trailer link in use, its YouTube metadata, and how well it matches
    the film. (Which links replaced MovieNet's, and why, stays in data/ids/; not part of the release.)"""
    status = pd.read_csv(IDS_DIR / "trailer_status.csv", dtype=str)
    films = pd.read_csv(IDS_DIR / "films_1100.csv", dtype=str).set_index("imdb_id")
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
            imdb_id=i, film=m["title"], year=m["year"],
            url=url(r.youtube_id), title=now.get("title") or r.title, channel=now.get("channel", ""),
            duration_s=now.get("duration") or r.duration,
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
size_categories: [10K<n<100K]
configs:
  - config_name: trailers
    data_files: trailers.csv
  - config_name: alignment
    data_files: alignment.parquet
  - config_name: trailer_shots
    data_files: trailer_shots.parquet
  - config_name: dialogue_alignment
    data_files: dialogue_alignment.parquet
  - config_name: av_sync
    data_files: av_sync.csv
  - config_name: hand_labels
    data_files: hand_labels.csv
---

# Trailer-to-film alignment for MovieNet

Which shots of a film end up in its trailer. For {n} MovieNet films: a working YouTube link to a
trailer (checked {checked}), shot boundaries for each trailer ({n_shots:,} trailer shots), an alignment
of every trailer shot to the film shot it was cut from (or "no match"), an alignment of trailer dialogue
to the film's subtitles, and {n_labels} hand-checked alignments.

## Files

| file | what |
|---|---|
| `trailers.csv` | one row per film: the trailer link, its YouTube title, channel and length, and how well it matches the film (trailer shots, share of shots matched, span of the film covered) |
| `trailer_shots.parquet` | trailer shot boundaries (frames and seconds) |
| `alignment.parquet` | trailer shot -> MovieNet film shot index, similarity score, matched flag, top-5 candidates |
| `dialogue_alignment.parquet` | trailer speech segment -> film subtitle time; times and scores only |
| `av_sync.csv` | per-trailer speech share, words per minute, share of lines found in the film, sound/picture sync rate |
| `hand_labels.csv` | hand labels: the source film shot, `s` (same scene, other shot), `n` (not in film) or `u` (unsure) |

Film shot indices refer to MovieNet's shot annotations (`movie1K.shot_detection`). Join on `imdb_id`
(plus `shot_idx` for trailer shots).

## How it was made

- Trailer links start from MovieNet's list. Dead links, scene clips, dubs, subtitled uploads, teasers,
  TV spots and home-video ads were replaced by searching YouTube for the original theatrical trailer. A
  replacement is kept only if its matched shots span at least 20% of the film and either 25% of its shots
  clear a strict similarity cutoff or most of its shots match this film better than other films.
- Shots: TransNetV2, then two passes over shots whose keyframes disagree more than real shots ever do
  (a missed cut): a split at TransNetV2's most likely frame, then a split where CLIP sees the biggest
  change between neighboring frames. {below_floor:.1%} of trailer shots still look like two shots glued
  together (about the rate of MovieNet's own shots).
- Alignment: CLIP ViT-B/32 on 3 keyframes per shot; a shot counts as matched above a similarity cutoff
  picked from the hand labels (best F1: precision {prec:.2f}, recall {rec:.2f}, cutoff {thr}).
- Dialogue: Whisper (small) transcripts matched to MovieNet subtitles with fuzzy text matching.

## Limitations

- One trailer per film; for older films the original theatrical trailer is not always on YouTube.
- About {wrong:.0%} of matched shots are wrong (look-alike shots, or footage not in the film).
- The CLIP split pass also cuts some real shots in two (about 1 in 5 of its cuts, from a by-eye check).
- Hand labels come from one labeler, choosing among the aligner's top 5, on {n_label_trailers} trailers.
- Links can die; the table records the link that worked when it was checked.

## Terms

Annotations only. Videos stay on YouTube and are not redistributed; MovieNet's keyframes, subtitles
and annotations are under MovieNet's own terms (research use). These annotations: CC BY-NC 4.0.
"""


TEXT_COLS = {"text", "words", "transcript", "subtitle", "line"}


def check(t):
    """The release holds links and annotations only, and its tables agree with each other."""
    tables = {p.name: (pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)) for p in OUT.glob("*.*")
              if p.suffix in (".csv", ".parquet")}
    for name, df in tables.items():
        bad = TEXT_COLS & set(df.columns)
        assert not bad, f"{name} has text columns {bad}"
        assert not any(df[c].astype(str).str.contains(r"\.(?:jpg|png|mp4|wav)$|/Users/|cache/").any()
                       for c in df.columns if df[c].dtype == object), f"{name} has file paths"
    ids = set(t.imdb_id)
    for name in ["alignment.parquet", "trailer_shots.parquet"]:
        assert set(tables[name].imdb_id) == ids, f"{name} films differ from trailers.csv"
    assert t.imdb_id.is_unique and t.url.str.startswith("https://www.youtube.com/watch?v=").all()
    a, sh = tables["alignment.parquet"], tables["trailer_shots.parquet"]
    assert len(a) == len(sh), "alignment and trailer shots disagree on the number of shots"
    print(f"check ok: {len(tables)} tables, {len(ids)} films, {len(a):,} trailer shots, no text or file paths")


def push(repo, public):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=not public, exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo, repo_type="dataset", commit_message="Release")
    print(f"uploaded -> https://huggingface.co/datasets/{repo} ({'public' if public else 'private'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="USER/NAME", help="upload to this Hugging Face dataset repo")
    ap.add_argument("--public", action="store_true", help="make the repo public (default private)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t = trailers()
    t.to_csv(OUT / "trailers.csv", index=False)
    shots = stack("trailer_shots")
    shots.to_parquet(OUT / "trailer_shots.parquet", index=False)
    stack("alignment").to_parquet(OUT / "alignment.parquet", index=False)
    stack("dialogue_alignment").to_parquet(OUT / "dialogue_alignment.parquet", index=False)
    pd.read_csv(FEATURES / "av_sync.csv").to_csv(OUT / "av_sync.csv", index=False)
    from .label_eval import load_gold  # drops labels of trailers that were swapped since
    gold = load_gold()
    names = {n: f"labeler_{k}" for k, n in enumerate(sorted(gold.labeler.unique()), start=1)}  # anonymized
    labels = [dict(imdb_id=k.split(":")[0], trailer_shot=int(k.split(":")[1]), label=v, labeler=names[l])
              for k, v, l in zip(gold.key, gold.label, gold.labeler)]
    labels = pd.DataFrame(labels)
    labels.to_csv(OUT / "hand_labels.csv", index=False)
    ev = json.loads((FEATURES / "alignment_eval.json").read_text())["best_f1"]
    diag = json.loads((FEATURES / "alignment_diagnostics.json").read_text())["cuts"]
    (OUT / "README.md").write_text(CARD.format(
        n=len(t), checked=t.checked.iloc[0], n_shots=len(shots), n_labels=len(labels),
        n_label_trailers=labels.imdb_id.nunique(), below_floor=diag["trailer_below_floor"],
        prec=ev["precision"], rec=ev["recall"], thr=ev["threshold"], wrong=1 - ev["precision"]))
    print(f"{len(t)} films -> {OUT}")
    check(t)
    if a.push:
        push(a.push, a.public)


if __name__ == "__main__":
    main()
