"""Find replacement trailers for core films whose MovieNet YouTube link is dead.

For each film: search YouTube for "<title> (<year>) official trailer", drop non-trailers by title
(clips, reviews, fan edits, dubs, subtitled uploads, teasers, TV spots, re-releases; the film's own
title is ignored, so "Freddy vs. Jason" isn't dropped) and wrong lengths, put official or
theatrical titles and 90 to 180 s videos first, then try candidates in that order. A candidate is
accepted only if enough of its shots align to that film's MovieNet keyframes, which catches wrong films
(sequels, remakes, a different film with the same title) that title matching can't, and if the matched
shots span enough of the film, which catches scene clips. --ids replaces working trailers too (old
files kept in cache/replaced/{id}/ and restored if no candidate passes).

Searched trailers can be re-release or anniversary cuts, so every row records where it came from.

Outputs
  data/ids/trailer_status.csv    replaced rows get the new youtube_id and source=search
  data/ids/trailer_recovery.csv  every candidate tried, with its match share

  python -m trailercut.recover [--tries 3] [--min-match 0.25] [--min-span 0.2] [--ids ID ...]
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yt_dlp

from . import movienet_io as mn
from .paths import CACHE, CLIP, FEATURES, IDS, IDS_DIR, TRAILERS
from .trailers import FIELDS, STATUS, fetch

LOG = IDS_DIR / "trailer_recovery.csv"
LOG_FIELDS = ["imdb_id", "rank", "youtube_id", "title", "channel", "duration", "match_share", "accepted"]
SKIP = re.compile(r"fan.?made|unofficial|modern trailer|re-?cut|reaction|review|explained|scene|clip|ending|behind the|featurette|"
                  r"interview|honest|parody|recap|breakdown|easter egg|vs\.?|\bcast\b|"
                  # not the original theatrical trailer in the original language
                  r"teaser|tv spot|blu-?ray|\bdvd\b|\bvhs\b|anniversary|re-?release|restor|remaster|"
                  r"german|deutsch|español|espanol|castellano|latino|\bvf\b|vostfr|français|bande.annonce|italiano|"
                  r"dublado|legendado|subtitulado|untertitel|sottotitol|\bsub(s|titled)?\b|sub\.|자막|字幕|magyar|türkçe|русск|ελληνικ", re.I)
OFFICIAL = re.compile(r"official|theatrical", re.I)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("&", "and"))


def candidates(imdb_id, n=10):
    m = mn.meta(imdb_id)
    title = re.sub(r"\s*\(\d{4}.*", "", m["title"])
    query = f"{title} ({m['year']}) official trailer" if m["year"] else f"{title} official trailer"
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as ydl:
            res = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    except Exception as e:
        print(imdb_id, "search failed:", str(e)[:80])
        return []
    # the film's main title words have to appear in the video title (handles "Part II" vs "Part 2" loosely)
    key = [w for w in norm(title.split(":")[0]).split() if w not in {"the", "a", "an", "of"}] or norm(title).split()
    out = []
    for e in res.get("entries") or []:
        t, d = e.get("title") or "", e.get("duration") or 0
        # the film's own title doesn't count against it ("Freddy vs. Jason", "Scene of the Crime")
        if SKIP.search(re.sub(re.escape(title), "", t, flags=re.I)) or not 45 <= d <= 300 or not all(w in norm(t) for w in key):
            continue
        out.append(dict(youtube_id=e["id"], title=t, channel=e.get("channel") or "", duration=d))
    # official / theatrical titles first, then trailers of typical length; otherwise keep search order
    return sorted(out, key=lambda c: (not OFFICIAL.search(c["title"]), not 90 <= c["duration"] <= 180))


def trailer_files(imdb_id):
    return [*TRAILERS.glob(f"{imdb_id}.*"), FEATURES / "trailer_shots" / f"{imdb_id}.csv",
            CACHE / "trailer_audio" / f"{imdb_id}.wav", CLIP / "trailer" / f"{imdb_id}.npy",
            CACHE / "trailer_asr" / f"{imdb_id}.json", CACHE / "trailer_frames" / imdb_id]


def clear(imdb_id):
    for f in trailer_files(imdb_id):
        shutil.rmtree(f) if f.is_dir() else f.unlink(missing_ok=True)


def backup(imdb_id, restore=False):
    """Move a working trailer's files aside before replacing it (--ids), or put them back."""
    keep = CACHE / "replaced" / imdb_id
    if restore:
        for item in json.loads((keep / "_paths.json").read_text()):
            if (keep / Path(item).name).exists():
                shutil.move(str(keep / Path(item).name), item)
        shutil.rmtree(keep)
        return
    if (keep / "_paths.json").exists():
        listed = json.loads((keep / "_paths.json").read_text())
        if not listed or any((keep / Path(item).name).exists() for item in listed):
            return  # an interrupted run already holds the original here; don't overwrite it with a candidate
        # listed files all gone (left over from a finished restore): stale, back up what's in place now
    keep.mkdir(parents=True, exist_ok=True)
    moved = [str(f) for f in trailer_files(imdb_id) if f.exists()]
    for f in moved:
        shutil.move(f, keep / Path(f).name)
    (keep / "_paths.json").write_text(json.dumps(moved))


def run(*mod):
    subprocess.run([sys.executable, "-m", *mod], check=True)


def save(status, rows):
    """Status log rewritten, recovery log appended; called after every film."""
    with open(STATUS, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, restval="movienet")
        w.writeheader()
        w.writerows(sorted(status.values(), key=lambda r: r["imdb_id"]))
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--min-match", type=float, default=0.25)
    ap.add_argument("--min-span", type=float, default=0.2, help="matched shots must span this much of the film")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=[], help="replace these films' working trailers too")
    a = ap.parse_args()

    status = {r["imdb_id"]: r for r in csv.DictReader(open(STATUS))}
    todo = a.ids or [i for i, r in status.items() if r["status"] != "ok"]
    if a.limit:
        todo = todo[: a.limit]
    # the impostor cutoff, not the label-picked one: this check has to reject trailers of other films,
    # and the label-picked cutoff lets most shots from a wrong film match something
    meta = json.loads((FEATURES / "alignment_threshold.json").read_text())
    threshold = meta.get("impostor_threshold", meta["threshold"])

    with ThreadPoolExecutor(4) as ex:
        cands = dict(zip(todo, ex.map(candidates, todo)))
    for i in a.ids:  # never re-pick the link being replaced; keep its files until a candidate passes
        cands[i] = [c for c in cands[i] if c["youtube_id"] != status[i]["youtube_id"]]
        backup(i)
    print(f"{len(todo)} films, {sum(bool(c) for c in cands.values())} with search candidates")

    from .align import BLANK, align_one
    log = []
    for rank in range(a.tries):
        batch = {i: c[rank] for i, c in cands.items() if len(c) > rank}
        if not batch:
            break
        with ThreadPoolExecutor(3) as ex:
            got = dict(zip(batch, ex.map(lambda i: fetch(i, batch[i]["youtube_id"]), batch)))
        ok = [i for i, r in got.items() if r["status"] == "ok"]
        print(f"round {rank + 1}: {len(ok)}/{len(batch)} downloaded")
        if ok:
            run("trailercut.trailer_prep", "--ids", *ok)  # only this round's films
            run("trailercut.embed", "trailer", "--ids", *ok)
        for i, c in batch.items():
            share = None
            if got[i]["status"] == "ok":
                try:
                    df = align_one(i, threshold)
                    share = float(df.matched[~df.blank].mean())
                    # a scene clip of the right film aligns perfectly but covers a sliver of it
                    pos = df.film_story_pos.dropna()
                    if len(pos) and pos.quantile(0.9) - pos.quantile(0.1) < a.min_span:
                        share = 0.0
                except Exception as e:
                    print(i, "align failed:", e)
            accepted = share is not None and share >= a.min_match
            log.append(dict(imdb_id=i, rank=rank, **c, match_share=share, accepted=accepted))
            if accepted:
                status[i] = dict(imdb_id=i, youtube_id=c["youtube_id"], status="ok", duration=c["duration"],
                                 title=c["title"], error="", source="search")
                del cands[i]
            else:
                clear(i)
            save(status, log[-1:])  # after every film, so an interrupted run keeps its decisions
        print(f"round {rank + 1}: {sum(r['accepted'] for r in log)} recovered so far")

    for i in a.ids:
        if i in cands:  # no candidate passed: put the old trailer back
            clear(i)
            backup(i, restore=True)
    save(status, [])
    print(f"recovered {sum(r['accepted'] for r in log)}/{len(todo)}")


if __name__ == "__main__":
    main()
