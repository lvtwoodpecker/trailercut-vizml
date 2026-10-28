"""Find replacement trailers for core films whose MovieNet YouTube link is dead.

For each film: search YouTube for "<title> (<year>) trailer", drop obvious non-trailers (clips,
reviews, fan edits, wrong length), then try candidates in search order. A candidate is accepted
only if enough of its shots align to that film's MovieNet keyframes, which catches wrong films
(sequels, remakes, a different film with the same title) that title matching can't.

Searched trailers can be re-release or anniversary cuts, so every row records where it came from.

Outputs
  data/ids/trailer_status.csv    replaced rows get the new youtube_id and source=search
  data/ids/trailer_recovery.csv  every candidate tried, with its match share

  python -m trailercut.recover [--tries 3] [--min-match 0.25]
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import yt_dlp

from . import movienet_io as mn
from .paths import CACHE, CLIP, FEATURES, IDS, TRAILERS
from .trailers import FIELDS, STATUS, fetch

LOG = IDS.parent / "trailer_recovery.csv"
LOG_FIELDS = ["imdb_id", "rank", "youtube_id", "title", "channel", "duration", "match_share", "accepted"]
SKIP = re.compile(r"fan.?made|unofficial|modern trailer|re-?cut|reaction|review|explained|scene|clip|ending|behind the|featurette|"
                  r"interview|honest|parody|recap|breakdown|easter egg|vs\.?|\bcast\b", re.I)


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("&", "and"))


def candidates(imdb_id, n=10):
    m = mn.meta(imdb_id)
    title = re.sub(r"\s*\(\d{4}.*", "", m["title"])
    query = f"{title} ({m['year']}) trailer" if m["year"] else f"{title} trailer"
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
        if SKIP.search(t) or not 45 <= d <= 300 or not all(w in norm(t) for w in key):
            continue
        out.append(dict(youtube_id=e["id"], title=t, channel=e.get("channel") or "", duration=d))
    return out


def clear(imdb_id):
    for f in TRAILERS.glob(f"{imdb_id}.*"):
        f.unlink()
    for f in [FEATURES / "trailer_shots" / f"{imdb_id}.csv", CACHE / "trailer_audio" / f"{imdb_id}.wav",
              CLIP / "trailer" / f"{imdb_id}.npy"]:
        f.unlink(missing_ok=True)
    shutil.rmtree(CACHE / "trailer_frames" / imdb_id, ignore_errors=True)


def run(*mod):
    subprocess.run([sys.executable, "-m", *mod], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--min-match", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    status = {r["imdb_id"]: r for r in csv.DictReader(open(STATUS))}
    todo = [i for i, r in status.items() if r["status"] != "ok"]
    if a.limit:
        todo = todo[: a.limit]
    # the impostor cutoff, not the label-picked one: this check has to reject trailers of other films,
    # and the label-picked cutoff lets most shots from a wrong film match something
    meta = json.loads((FEATURES / "alignment_threshold.json").read_text())
    threshold = meta.get("impostor_threshold", meta["threshold"])

    with ThreadPoolExecutor(4) as ex:
        cands = dict(zip(todo, ex.map(candidates, todo)))
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
            run("trailercut.trailer_prep")
            run("trailercut.embed", "trailer")
        for i, c in batch.items():
            share = None
            if got[i]["status"] == "ok":
                try:
                    df = align_one(i, threshold)
                    share = float(df.matched[~df.blank].mean())
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
        print(f"round {rank + 1}: {sum(r['accepted'] for r in log)} recovered so far")

    with open(STATUS, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, restval="movienet")
        w.writeheader()
        w.writerows(sorted(status.values(), key=lambda r: r["imdb_id"]))
    prev = list(csv.DictReader(open(LOG))) if LOG.exists() else []
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, LOG_FIELDS)
        w.writeheader()
        w.writerows(prev + log)
    print(f"recovered {sum(r['accepted'] for r in log)}/{len(todo)}")


if __name__ == "__main__":
    main()
