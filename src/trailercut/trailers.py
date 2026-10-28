"""Download MovieNet trailers from YouTube and log which ones are still available.

Videos land in data/trailers/ (gitignored). The committed artifact is
data/ids/trailer_status.csv, so anyone can see the real usable N without the videos.

  python -m trailercut.trailers [--limit N] [--workers 4]
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp

from .paths import IDS, TRAILERS

STATUS = IDS.parent / "trailer_status.csv"
FIELDS = ["imdb_id", "youtube_id", "status", "duration", "title", "error", "source"]


def fetch(imdb_id, youtube_id):
    for f in TRAILERS.glob(f"{imdb_id}.*"):
        if f.suffix == ".mp4":
            return dict(imdb_id=imdb_id, youtube_id=youtube_id, status="ok", duration="", title="", error="")
    opts = {
        # 360p is enough for CLIP (224px) and shot detection (48x27); keeps disk small.
        "format": "best[height<=360][ext=mp4]/bv*[height<=360][ext=mp4]+ba[ext=m4a]/best[height<=360]",
        "merge_output_format": "mp4",
        "outtmpl": str(TRAILERS / f"{imdb_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={youtube_id}", download=True)
        return dict(imdb_id=imdb_id, youtube_id=youtube_id, status="ok",
                    duration=info.get("duration", ""), title=info.get("title", ""), error="")
    except Exception as e:
        msg = str(e).splitlines()[0][:200]
        low = msg.lower()
        status = "unavailable" if any(k in low for k in ("unavailable", "private", "removed", "terminated")) else "error"
        return dict(imdb_id=imdb_id, youtube_id=youtube_id, status=status, duration="", title="", error=msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    TRAILERS.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(IDS)))
    if a.limit:
        rows = rows[: a.limit]

    done = {}
    if STATUS.exists():
        done = {r["imdb_id"]: r for r in csv.DictReader(open(STATUS))}
    todo = [r for r in rows if done.get(r["imdb_id"], {}).get("status") != "ok"]

    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fetch, r["imdb_id"], r["youtube_id"]) for r in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            done[res["imdb_id"]] = res
            print(f"[{i}/{len(todo)}] {res['imdb_id']} {res['status']} {res['error'][:80]}")

    with open(STATUS, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, restval="movienet")
        w.writeheader()
        for r in sorted(done.values(), key=lambda r: r["imdb_id"]):
            w.writerow(r)
    counts = {}
    for r in done.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
