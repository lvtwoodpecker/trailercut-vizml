
"""Download MovieNet trailers from YouTube.

Supports both bulk downloads and individual IMDb IDs.

Examples:
  python -m trailercut.trailers
  python -m trailercut.trailers --limit 10
  python -m trailercut.trailers --ids tt0032138
"""
import argparse
import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import shutil

if shutil.which("ffprobe") is None:
    import static_ffmpeg
    static_ffmpeg.add_paths()

import yt_dlp

from .paths import IDS, IDS_DIR, TRAILERS

STATUS = IDS_DIR / "trailer_status.csv"
FIELDS = ["imdb_id", "youtube_id", "status", "duration", "title", "error", "source"]


def valid_video(path):
    """Check that an MP4 exists and contains a readable video stream."""
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return False


def fetch(imdb_id, youtube_id):
    out = TRAILERS / f"{imdb_id}.mp4"

    if valid_video(out):
        return dict(
            imdb_id=imdb_id,
            youtube_id=youtube_id,
            status="ok",
            duration="",
            title="",
            error="",
        )

    opts = {
        "format": (
            "best[height<=360][ext=mp4]/"
            "bv*[height<=360][ext=mp4]+ba[ext=m4a]/"
            "best[height<=360]"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(TRAILERS / f"{imdb_id}.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={youtube_id}",
                download=True,
            )

        if not valid_video(out):
            raise RuntimeError("Download finished but MP4 validation failed")

        return dict(
            imdb_id=imdb_id,
            youtube_id=youtube_id,
            status="ok",
            duration=info.get("duration", ""),
            title=info.get("title", ""),
            error="",
        )

    except Exception as e:
        msg = str(e).splitlines()[0][:200]
        low = msg.lower()

        status = (
            "unavailable"
            if any(k in low for k in (
                "unavailable", "private", "removed", "terminated"
            ))
            else "error"
        )

        return dict(
            imdb_id=imdb_id,
            youtube_id=youtube_id,
            status=status,
            duration="",
            title="",
            error=msg,
        )


def write_status(done):
    STATUS.parent.mkdir(parents=True, exist_ok=True)

    tmp = STATUS.with_suffix(".tmp")

    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, restval="movienet")
        w.writeheader()

        for r in sorted(done.values(), key=lambda r: r["imdb_id"]):
            w.writerow(r)

    os.replace(tmp, STATUS)


def load_rows():
    """Load the existing download mapping, preserving replacement links."""
    if STATUS.exists():
        with open(STATUS, newline="") as f:
            return list(csv.DictReader(f))

    with open(IDS, newline="") as f:
        return [
            r for r in csv.DictReader(f)
            if r.get("youtube_id")
        ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--ids",
        nargs="+",
        help="Download only these IMDb IDs",
    )
    a = ap.parse_args()

    TRAILERS.mkdir(parents=True, exist_ok=True)

    rows = load_rows()

    if a.ids:
        requested = set(a.ids)
        rows = [r for r in rows if r["imdb_id"] in requested]

        found = {r["imdb_id"] for r in rows}
        missing = requested - found

        if missing:
            # Also look in the original film list if status lacks an entry.
            with open(IDS, newline="") as f:
                original = list(csv.DictReader(f))

            by_id = {r["imdb_id"]: r for r in original}
            for imdb_id in sorted(missing):
                if imdb_id in by_id and by_id[imdb_id].get("youtube_id"):
                    rows.append(by_id[imdb_id])

            found = {r["imdb_id"] for r in rows}
            missing = requested - found

        if missing:
            ap.error(f"No YouTube mapping found for: {', '.join(sorted(missing))}")

    if a.limit:
        rows = rows[:a.limit]

    done = {}

    if STATUS.exists():
        with open(STATUS, newline="") as f:
            done = {r["imdb_id"]: r for r in csv.DictReader(f)}

    # Retry missing or invalid local files even if the status CSV says "ok".
    todo = [
        r for r in rows
        if not valid_video(TRAILERS / f'{r["imdb_id"]}.mp4')
    ]

    if not todo:
        for r in rows:
            print(f'{r["imdb_id"]} READY (video already downloaded)')
        return

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {
            ex.submit(fetch, r["imdb_id"], r["youtube_id"]): r
            for r in todo
        }

        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            res["source"] = "movienet"
            done[res["imdb_id"]] = res
            write_status(done)

            print(
                f'[{i}/{len(todo)}] {res["imdb_id"]} '
                f'{res["status"]} {res["error"][:80]}'
            )

    counts = {}
    for r in done.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print(counts)


if __name__ == "__main__":
    main()
