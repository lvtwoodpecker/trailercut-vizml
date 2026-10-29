"""Replace trailers in three resumable steps, so YouTube's rate limits only slow the middle one.

  python -m trailercut.swap search [--ids ...]     ranked candidates -> data/ids/swap_candidates.csv
                                                   (default ids: `swap` rows of data/ids/trailer_audit.csv)
  python -m trailercut.swap download [--sleep 30] [--per-film 1] [--cookies-from-browser chrome]
                                                   one video at a time into data/trailers/candidates/,
                                                   walking down each film's candidates past unavailable
                                                   ones; resumable; stops after repeated bot checks
  python -m trailercut.swap apply                  for each film, try its downloaded candidates in rank
                                                   order with the usual checks (25% of shots match,
                                                   matched shots span 20% of the film); keep the old
                                                   trailer if none passes

Candidates come from recover.candidates (official titles first; dubs, teasers, TV spots and
re-releases rejected by title). Download log: data/ids/swap_downloads.csv.
"""
import argparse
import csv
import json
import shutil
import time

import pandas as pd
import yt_dlp

from .paths import FEATURES, IDS, TRAILERS
from .recover import LOG, LOG_FIELDS, backup, candidates, clear, run
from .trailers import FIELDS, STATUS

CANDS = IDS.parent / "swap_candidates.csv"
DOWNLOADS = IDS.parent / "swap_downloads.csv"
DIR = TRAILERS / "candidates"
FORMAT = "best[height<=360][ext=mp4]/bv*[height<=360][ext=mp4]+ba[ext=m4a]/best[height<=360]"


def search(ids):
    status = {r["imdb_id"]: r for r in csv.DictReader(open(STATUS))}
    rows = []
    for i in ids:
        for rank, c in enumerate(c for c in candidates(i) if c["youtube_id"] != status[i]["youtube_id"]):
            rows.append(dict(imdb_id=i, rank=rank, **c))
        time.sleep(2)
    pd.DataFrame(rows).to_csv(CANDS, index=False)
    print(f"{len(rows)} candidates for {len({r['imdb_id'] for r in rows})} of {len(ids)} films -> {CANDS}")


def download(sleep, per_film, cookies):
    """Per film, walk down its ranked candidates until `per_film` of them downloaded. Unavailable
    videos (removed, region-locked) are skipped for good; bot-check errors are retried on the next run."""
    cands = pd.read_csv(CANDS).sort_values(["imdb_id", "rank"])
    done = pd.read_csv(DOWNLOADS) if DOWNLOADS.exists() else pd.DataFrame(columns=["imdb_id", "youtube_id", "status", "error"])
    log = {r["youtube_id"]: r for r in done.to_dict("records")}
    DIR.mkdir(parents=True, exist_ok=True)
    opts = dict(format=FORMAT, merge_output_format="mp4", quiet=True, no_warnings=True, noprogress=True, retries=2)
    if cookies:
        opts["cookiesfrombrowser"] = (cookies,)
    blocked = 0
    for i, g in cands.groupby("imdb_id"):
        got = sum(log.get(y, {}).get("status") == "ok" for y in g.youtube_id)
        for c in g.itertuples():
            if got >= per_film:
                break
            prev = log.get(c.youtube_id)
            if prev and (prev["status"] == "ok" or "not a bot" not in str(prev["error"])):
                continue  # downloaded already, or permanently unavailable
            out = DIR / f"{c.imdb_id}__{c.youtube_id}.mp4"
            try:
                with yt_dlp.YoutubeDL(dict(opts, outtmpl=str(out.with_suffix(".%(ext)s")))) as y:
                    y.download([f"https://www.youtube.com/watch?v={c.youtube_id}"])
                log[c.youtube_id] = dict(imdb_id=i, youtube_id=c.youtube_id, status="ok", error="")
                got, blocked = got + 1, 0
                print("ok", i, c.youtube_id, flush=True)
            except Exception as e:
                msg = str(e).splitlines()[0][:150]
                log[c.youtube_id] = dict(imdb_id=i, youtube_id=c.youtube_id, status="error", error=msg)
                blocked = blocked + 1 if "not a bot" in msg else 0
                print("error", i, msg[:80], flush=True)
            pd.DataFrame(log.values()).to_csv(DOWNLOADS, index=False)
            if blocked >= 3:
                print("YouTube is blocking (3 bot checks in a row); stopping. Rerun later, it resumes.")
                return
            time.sleep(sleep)


def apply():
    from .align import align_one
    cands = pd.read_csv(CANDS)
    status = {r["imdb_id"]: r for r in csv.DictReader(open(STATUS))}
    meta = json.loads((FEATURES / "alignment_threshold.json").read_text())
    threshold = meta.get("impostor_threshold", meta["threshold"])
    files = {p.stem.split("__")[1]: p for p in DIR.glob("*__*.mp4")}
    log = []
    for i, g in cands.groupby("imdb_id"):
        got = [c for c in g.sort_values("rank").itertuples() if c.youtube_id in files]
        if not got:
            continue
        backup(i)
        accepted = False
        for c in got:
            shutil.copyfile(files[c.youtube_id], TRAILERS / f"{i}.mp4")
            run("trailercut.trailer_prep")
            run("trailercut.embed", "trailer")
            df = align_one(i, threshold)
            pos = df.film_story_pos.dropna()
            share = float(df.matched[~df.blank].mean())
            span = float(pos.quantile(0.9) - pos.quantile(0.1)) if len(pos) else 0.0
            accepted = share >= 0.25 and span >= 0.2
            log.append(dict(imdb_id=i, rank=c.rank, youtube_id=c.youtube_id, title=c.title, channel=c.channel,
                            duration=c.duration, match_share=share, accepted=accepted))
            if accepted:
                status[i] = dict(imdb_id=i, youtube_id=c.youtube_id, status="ok", duration=c.duration,
                                 title=c.title, error="", source="search")
                break
            clear(i)
        if not accepted:
            clear(i)
            backup(i, restore=True)
        print(i, "swapped" if accepted else "kept old")
    with open(STATUS, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, restval="movienet")
        w.writeheader()
        w.writerows(sorted(status.values(), key=lambda r: r["imdb_id"]))
    prev = list(csv.DictReader(open(LOG))) if LOG.exists() else []
    with open(LOG, "w", newline="") as f:
        w = csv.DictWriter(f, LOG_FIELDS)
        w.writeheader()
        w.writerows(prev + log)
    print(f"swapped {sum(r['accepted'] for r in log)} films")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["search", "download", "apply"])
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--sleep", type=float, default=30)
    ap.add_argument("--per-film", type=int, default=1, help="successful downloads wanted per film")
    ap.add_argument("--cookies-from-browser", default=None)
    a = ap.parse_args()
    if a.step == "search":
        audit = pd.read_csv(IDS.parent / "trailer_audit.csv")
        status = pd.read_csv(STATUS).set_index("imdb_id")
        ids = a.ids or [i for i in audit[audit["swap"]].imdb_id if status.loc[i, "source"] != "search"]
        search(ids)
    elif a.step == "download":
        download(a.sleep, a.per_film, a.cookies_from_browser)
    else:
        apply()


if __name__ == "__main__":
    main()
