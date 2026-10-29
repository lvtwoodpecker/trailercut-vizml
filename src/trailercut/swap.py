"""Replace trailers in three resumable steps, so YouTube's rate limits only slow the middle one.

  python -m trailercut.swap prepare --set expand   new films (data/ids/films_1100.csv, not yet in the
                                                   status log): keep MovieNet's link as first choice if
                                                   its metadata looks like a clean trailer, otherwise
                                                   search; nothing is downloaded

  python -m trailercut.swap search [--ids ...]     ranked candidates -> data/ids/swap_candidates.csv
                                                   (default ids: `swap` rows of data/ids/trailer_audit.csv)
  python -m trailercut.swap download [--sleep 30] [--per-film 1] [--cookies-from-browser chrome]
                                                   one video at a time into data/trailers/candidates/,
                                                   walking down each film's candidates past unavailable
                                                   ones; resumable; stops after repeated bot checks
  python -m trailercut.swap sheet                  contact sheets of the swaps (cache/sheets/, not committed)
  python -m trailercut.swap apply                  for each film, try its downloaded candidates in rank
                                                   order with the usual checks (25% of shots match,
                                                   matched shots span 20% of the film); keep the old
                                                   trailer if none passes

Candidates come from recover.candidates (official titles first; dubs, teasers, TV spots and
re-releases rejected by title). Download log: data/ids/swap_downloads.csv. With --set expand, every
step uses data/ids/expand_candidates.csv and expand_downloads.csv instead. Videos found unavailable
are marked in the candidate list's `unavailable` column and skipped; download --recheck retries them.
"""
import argparse
import csv
import json
import shutil
import time

import pandas as pd
import yt_dlp

from .paths import FEATURES, IDS, TRAILERS
from .recover import LOG, LOG_FIELDS, SKIP, backup, candidates, clear, run
from .trailer_audit import yt_meta
from .trailers import FIELDS, STATUS

# --set swap: replacing flagged core trailers; --set expand: trailers for films new to the dataset
CANDS = IDS.parent / "swap_candidates.csv"
DOWNLOADS = IDS.parent / "swap_downloads.csv"
DIR = TRAILERS / "candidates"
FORMAT = "best[height<=360][ext=mp4]/bv*[height<=360][ext=mp4]+ba[ext=m4a]/best[height<=360]"


GONE = ("not available", "unavailable", "private", "removed", "terminated", "copyright")


def mark_unavailable(cands):
    """Carry unavailable videos from the download log into the candidate list (with the reason),
    so later runs skip them without asking YouTube again."""
    if not DOWNLOADS.exists():
        return cands
    log = pd.read_csv(DOWNLOADS)
    gone = log[(log.status == "error") & log.error.str.lower().str.contains("|".join(GONE))]
    reason = dict(zip(gone.youtube_id, gone.error.str.replace(r"^ERROR: \[youtube\] \S+: ", "", regex=True)))
    cands["unavailable"] = cands.youtube_id.map(reason).fillna(cands.get("unavailable", pd.Series(index=cands.index, dtype=str)))
    cands.to_csv(CANDS, index=False)
    return cands


def prepare(sleep):
    """Pick first-choice candidates for films without a trailer yet, from metadata only."""
    films = pd.read_csv(IDS.parent / "films_1100.csv", dtype=str).fillna("")
    have = {r["imdb_id"] for r in csv.DictReader(open(STATUS))}
    rows = []
    for f in films[~films.imdb_id.isin(have)].itertuples():
        clean = False
        if f.youtube_id:
            m = yt_meta(f.youtube_id)
            time.sleep(1)  # metadata requests count toward YouTube's rate limit too
            t, d = m.get("title") or "", m.get("duration") or 0
            clean = "error" not in m and not SKIP.search(t) and 60 <= d <= 240
            if clean:
                rows.append(dict(imdb_id=f.imdb_id, rank=0, youtube_id=f.youtube_id, title=t,
                                 channel=m.get("channel") or "", duration=d, source="movienet"))
        if not clean:  # dead link, flagged title, odd length, or no link: search
            for rank, c in enumerate(candidates(f.imdb_id), start=1):
                rows.append(dict(imdb_id=f.imdb_id, rank=rank, **c, source="search"))
            time.sleep(sleep)
    mark_unavailable(pd.DataFrame(rows))
    d = pd.DataFrame(rows)
    print(f"{d.imdb_id.nunique()} films: {(d['rank'] == 0).sum()} keep MovieNet's link, "
          f"{d[d['rank'] > 0].imdb_id.nunique()} need a search pick -> {CANDS}")


def search(ids):
    status = {r["imdb_id"]: r for r in csv.DictReader(open(STATUS))}
    rows = []
    for i in ids:
        for rank, c in enumerate(c for c in candidates(i) if c["youtube_id"] != status[i]["youtube_id"]):
            rows.append(dict(imdb_id=i, rank=rank, **c))
        time.sleep(2)
    mark_unavailable(pd.DataFrame(rows))
    print(f"{len(rows)} candidates for {len({r['imdb_id'] for r in rows})} of {len(ids)} films -> {CANDS}")


def download(sleep, per_film, cookies, recheck=False):
    """Per film, walk down its ranked candidates until `per_film` of them downloaded. Unavailable
    videos (removed, region-locked) are skipped for good; bot-check errors are retried on the next run."""
    cands = mark_unavailable(pd.read_csv(CANDS))
    if not recheck:  # skip videos already known to be unavailable (removed, private, region-locked)
        cands = cands[cands.unavailable.isna()]
    cands = cands.sort_values(["imdb_id", "rank"])
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
            if prev and prev["status"] == "ok":
                continue
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
            if log[c.youtube_id]["status"] == "error" and any(k in log[c.youtube_id]["error"].lower() for k in GONE):
                mark_unavailable(pd.read_csv(CANDS))
            if blocked >= 3:
                print("YouTube is blocking (3 bot checks in a row); stopping. Rerun later, it resumes.")
                return
            time.sleep(sleep)


def save(status, rows):
    """Write the status log and append to the recovery log; called after every film so an interrupted
    run loses at most the film in progress."""
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


def apply():
    from .align import align_one
    cands = pd.read_csv(CANDS)
    status = {r["imdb_id"]: r for r in csv.DictReader(open(STATUS))}
    meta = json.loads((FEATURES / "alignment_threshold.json").read_text())
    threshold = meta.get("impostor_threshold", meta["threshold"])
    files = {p.stem.split("__")[1]: p for p in DIR.glob("*__*.mp4")}
    tried = pd.read_csv(LOG, dtype=str) if LOG.exists() else pd.DataFrame(columns=LOG_FIELDS)
    n = 0
    for i, g in cands.groupby("imdb_id"):
        got = [c for c in g.sort_values("rank").itertuples() if c.youtube_id in files]
        seen = set(tried[tried.imdb_id == i].youtube_id)
        swapped = i in status and status[i]["youtube_id"] in set(g.youtube_id)
        if not got or swapped or all(c.youtube_id in seen for c in got):
            continue  # nothing downloaded, or already decided in an earlier run
        backup(i)
        accepted, rows = False, []
        for c in got:
            shutil.copyfile(files[c.youtube_id], TRAILERS / f"{i}.mp4")
            run("trailercut.trailer_prep")
            run("trailercut.embed", "trailer")
            df = align_one(i, threshold)
            pos = df.film_story_pos.dropna()
            share = float(df.matched[~df.blank].mean())
            span = float(pos.quantile(0.9) - pos.quantile(0.1)) if len(pos) else 0.0
            accepted = share >= 0.25 and span >= 0.2
            rows.append(dict(imdb_id=i, rank=c.rank, youtube_id=c.youtube_id, title=c.title, channel=c.channel,
                             duration=c.duration, match_share=share, accepted=accepted))
            if accepted:
                status[i] = dict(imdb_id=i, youtube_id=c.youtube_id, status="ok", duration=c.duration,
                                 title=c.title, error="", source=getattr(c, "source", "search"))
                break
            clear(i)
        if not accepted:
            clear(i)
            backup(i, restore=True)
        save(status, rows)
        n += accepted
        print(i, "swapped" if accepted else "kept old", flush=True)
    print(f"swapped {n} films")


def sheet(per_page=8):
    """Contact sheets of the swapped films for a quick look: old vs new YouTube title, and the new
    trailer's best-matching shots next to the film keyframes they matched. Written to cache/sheets/
    (film frames, so never committed)."""
    from PIL import Image, ImageDraw
    from . import movienet_io as mn
    from .paths import CACHE
    log = pd.read_csv(LOG, dtype=str)
    log = log[log.accepted == "True"]
    cands = pd.read_csv(CANDS, dtype=str)
    films = [i for i in cands.imdb_id.unique() if i in set(log.imdb_id)]
    old = {r["imdb_id"]: r for r in csv.DictReader(open(IDS.parent / "trailer_audit.csv"))}
    out = CACHE / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    W, H = 200, 112
    for p in range(0, len(films), per_page):
        batch = films[p:p + per_page]
        img = Image.new("RGB", (W * 8, (H + 30) * len(batch)), (20, 20, 20))
        dr = ImageDraw.Draw(img)
        for r, i in enumerate(batch):
            y = r * (H + 30)
            new = log[log.imdb_id == i].iloc[-1]
            dr.text((2, y + 1), f"{mn.meta(i)['title'][:40]}  old: {old.get(i, {}).get('yt_title', '')[:45]}", fill=(200, 200, 200))
            dr.text((2, y + 14), f"new: {new.title[:70]}  ({new.channel})", fill=(120, 200, 255))
            al = pd.read_csv(FEATURES / "alignment" / f"{i}.csv") if (FEATURES / "alignment" / f"{i}.csv").exists() else None
            if al is None:
                continue
            al = al[~al.blank].sort_values("score", ascending=False)
            pick = []
            for x in al.itertuples():
                if all(abs(x.shot_idx - q.shot_idx) > 2 for q in pick):
                    pick.append(x)
                if len(pick) == 4:
                    break
            for k, x in enumerate(pick):
                t = Image.open(CACHE / "trailer_frames" / i / mn.keyframe_name(x.shot_idx, 1)).convert("RGB").resize((W, H))
                f = mn.open_image(mn.keyframe_paths(i, int(x.film_shot))[1]).convert("RGB").resize((W, H))
                img.paste(t, (k * 2 * W, y + 28))
                img.paste(f, (k * 2 * W + W, y + 28))
        img.save(out / f"swaps_{p // per_page + 1:02d}.jpg", quality=80)
    print(f"{len(films)} swapped films -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "search", "download", "apply", "sheet"])
    ap.add_argument("--set", default="swap", choices=["swap", "expand"], help="which candidate list to work on")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--sleep", type=float, default=30)
    ap.add_argument("--per-film", type=int, default=1, help="successful downloads wanted per film")
    ap.add_argument("--cookies-from-browser", default=None)
    ap.add_argument("--recheck", action="store_true", help="also retry candidates marked unavailable (e.g. from another region)")
    a = ap.parse_args()
    global CANDS, DOWNLOADS
    if a.set == "expand":
        CANDS, DOWNLOADS = IDS.parent / "expand_candidates.csv", IDS.parent / "expand_downloads.csv"
    if a.step == "prepare":
        prepare(sleep=2)
    elif a.step == "search":
        audit = pd.read_csv(IDS.parent / "trailer_audit.csv")
        status = pd.read_csv(STATUS).set_index("imdb_id")
        ids = a.ids or [i for i in audit[audit["swap"]].imdb_id if status.loc[i, "source"] != "search"]
        search(ids)
    elif a.step == "download":
        download(a.sleep, a.per_film, a.cookies_from_browser, a.recheck)
    elif a.step == "sheet":
        sheet()
    else:
        apply()


if __name__ == "__main__":
    main()
