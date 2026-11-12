"""Build the public website: the three pages, their data, and thumbnail sheets. Trailers play from YouTube.

The local pages play our downloaded trailer files and read loose keyframes; neither can be published.
The public build swaps both: the Trailer View plays each trailer through YouTube's player (same video
as our download, so our shot times line up), and the frames come from one WebP thumbnail sheet per
trailer (every trailer shot) and per film (every film shot), about 2,200 files in all.

  site/index.html, rq1.html, rq2/...   the pages (index.html gets the public config)
  site/data/{id}.json                  per-trailer data + `yt` (YouTube id) + `sheets` (where each frame is)
  site/data/index.json, rq1.json       as in viz/data/
  site/rq2/data/features/clusters.csv  RQ2's table
  site/thumbs/{id}_t.webp, {id}_f.webp thumbnail sheets (or upload them elsewhere and pass --frames-base)

Needs viz/data/ (viewer_export, rq1_export), trailer frames and film keyframes. Sheets already built are
kept, so a rerun only does what's missing.

  python -m trailercut.site_build [--out site] [--frames-base thumbs/] [--limit N] [--workers 8]
"""
import argparse
import json
import math
import shutil
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from . import movienet_io as mn
from .paths import CACHE, FEATURES, IDS_DIR, ROOT

VIZ = ROOT / "viz"
H = 108          # thumbnail height; width keeps the source shape (trailers 16:9, films mostly ~2.4:1)
MAX_SIDE = 16000  # WebP caps a side at 16,383 px


def sheet(paths, out, w):
    """Frames in a grid, row by row; a missing frame stays grey. Returns the grid shape."""
    cols = max(1, min(len(paths), MAX_SIDE // w))
    rows = math.ceil(len(paths) / cols)
    im = Image.new("RGB", (cols * w, rows * H), (225, 224, 217))
    for k, p in enumerate(paths):
        try:
            im.paste(mn.open_image(p).convert("RGB").resize((w, H), Image.LANCZOS), ((k % cols) * w, (k // cols) * H))
        except (FileNotFoundError, KeyError, OSError):
            pass
    im.save(out, "WEBP", quality=70, method=4)
    return cols, rows


def film_width(imdb_id):
    p = mn.keyframe_paths(imdb_id, 0)[1]
    try:
        w, h = mn.open_image(p).size
    except (FileNotFoundError, KeyError, OSError):
        w, h = 16, 9
    return max(1, round(H * w / h))


def build_one(job):
    imdb_id, n_t, n_f, thumbs = job
    out = {}
    t_file, f_file = thumbs / f"{imdb_id}_t.webp", thumbs / f"{imdb_id}_f.webp"
    tw, fw = round(H * 16 / 9), film_width(imdb_id)
    t_paths = [CACHE / "trailer_frames" / imdb_id / f"shot_{i:04d}_img_1.jpg" for i in range(n_t)]
    f_paths = [mn.keyframe_paths(imdb_id, i)[1] for i in range(n_f)]
    for key, file, paths, w in [("t", t_file, t_paths, tw), ("f", f_file, f_paths, fw)]:
        if file.exists():
            with Image.open(file) as im:
                cols, rows = im.size[0] // w, im.size[1] // H
        else:
            cols, rows = sheet(paths, file, w)
        out[key] = dict(file=file.name, cols=cols, rows=rows, w=w, h=H)
    return imdb_id, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site")
    ap.add_argument("--frames-base", default="thumbs/", help="URL prefix of the thumbnail sheets as the page sees it")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = (ROOT / a.out).resolve()
    thumbs = out / "thumbs"
    (out / "data").mkdir(parents=True, exist_ok=True)
    thumbs.mkdir(exist_ok=True)

    page = (VIZ / "index.html").read_text()
    config = f'<script>window.SITE = {json.dumps(dict(mode="public", frames=a.frames_base))};</script>'
    assert "<!-- site-config -->" in page
    (out / "index.html").write_text(page.replace("<!-- site-config -->", config))
    shutil.copy(VIZ / "ytmedia.js", out / "ytmedia.js")
    shutil.copy(VIZ / "rq1.html", out / "rq1.html")
    shutil.copytree(VIZ / "rq2", out / "rq2", dirs_exist_ok=True)
    (out / "rq2" / "data" / "features").mkdir(parents=True, exist_ok=True)
    shutil.copy(FEATURES / "clusters.csv", out / "rq2" / "data" / "features" / "clusters.csv")
    for f in ["index.json", "rq1.json"]:
        shutil.copy(VIZ / "data" / f, out / "data" / f)

    yt = dict(pd.read_csv(IDS_DIR / "trailer_status.csv", dtype=str)[["imdb_id", "youtube_id"]].values)
    ids = [d["id"] for d in json.loads((VIZ / "data" / "index.json").read_text())]
    ids = ids[: a.limit] if a.limit else ids
    docs = {i: json.loads((VIZ / "data" / f"{i}.json").read_text()) for i in ids}
    jobs = [(i, len(docs[i]["shots"]), docs[i]["film"]["n"], thumbs) for i in ids]
    with Pool(a.workers) as pool:
        for imdb_id, sheets in tqdm(pool.imap_unordered(build_one, jobs), total=len(jobs), desc="sheets"):
            d = docs[imdb_id]
            d["yt"] = yt.get(imdb_id)
            d["sheets"] = sheets
            (out / "data" / f"{imdb_id}.json").write_text(json.dumps(d, separators=(",", ":")))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{len(ids)} trailers -> {out} ({size / 1e9:.2f} GB, {sum(1 for _ in out.rglob('*'))} files)")


if __name__ == "__main__":
    main()
