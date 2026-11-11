"""Readers for the MovieNet files we use: shots, keyframes, story alignment, subtitles, meta."""
import ast
import csv
import io
import json
import os
import re
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd

from PIL import Image

from .paths import FRAMES, IDS, IDS_DIR, MOVIENET


def core_ids():
    return [r["imdb_id"] for r in csv.DictReader(open(IDS))]


@lru_cache(None)
def video_info():
    return json.load(open(MOVIENET / "movie1K.video_info.v1.json"))


def shots(imdb_id):
    """One row per film shot: frame range, time range (s), and the 3 keyframe frame indices."""
    arr = np.loadtxt(MOVIENET / "shot" / f"{imdb_id}.txt", dtype=int, ndmin=2)
    info = video_info()[imdb_id]
    df = pd.DataFrame(arr[:, :2], columns=["frame_start", "frame_end"])
    df.insert(0, "shot_idx", np.arange(len(df)))
    df["t_start"] = df.frame_start / info["fps"]
    df["t_end"] = (df.frame_end + 1) / info["fps"]
    df["shot_len"] = df.t_end - df.t_start
    df["story_pos"] = (df.t_start + df.t_end) / 2 / info["duration"]
    return df


def keyframe_name(shot_idx, i):
    return f"shot_{shot_idx:04d}_img_{i}.jpg"


def keyframe_paths(imdb_id, shot_idx):
    """Keyframes live either in frames/{id}/ or, to save inodes on clusters, in frames/{id}.zip.
    Zip members come back as "path/to/{id}.zip::member", which open_image understands."""
    d = FRAMES / imdb_id
    if d.is_dir():
        return [d / keyframe_name(shot_idx, i) for i in range(3)]
    return [f"{FRAMES / imdb_id}.zip::{keyframe_name(shot_idx, i)}" for i in range(3)]


def keyframe_count(imdb_id):
    d, z = FRAMES / imdb_id, FRAMES / f"{imdb_id}.zip"
    if d.is_dir():
        return len(list(d.glob("*.jpg")))
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            return len(zf.namelist())
    return 0


@lru_cache(None)
def _zip(path, pid):
    # one handle per process: DataLoader workers must not share a file offset
    return zipfile.ZipFile(path)


def open_image(ref):
    ref = str(ref)
    if "::" in ref:
        path, member = ref.split("::")
        return Image.open(io.BytesIO(_zip(path, os.getpid()).read(member)))
    return Image.open(ref)


def story(imdb_id):
    """Synopsis paragraphs aligned to shot ranges (only for the 328 story-annotated films)."""
    ann = json.load(open(MOVIENET / "annotation" / f"{imdb_id}.json"))
    rows = []
    for i, p in enumerate(ann.get("story") or []):
        rows.append(dict(para_idx=i, shot_lo=p["shot"][0], shot_hi=p["shot"][1],
                         t_lo=p["duration"][0], t_hi=p["duration"][1], text=p["description"]))
    return pd.DataFrame(rows)


_TS = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _sec(ts):
    h, m, s, ms = map(int, _TS.match(ts.strip()).groups())
    return h * 3600 + m * 60 + s + ms / 1000


def subtitles(imdb_id):
    path = MOVIENET / "subtitle" / f"{imdb_id}.srt"
    if not path.exists():
        return pd.DataFrame(columns=["t_start", "t_end", "text"])
    raw = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "")
    rows = []
    for block in raw.strip().split("\n\n"):
        lines = block.strip().split("\n")
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        a, b = lines[idx].split("-->")
        try:
            rows.append(dict(t_start=_sec(a), t_end=_sec(b), text=" ".join(lines[idx + 1:]).strip()))
        except AttributeError:
            continue
    return pd.DataFrame(rows)


# data/ids/imdb_meta.csv, from IMDb's title.basics (non-commercial datasets): IMDb's current id, its
# current genres (up to 3), and title and year for the films whose MovieNet title is cut at a hyphen
# or colon ("X" for X-Men) and so has no "(year)"
YEAR = re.compile(r"\((?:TV Movie |Video )?(\d{4})\)")


@lru_cache(None)
def imdb_meta():
    return pd.read_csv(IDS_DIR / "imdb_meta.csv", dtype={"year": "Int64"}, keep_default_na=False,
                       na_values={"year": [""]}).set_index("imdb_id")


def meta(imdb_id):
    m = json.load(open(MOVIENET / "meta" / f"{imdb_id}.json"))
    title = m.get("title") or ""
    year = YEAR.search(title)
    year = int(year.group(1)) if year else None
    fix = imdb_meta().loc[imdb_id] if imdb_id in imdb_meta().index else None
    if year is None and fix is not None and fix.title:
        year = int(fix.year) if pd.notna(fix.year) else None
        title = f"{fix.title} ({year})" if year else fix.title
    genres = m.get("genres")
    genres = ast.literal_eval(genres) if isinstance(genres, str) and genres.startswith("[") else (genres or [])
    # MovieNet's genres are an older IMDb snapshot: add the ones IMDb lists now (Jaws: Horror)
    extra = fix.genres_imdb.split("|") if fix is not None and fix.genres_imdb else []
    genres_movienet, genres = genres, sorted(set(genres) | set(extra))
    return dict(imdb_id=imdb_id, title=title, year=year,
                imdb_current=fix.imdb_current if fix is not None else imdb_id,
                genres=genres, genres_movienet=genres_movienet, rating=float(m["imdb_rating"]) if m.get("imdb_rating") else None,
                duration=video_info()[imdb_id]["duration"])
