"""Readers for the MovieNet files we use: shots, keyframes, story alignment, subtitles, meta."""
import ast
import csv
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .paths import FRAMES, IDS, MOVIENET


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


def keyframe_paths(imdb_id, shot_idx):
    return [FRAMES / imdb_id / f"shot_{shot_idx:04d}_img_{i}.jpg" for i in range(3)]


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


# core films whose MovieNet title has no "(year)"
YEAR_FIX = {"tt0076759": 1977, "tt0080684": 1980, "tt0086190": 1983, "tt0120915": 1999,
            "tt0121765": 2002, "tt0145487": 2002, "tt2488496": 2015}


def meta(imdb_id):
    m = json.load(open(MOVIENET / "meta" / f"{imdb_id}.json"))
    title = m.get("title") or ""
    year = re.search(r"\((\d{4})", title)
    genres = m.get("genres")
    genres = ast.literal_eval(genres) if isinstance(genres, str) and genres.startswith("[") else (genres or [])
    return dict(imdb_id=imdb_id, title=title, year=int(year.group(1)) if year else YEAR_FIX.get(imdb_id),
                genres=genres, rating=float(m["imdb_rating"]) if m.get("imdb_rating") else None,
                duration=video_info()[imdb_id]["duration"])
