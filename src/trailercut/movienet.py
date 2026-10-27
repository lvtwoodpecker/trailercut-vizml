"""
MovieNet fetcher (no OpenDataLab account needed).

Source: unofficial, ungated Hugging Face mirror ZhengPeng7/MovieNet
(same files as the official release: annotation.v1.zip, meta.v1.zip,
subtitle1K.v1.zip, keyframes split into frames_part_aa..ai, trailer URL list).

Usage
  pip install requests tqdm
  python movienet_fetch.py small                     # annotation + meta + subtitles + json lists (~670 MB)
  python movienet_fetch.py keyframes --ids core_films_293.csv --limit 20
  python movienet_fetch.py trailers  --ids core_films_293.csv --limit 20   # needs yt-dlp

The keyframes command does NOT download the 172 GB archive. It reads the zip's
central directory once with HTTP range requests (~565 MB, cached to disk),
then pulls only the byte range for each requested film (~100-200 MB per film).
"""
import argparse
import csv
import io
import os
import pickle
import struct
import subprocess
import sys
import zipfile
import zlib

import requests

BASE = "https://huggingface.co/datasets/ZhengPeng7/MovieNet/resolve/main/"
SMALL_FILES = [
    "annotation.v1.zip",            # 55 MB: story (synopsis->shot alignment, 328 films), scene, cast, cinematic_style
    "meta.v1.zip",                  # 564 MB: IMDb/TMDb ids, genres, rating, director, cast, synopsis
    "subtitle1K.v1.zip",            # 31 MB
    "movie1K.video_info.v1.json",   # duration, fps, frame count for the 1,100 films with keyframes
    "movie1K.split.v1.json",
    "movie1K.shot_detection.v1.zip",
    "trailer.urls.unique30K.v1.json",
    "list.v1.txt",
]
GB20 = 21474836480
FRAME_PARTS = [(f"frames_part_a{c}", GB20) for c in "abcdefgh"] + [("frames_part_ai", 2311599378)]

S = requests.Session()


def download(name, outdir):
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, name)
    if os.path.exists(dst):
        print("have", dst)
        return dst
    with S.get(BASE + name, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dst + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        os.replace(tmp, dst)
    print("got", dst)
    return dst


class RangeFile(io.RawIOBase):
    """Seekable read-only view over the split frames zip, backed by HTTP range requests."""

    def __init__(self, parts):
        self.parts = []
        start = 0
        for name, size in parts:
            self.parts.append((name, start, size))
            start += size
        self.size = start
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            name, pstart, psize = next(p for p in self.parts if p[1] <= self.pos < p[1] + p[2])
            lo = self.pos - pstart
            hi = min(psize, lo + n) - 1
            r = S.get(BASE + name, headers={"Range": f"bytes={lo}-{hi}"}, timeout=600)
            if r.status_code != 206:
                raise IOError(f"range request failed: {r.status_code}")
            out += r.content
            got = hi - lo + 1
            self.pos += got
            n -= got
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def load_index(cache):
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    print("Reading frames central directory (~565 MB, one time)...")
    zf = zipfile.ZipFile(RangeFile(FRAME_PARTS))
    idx = {}
    for zi in zf.infolist():
        parts = zi.filename.split("/")
        if len(parts) == 3 and parts[2]:
            idx.setdefault(parts[1], []).append((zi.filename, zi.header_offset, zi.compress_size, zi.compress_type))
    with open(cache, "wb") as f:
        pickle.dump(idx, f)
    print(f"Indexed {len(idx)} films")
    return idx


def fetch_film(rf, entries, outdir):
    entries = sorted(entries, key=lambda e: e[1])
    lo = entries[0][1]
    hi = entries[-1][1] + entries[-1][2] + 30 + len(entries[-1][0]) + 1024
    total = sum(e[2] for e in entries)
    if hi - lo > 3 * total + (1 << 20):
        raise RuntimeError("film entries are not contiguous; would need per-file reads")
    rf.seek(lo)
    blob = rf.read(min(hi, rf.size) - lo)
    os.makedirs(outdir, exist_ok=True)
    for name, off, csize, method in entries:
        p = off - lo
        sig, = struct.unpack_from("<I", blob, p)
        assert sig == 0x04034B50, "bad local header"
        fl, el = struct.unpack_from("<HH", blob, p + 26)
        data = blob[p + 30 + fl + el: p + 30 + fl + el + csize]
        if method == 8:
            data = zlib.decompressobj(-15).decompress(data)
        with open(os.path.join(outdir, os.path.basename(name)), "wb") as f:
            f.write(data)
    return len(entries)


def read_ids(path, limit):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["small", "keyframes", "trailers"])
    ap.add_argument("--ids", default="core_films_293.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="movienet")
    a = ap.parse_args()

    if a.cmd == "small":
        for name in SMALL_FILES:
            download(name, a.out)
    elif a.cmd == "keyframes":
        idx = load_index(os.path.join(a.out, "frames_index.pkl"))
        rf = RangeFile(FRAME_PARTS)
        for row in read_ids(a.ids, a.limit):
            mid = row["imdb_id"]
            dst = os.path.join(a.out, "frames", mid)
            if mid not in idx:
                print("no keyframes for", mid)
                continue
            if os.path.isdir(dst) and len(os.listdir(dst)) == len(idx[mid]):
                print("have", mid)
                continue
            n = fetch_film(rf, idx[mid], dst)
            print(f"{mid}: {n} keyframes")
    elif a.cmd == "trailers":
        dst = os.path.join(a.out, "trailers")
        os.makedirs(dst, exist_ok=True)
        for row in read_ids(a.ids, a.limit):
            url = "https://www.youtube.com/watch?v=" + row["youtube_id"]
            subprocess.run(["yt-dlp", "-f", "mp4[height<=480]/best[height<=480]", "-o",
                            os.path.join(dst, row["imdb_id"] + ".%(ext)s"), url])


if __name__ == "__main__":
    sys.exit(main())
