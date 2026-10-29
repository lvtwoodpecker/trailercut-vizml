"""Flag trailers that aren't the film's original theatrical trailer in its original language.

Rules only (reproducible), from the YouTube metadata of the link in use (fetched without downloading),
the transcript and the alignment:
  not_trailer   title says TV spot, teaser, clip, scene, featurette, Blu-ray/DVD, review, fan made, ...
  rerelease     anniversary, re-release, restoration, remaster, 4K, or a year in the title far from the film's
  ad            the narration sells a home-video edition (Blu-ray, DVD, collector's edition, "own it")
  short / long  shorter than 60 s (TV spots, teasers) / longer than 240 s
  low_res       best available picture under 360p (VHS-era uploads)
  clip          matched shots span under 20% of the film
  dub           confident spoken language differs from the film's (from its first production country)

Hard flags (not_trailer, ad, clip, dub, short) mean replace; soft ones (rerelease, low_res, long)
mean replace only if a better candidate exists. `swap` column marks the hard cases.
Writes data/ids/trailer_audit.csv (committed). Metadata is cached in cache/yt_meta/.

  python -m trailercut.trailer_audit
"""
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yt_dlp

from . import movienet_io as mn
from .paths import CACHE, FEATURES, IDS

META = CACHE / "yt_meta"
HARD = {"not_trailer", "ad", "clip", "dub", "short"}
OUT = IDS.parent / "trailer_audit.csv"
NOT_TRAILER = re.compile(r"tv spot|teaser|\bclip\b|\bscene\b|featurette|behind the|making of|blu-?ray|\bdvd\b|\bvhs\b|"
                         r"interview|review|reaction|fan.?made|music video|soundtrack|\bost\b|recap|\bending\b|promo", re.I)
RERELEASE = re.compile(r"anniversary|re-?release|restor|remaster|\b4k\b|reissue|special edition", re.I)
AD = re.compile(r"blu-?ray|\bdvd\b|collector|special edition|own it|available now on|on video", re.I)
LANG = {"USA": "en", "UK": "en", "Canada": "en", "Australia": "en", "Ireland": "en", "New Zealand": "en",
        "France": "fr", "Germany": "de", "West Germany": "de", "Italy": "it", "Spain": "es", "Mexico": "es",
        "Japan": "ja", "China": "zh", "Hong Kong": "zh", "Taiwan": "zh", "South Korea": "ko", "India": "hi",
        "Denmark": "da", "Sweden": "sv", "Norway": "no", "Brazil": "pt", "Czech Republic": "cs", "Russia": "ru"}


def yt_meta(yid):
    path = META / f"{yid}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as y:
            i = y.extract_info(f"https://www.youtube.com/watch?v={yid}", download=False)
        m = dict(title=i.get("title"), channel=i.get("channel"), duration=i.get("duration"),
                 height=max((f.get("height") or 0) for f in i.get("formats", [])))
    except Exception as e:  # YouTube rate-limits metadata requests; don't cache failures
        return dict(error=str(e).splitlines()[0][:120])
    META.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m))
    return m


def audit(row, meta):
    i = row["imdb_id"]
    f = mn.meta(i)
    # fall back to what was logged at download time when YouTube refuses the metadata request
    title = meta.get("title") or row.get("title") or ""
    flags = []
    if NOT_TRAILER.search(title):
        flags.append("not_trailer")
    year_re = r"\b(19[2-9]\d|20[0-2]\d)\b"
    own = set(re.findall(year_re, re.sub(r"\(\d{4}\)", "", f["title"])))  # "2001: A Space Odyssey"
    years = [int(y) for y in re.findall(year_re, title) if y not in own]
    if RERELEASE.search(title) or (f["year"] and any(abs(y - f["year"]) >= 3 for y in years)):
        flags.append("rerelease")
    asr_path = CACHE / "trailer_asr" / f"{i}.json"
    asr = json.loads(asr_path.read_text()) if asr_path.exists() else {"segments": [], "language": None}
    text = " ".join(s["text"] for s in asr["segments"])
    if AD.search(text):
        flags.append("ad")
    dur = meta.get("duration") or float(row.get("duration") or 0)
    if dur and dur < 60:
        flags.append("short")
    if dur and dur > 240:
        flags.append("long")
    if meta.get("height") and meta["height"] < 360:
        flags.append("low_res")
    al = pd.read_csv(FEATURES / "alignment" / f"{i}.csv")
    pos = al.film_story_pos.dropna()
    span = float(pos.quantile(0.9) - pos.quantile(0.1)) if len(pos) else 0.0
    if span < 0.2:
        flags.append("clip")
    country = json.loads((mn.MOVIENET / "meta" / f"{i}.json").read_text()).get("country") or ""
    want = LANG.get(country.split(",")[0].strip())
    speech = sum(s["end"] - s["start"] for s in asr["segments"])
    if want and asr["language"] and asr["language"] != want and asr.get("language_p", 0) >= 0.8 and speech > 15:
        flags.append("dub")
    return dict(imdb_id=i, film=f["title"], youtube_id=row["youtube_id"], source=row.get("source", "movienet"),
                yt_title=title, channel=meta.get("channel") or "", duration=dur, height=meta.get("height") or "",
                language=asr["language"], expected_language=want or "", span=round(span, 3),
                flags=" ".join(flags), n_flags=len(flags), swap=bool(set(flags) & HARD))


def main():
    rows = list(csv.DictReader(open(IDS.parent / "trailer_status.csv")))
    with ThreadPoolExecutor(4) as ex:
        metas = list(ex.map(lambda r: yt_meta(r["youtube_id"]), rows))
    out = pd.DataFrame([audit(r, m) for r, m in zip(rows, metas)]).sort_values(["n_flags", "imdb_id"], ascending=[False, True])
    out.to_csv(OUT, index=False)
    flagged = out[out.n_flags > 0]
    print(f"{len(flagged)} of {len(out)} trailers flagged")
    print(flagged["flags"].str.split().explode().value_counts().to_string())


if __name__ == "__main__":
    main()
