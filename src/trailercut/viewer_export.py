"""Export compact per-trailer JSON for the D3 views in viz/.

One file per trailer so the page loads only what it shows, plus an index for the picker.
Model predictions and SHAP are included when model.py has been run; the view works without them.
They are joined by shot, so logo/credit shots at the film's edges (skipped by the model) have no score.
Trailer speech segments come with their film position when dialogue_align matched them to a subtitle
(text included: viz/data is local and gitignored).
Each film and trailer shot also gets the average color of its middle keyframe, so the view can draw
the film as a "movie barcode" (cached in cache/shot_colors/, ~1 ms per keyframe).
Loudness and cut curves are computed here too (cached in cache/curves/, recomputed when the trailer's
shot table changes), so the views don't need the RQ2 step (trailer_features).

Outputs (gitignored, rebuilt in seconds)
  viz/data/index.json        one entry per trailer: id, title, year, genre, shots, match rate
  viz/data/{imdb_id}.json    trailer shots + matches, audio/cut curves, film shots with trailerness
                             and grouped SHAP, named-feature detail for the shots worth hovering

  python -m trailercut.viewer_export [--detail 60]
"""
import argparse
import json
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import movienet_io as mn
from .model import GROUPS
from .paths import CACHE, FEATURES, ROOT, IDS_DIR
# from .film_features import film_table

OUT = ROOT / "viz" / "data"
COLORS = CACHE / "shot_colors"
GROUP_ORDER = ["position", "dialogue", "scale", "content", "editing", "clip"]

def youtube_ids():
    path = IDS_DIR / "trailer_status.csv"

    if not path.exists():
        print(f"WARNING: {path} not found")
        return {}

    with open(path, newline="") as f:
        result = {
            row["imdb_id"]: row["youtube_id"].strip()
            for row in csv.DictReader(f)
            if row.get("imdb_id") and row.get("youtube_id", "").strip()
        }

    print(f"Loaded {len(result)} YouTube IDs")

    return result

def r(x, n=3):
    if x is None:
        return None
    x = float(x)  # numpy float32 NaN is not a Python float, so convert before checking
    return None if np.isnan(x) else round(x, n)


def avg_color(ref):
    try:
        im = mn.open_image(ref)
        im.draft("RGB", (40, 24))  # JPEG decode at reduced size; plenty for a mean color
        rgb = np.asarray(im.convert("RGB").resize((8, 5))).reshape(-1, 3).mean(0)
        return "#%02x%02x%02x" % tuple(int(v) for v in rgb)
    except (OSError, KeyError):
        return None


def shot_colors(imdb_id, n_film, n_trailer, enabled=True):
    if not enabled:
        return {
            "film": [None] * n_film,
            "trailer": [None] * n_trailer
        }

    path = COLORS / f"{imdb_id}.json"

    if path.exists():
        c = json.loads(path.read_text())
        if len(c["film"]) == n_film and len(c["trailer"]) == n_trailer:
            print(f"Using cached colors: {imdb_id}", flush=True)
            return c

    print(f"Calculating film colors: {imdb_id}", flush=True)

    film_paths = [
        mn.keyframe_paths(imdb_id, s)[1]
        for s in range(n_film)
    ]

    print("First film frame:", film_paths[0], flush=True)

    film_colors = [
        avg_color(ref)
        for ref in tqdm(film_paths, desc="Film colors")
    ]

    print(
        f"Film colors: {sum(c is not None for c in film_colors)}/{n_film}",
        flush=True
    )

    frames = CACHE / "trailer_frames" / imdb_id

    trailer_colors = [
        avg_color(frames / mn.keyframe_name(s, 1))
        for s in range(n_trailer)
    ]

    c = dict(film=film_colors, trailer=trailer_colors)

    COLORS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c))

    return c


def curve(imdb_id):
    """Loudness (dB) and cut counts on a 100-bin grid over the trailer, for the sound track and strip.
    Cached in cache/curves/, keyed on the shot table so a re-split or swap recomputes it."""
    from .trailer_features import GRID, audio
    shots_csv = FEATURES / "trailer_shots" / f"{imdb_id}.csv"
    path = CACHE / "curves" / f"{imdb_id}.json"
    stamp = shots_csv.stat().st_mtime
    if path.exists():
        c = json.loads(path.read_text())
        if c.get("stamp") == stamp:
            return c
    sh = pd.read_csv(shots_csv)
    wav = CACHE / "trailer_audio" / f"{imdb_id}.wav"

    if wav.exists():
        try:
            _, rms = audio(imdb_id, sh.t_end.max())
        except Exception:
            rms = np.zeros(100)
    else:
        rms = np.zeros(100)
    c = dict(rms_db=np.round(rms, 2).tolist(),
             cuts=np.histogram(sh.trailer_pos, bins=GRID, range=(0, 1))[0].tolist(), stamp=stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c))
    return c


def load_model_outputs():
    preds, shap = FEATURES / "model_preds.parquet", FEATURES / "model_shap.parquet"
    if not preds.exists() or not shap.exists():
        print("no model outputs yet; exporting without trailerness/SHAP")
        return None, None
    return pd.read_parquet(preds), pd.read_parquet(shap)


def shap_groups(shap):
    cols = {g: [c for c in (GROUPS[g] or []) if c in shap.columns] for g in GROUP_ORDER if g != "clip"}
    cols["content"] = [c for c in shap.columns if c.startswith("c_")]
    out = pd.DataFrame({g: shap[c].sum(1, min_count=1) for g, c in cols.items()})
    out["clip"] = shap["clip"]
    return out[GROUP_ORDER], cols


def trailer_json(imdb_id, curves, films, preds, shap, n_detail, yt_ids, include_colors=True):
    al = pd.read_csv(FEATURES / "alignment" / f"{imdb_id}.csv")
    m = mn.meta(imdb_id)
    out = dict(
        id=imdb_id, title=m["title"], year=m["year"], genres=m["genres"],
        duration=r(al.t_end.max(), 2),
        curves=curves.get(imdb_id),
        # [t_start, t_end, matched, film_shot or -1, score, blank]
        shots=[[r(s.t_start, 2), r(s.t_end, 2), int(s.matched), int(s.film_shot) if s.matched else -1,
                r(s.score), int(s.blank)] for s in al.itertuples()],
                youtube_id=yt_ids.get(imdb_id),
    )
    f = films[films.imdb_id == imdb_id]
    runtime = f.t_end.max()
    colors = shot_colors(imdb_id, len(f), len(al), enabled=include_colors)
    out["shot_colors"] = colors["trailer"]
    film = dict(n=len(f), runtime=r(runtime, 1), story_pos=[r(x) for x in f.story_pos], shot_len=[r(x, 2) for x in f.shot_len],
                # shot extent as a fraction of runtime, for the barcode
                start=[r(x / runtime, 5) for x in f.t_start], end=[r(x / runtime, 5) for x in f.t_end],
                color=colors["film"], in_trailer=f.in_trailer.astype(int).tolist())
    if preds is not None:
        rows = preds.imdb_id.values == imdb_id
        p = preds[rows]
        groups, cols, raw = shap[0][rows], shap[1], shap[2][rows]
        film["trailerness"] = [r(x, 4) for x in p.full]
        film["trailerness_readable"] = [r(x, 4) for x in p.readable]
        film["shap_groups"] = GROUP_ORDER
        film["shap"] = [[r(x) for x in row] for row in groups.values]
        # named features (value + SHAP) only for shots someone will hover: used in the trailer or ranked high
        pick = set(np.flatnonzero(f.in_trailer.values)) | set(np.argsort(-np.nan_to_num(p.full.values, nan=-1))[:n_detail])
        feats = [c for g in GROUP_ORDER if g != "clip" for c in cols[g]]
        film["detail_features"] = feats
        film["detail"] = {int(i): [[r(f[c].values[i]), r(raw[c].values[i])] for c in feats] for i in sorted(pick)}
    out["film"] = film
    out["speech"] = speech(imdb_id, runtime)
    return out


_WHISPER_MODEL = None

def get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import torch
        from faster_whisper import WhisperModel
        cuda = torch.cuda.is_available()
        _WHISPER_MODEL = WhisperModel(
            "small",
            device="cuda" if cuda else "cpu",
            compute_type="float16" if cuda else "int8",
        )
    return _WHISPER_MODEL


def speech(imdb_id, runtime):
    """[t_start, t_end, matched, film position of the matched line (0 to 1) or null, text,
    words as [start, end, word]] per segment."""
    csv_path = FEATURES / "dialogue_alignment" / f"{imdb_id}.csv"
    asr_path = CACHE / "trailer_asr" / f"{imdb_id}.json"
    wav_path = CACHE / "trailer_audio" / f"{imdb_id}.wav"

    # 1. Dynamically transcribe trailer audio if missing
    if not asr_path.exists() and wav_path.exists():
        from . import trailer_asr
        print(f"Transcribing trailer audio: {imdb_id}", flush=True)
        asr_path.parent.mkdir(parents=True, exist_ok=True)
        asr_data = trailer_asr.transcribe(get_whisper(), wav_path)
        asr_path.write_text(json.dumps(asr_data))

    # 2. Dynamically align dialogue to MovieNet subtitles if missing
    if not csv_path.exists() and asr_path.exists():
        from . import dialogue_align
        print(f"Aligning trailer dialogue: {imdb_id}", flush=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df, _ = dialogue_align.align_film(imdb_id)
        df.round(3).to_csv(csv_path, index=False)

    if not csv_path.exists() or not asr_path.exists():
        return []

    segs = json.loads(asr_path.read_text())["segments"]
    return [
        [
            r(d.t_start, 2),
            r(d.t_end, 2),
            int(d.matched),
            r(d.film_t_start / runtime, 4) if d.matched else None,
            segs[d.seg_idx]["text"],
            [[w["start"], w["end"], w["word"]] for w in segs[d.seg_idx]["words"]],
        ]
        for d in pd.read_csv(csv_path).itertuples()
    ]


# def vtt(imdb_id):
#     """WebVTT captions from the trailer transcript, so the browser's player can show and toggle them."""
#     asr = CACHE / "trailer_asr" / f"{imdb_id}.json"
#     if not asr.exists():
#         return
#     ts = lambda t: f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"
#     cues = []
#     for seg in json.loads(asr.read_text())["segments"]:
#         words = seg["words"] or [dict(start=seg["start"], end=seg["end"], word=seg["text"])]
#         chunk = []
#         for w in words:  # Whisper can merge ~30 s into one segment; cap cues at ~6 s or 12 words and break at pauses
#             if chunk and (w["end"] - chunk[0]["start"] > 6 or len(chunk) >= 12 or w["start"] - chunk[-1]["end"] > 1.5):
#                 cues.append(chunk)
#                 chunk = []
#             chunk.append(w)
#         if chunk:
#             cues.append(chunk)
#     cues = [f"{ts(c[0]['start'])} --> {ts(c[-1]['end'])}\n{' '.join(w['word'] for w in c)}" for c in cues]
#     (OUT / f"{imdb_id}.vtt").write_text("WEBVTT\n\n" + "\n\n".join(cues) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=str,required=True,help="IMDb ID to export")
    ap.add_argument("--detail", type=int, default=60, help="top-ranked film shots per film with feature detail")
    ap.add_argument("--no-colors", action="store_true", help="Skip frame image access and color extraction")
    a = ap.parse_args()
    yt_ids = youtube_ids()
    OUT.mkdir(parents=True, exist_ok=True)
    curves = {}  # computed here (cached per trailer), so the views don't depend on the RQ2 feature step
    imdb_id = a.ids

    # Build basic film shot metadata directly from MovieNet.
    # No CLIP embeddings or film_features.parquet required.

    shots = mn.shots(imdb_id)

    films = shots.copy()
    films.insert(0, "imdb_id", imdb_id)

    films["in_trailer"] = 0

    alignment_path = FEATURES / "alignment" / f"{imdb_id}.csv"

    if alignment_path.exists():
        al = pd.read_csv(alignment_path)
        matched = al.loc[al.matched, "film_shot"].astype(int)
        films.loc[matched.unique(), "in_trailer"] = 1

    films = films.reset_index(drop=True)
    # preds, shap = load_model_outputs()
    preds, shap = None, None
    if preds is not None:
        # the model skips logo/credit shots at the film's edges; join on shot so rows line up, edges get no score
        key = films[["imdb_id", "shot_idx"]]
        preds = key.merge(preds.drop(columns=["shot_len", "in_trailer"]), how="left", on=["imdb_id", "shot_idx"])
        shap = key.merge(shap, how="left", on=["imdb_id", "shot_idx"])
        shap = (*shap_groups(shap), shap)
    i = a.ids

    if i not in films.imdb_id.values:
        raise ValueError(f"Unknown IMDb ID: {i}")

    curves[i] = curve(i)

    d = trailer_json(
        i, curves, films, preds, shap,
        a.detail, yt_ids,
        include_colors=not a.no_colors
    )

    (OUT / f"{i}.json").write_text(
        json.dumps(d, separators=(",", ":"), allow_nan=False)
    )

    # Build the single-movie index
    matched = [s[2] for s in d["shots"]]

    index = [
        dict(
            id=i,
            title=d["title"],
            year=d["year"],
            genre=(d["genres"] or [""])[0],
            shots=len(matched),
            match_rate=r(np.mean(matched))
        )
    ]

    (OUT / "index.json").write_text(
        json.dumps(index, separators=(",", ":"))
    )

    print(f"Exported {i} -> {OUT}")

if __name__ == "__main__":
    main()
