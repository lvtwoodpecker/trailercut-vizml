"""Export compact per-trailer JSON for the D3 views in viz/.

One file per trailer so the page loads only what it shows, plus an index for the picker.
Model predictions and SHAP are included when model.py has been run; the view works without them.
They are joined by shot, so logo/credit shots at the film's edges (skipped by the model) have no score.
Trailer speech segments come with their film position when dialogue_align matched them to a subtitle
(text included: viz/data is local and gitignored).
Each film and trailer shot also gets the average color of its middle keyframe, so the view can draw
the film as a "movie barcode" (cached in cache/shot_colors/, ~1 ms per keyframe).

Outputs (gitignored, rebuilt in seconds)
  viz/data/index.json        one entry per trailer: id, title, year, genre, shots, match rate
  viz/data/{imdb_id}.vtt     captions for the video player, from the Whisper transcript
  viz/data/{imdb_id}.json    trailer shots + matches, audio/cut curves, film shots with trailerness
                             and grouped SHAP, named-feature detail for the shots worth hovering

  python -m trailercut.viewer_export [--detail 60]
"""
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import movienet_io as mn
from .model import GROUPS
from .paths import CACHE, FEATURES, ROOT

OUT = ROOT / "viz" / "data"
COLORS = CACHE / "shot_colors"
GROUP_ORDER = ["position", "dialogue", "scale", "content", "editing", "clip"]


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


def shot_colors(imdb_id, n_film, n_trailer):
    path = COLORS / f"{imdb_id}.json"
    if path.exists():
        c = json.loads(path.read_text())
        if len(c["film"]) == n_film and len(c["trailer"]) == n_trailer:
            return c
    frames = CACHE / "trailer_frames" / imdb_id
    c = dict(film=[avg_color(mn.keyframe_paths(imdb_id, s)[1]) for s in range(n_film)],
             trailer=[avg_color(frames / mn.keyframe_name(s, 1)) for s in range(n_trailer)])
    COLORS.mkdir(parents=True, exist_ok=True)
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


def trailer_json(imdb_id, curves, films, preds, shap, n_detail):
    al = pd.read_csv(FEATURES / "alignment" / f"{imdb_id}.csv")
    m = mn.meta(imdb_id)
    out = dict(
        id=imdb_id, title=m["title"], year=m["year"], genres=m["genres"],
        duration=r(al.t_end.max(), 2),
        curves=curves.get(imdb_id),
        # [t_start, t_end, matched, film_shot or -1, score, blank, film shot placed by dialogue or -1]
        shots=[[r(s.t_start, 2), r(s.t_end, 2), int(s.matched), int(s.film_shot) if s.matched else -1,
                r(s.score), int(s.blank)] for s in al.itertuples()],
    )
    f = films[films.imdb_id == imdb_id]
    runtime = f.t_end.max()
    colors = shot_colors(imdb_id, len(f), len(al))
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
    # shots the picture matcher left unmatched, but whose spoken line was found in the film's subtitles,
    # get placed at that line's film moment (marked separately so the view can show the difference)
    starts = f.t_start.values
    for sh in out["shots"]:
        mid = (sh[0] + sh[1]) / 2
        line = next((d for d in out["speech"] if d[2] and d[0] <= mid < d[1]), None)
        placed = -1
        if not sh[2] and not sh[5] and line is not None:
            placed = int(max(0, np.searchsorted(starts, line[3] * runtime, side="right") - 1))
        sh.append(placed)
    return out


def speech(imdb_id, runtime):
    """[t_start, t_end, matched, film position of the matched line (0 to 1) or null, text,
    words as [start, end, word]] per segment."""
    csv, asr = FEATURES / "dialogue_alignment" / f"{imdb_id}.csv", CACHE / "trailer_asr" / f"{imdb_id}.json"
    if not csv.exists() or not asr.exists():
        return []
    segs = json.loads(asr.read_text())["segments"]
    return [[r(d.t_start, 2), r(d.t_end, 2), int(d.matched), r(d.film_t_start / runtime, 4) if d.matched else None,
             segs[d.seg_idx]["text"], [[w["start"], w["end"], w["word"]] for w in segs[d.seg_idx]["words"]]]
            for d in pd.read_csv(csv).itertuples()]


def vtt(imdb_id):
    """WebVTT captions from the trailer transcript, so the browser's player can show and toggle them."""
    asr = CACHE / "trailer_asr" / f"{imdb_id}.json"
    if not asr.exists():
        return
    ts = lambda t: f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}"
    cues = [f"{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}" for s in json.loads(asr.read_text())["segments"]]
    (OUT / f"{imdb_id}.vtt").write_text("WEBVTT\n\n" + "\n\n".join(cues) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=int, default=60, help="top-ranked film shots per film with feature detail")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    curves = json.loads((FEATURES / "trailer_curves.json").read_text())
    films = pd.read_parquet(FEATURES / "film_shots.parquet")
    preds, shap = load_model_outputs()
    if preds is not None:
        # the model skips logo/credit shots at the film's edges; join on shot so rows line up, edges get no score
        key = films[["imdb_id", "shot_idx"]]
        preds = key.merge(preds.drop(columns=["shot_len", "in_trailer"]), how="left", on=["imdb_id", "shot_idx"])
        shap = key.merge(shap, how="left", on=["imdb_id", "shot_idx"])
        shap = (*shap_groups(shap), shap)
    index = []
    for i in tqdm(sorted(films.imdb_id.unique())):
        d = trailer_json(i, curves, films, preds, shap, a.detail)
        (OUT / f"{i}.json").write_text(json.dumps(d, separators=(",", ":"), allow_nan=False))
        vtt(i)
        matched = [s[2] for s in d["shots"]]
        index.append(dict(id=i, title=d["title"], year=d["year"], genre=(d["genres"] or [""])[0],
                          shots=len(matched), match_rate=r(np.mean(matched))))
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(len(index), "trailers ->", OUT)

if __name__ == "__main__":
    main()
