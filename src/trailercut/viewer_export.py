"""Export compact per-trailer JSON for the D3 views in viz/.

One file per trailer so the page loads only what it shows, plus an index for the picker.
Model predictions and SHAP are included when model.py has been run; the view works without them.

Outputs (gitignored, rebuilt in seconds)
  viz/data/index.json        one entry per trailer: id, title, year, genre, shots, match rate
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
from .paths import FEATURES, ROOT

OUT = ROOT / "viz" / "data"
GROUP_ORDER = ["position", "dialogue", "scale", "content", "editing", "clip"]


def r(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def load_model_outputs():
    preds, shap = FEATURES / "model_preds.parquet", FEATURES / "model_shap.parquet"
    if not preds.exists() or not shap.exists():
        print("no model outputs yet; exporting without trailerness/SHAP")
        return None, None
    return pd.read_parquet(preds), pd.read_parquet(shap)


def shap_groups(shap):
    cols = {g: [c for c in (GROUPS[g] or []) if c in shap.columns] for g in GROUP_ORDER if g != "clip"}
    cols["content"] = [c for c in shap.columns if c.startswith("c_")]
    out = pd.DataFrame({g: shap[c].sum(1) for g, c in cols.items()})
    out["clip"] = shap["clip"]
    return out[GROUP_ORDER], cols


def trailer_json(imdb_id, curves, films, preds, shap, n_detail):
    al = pd.read_csv(FEATURES / "alignment" / f"{imdb_id}.csv")
    m = mn.meta(imdb_id)
    out = dict(
        id=imdb_id, title=m["title"], year=m["year"], genres=m["genres"],
        duration=r(al.t_end.max(), 2),
        curves=curves.get(imdb_id),
        # [t_start, t_end, matched, film_shot or -1, score, blank]
        shots=[[r(s.t_start, 2), r(s.t_end, 2), int(s.matched), int(s.film_shot) if s.matched else -1,
                r(s.score), int(s.blank)] for s in al.itertuples()],
    )
    f = films[films.imdb_id == imdb_id]
    film = dict(n=len(f), story_pos=[r(x) for x in f.story_pos], shot_len=[r(x, 2) for x in f.shot_len],
                in_trailer=f.in_trailer.astype(int).tolist())
    if preds is not None:
        rows = preds.imdb_id.values == imdb_id
        p = preds[rows]
        groups, cols, raw = shap[0][rows], shap[1], shap[2][rows]
        film["trailerness"] = [r(x, 4) for x in p.full]
        film["trailerness_readable"] = [r(x, 4) for x in p.readable]
        film["shap_groups"] = GROUP_ORDER
        film["shap"] = [[r(x) for x in row] for row in groups.values]
        # named features (value + SHAP) only for shots someone will hover: used in the trailer or ranked high
        pick = set(np.flatnonzero(f.in_trailer.values)) | set(np.argsort(-p.full.values)[:n_detail])
        feats = [c for g in GROUP_ORDER if g != "clip" for c in cols[g]]
        film["detail_features"] = feats
        film["detail"] = {int(i): [[r(f[c].values[i]), r(raw[c].values[i])] for c in feats] for i in sorted(pick)}
    out["film"] = film
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", type=int, default=60, help="top-ranked film shots per film with feature detail")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    curves = json.loads((FEATURES / "trailer_curves.json").read_text())
    films = pd.read_parquet(FEATURES / "film_shots.parquet")
    preds, shap = load_model_outputs()
    if preds is not None:
        shap = (*shap_groups(shap), shap)
    index = []
    for i in tqdm(sorted(films.imdb_id.unique())):
        d = trailer_json(i, curves, films, preds, shap, a.detail)
        (OUT / f"{i}.json").write_text(json.dumps(d, separators=(",", ":")))
        matched = [s[2] for s in d["shots"]]
        index.append(dict(id=i, title=d["title"], year=d["year"], genre=(d["genres"] or [""])[0],
                          shots=len(matched), match_rate=r(np.mean(matched))))
    (OUT / "index.json").write_text(json.dumps(index, separators=(",", ":")))
    print(len(index), "trailers ->", OUT)

if __name__ == "__main__":
    main()
