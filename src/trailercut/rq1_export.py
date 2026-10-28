"""Export what gets picked across all films, for the RQ1 page (viz/rq1.html).

Every number is a lift: the pick rate of a kind of shot divided by the pick rate of the average shot.
Intervals are 95% from resampling films (not shots), so one film can't carry a result.
Uses the same shots as the model (logos/credits at the film's edges excluded, see model.EDGE).

Writes viz/data/rq1.json (gitignored, rebuilt in under a minute)

  python -m trailercut.rq1_export [--boot 300]
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import concepts
from .model import load
from .paths import FEATURES, ROOT

OUT = ROOT / "viz" / "data" / "rq1.json"
SCALES = {"scale_ecu": "extreme close-up", "scale_cu": "close-up", "scale_ms": "medium", "scale_fs": "full", "scale_ls": "long"}


def lift_table(df, key, boot, rng):
    """Lift and film-bootstrap CI for each value of df[key]."""
    films = df.imdb_id.unique()
    base_rate = df.in_trailer.mean()
    cnt = df.groupby(["imdb_id", key]).in_trailer.agg(["size", "sum"]).unstack(fill_value=0)
    n, pos = cnt["size"].reindex(films).fillna(0).values, cnt["sum"].reindex(films).fillna(0).values
    levels = list(cnt["size"].columns)
    tot_n, tot_pos = n.sum(1), pos.sum(1)
    draws = []
    for _ in range(boot):
        idx = rng.integers(0, len(films), len(films))
        rate = pos[idx].sum(0) / np.maximum(n[idx].sum(0), 1)
        draws.append(rate / (tot_pos[idx].sum() / tot_n[idx].sum()))
    draws = np.array(draws)
    lift = pos.sum(0) / np.maximum(n.sum(0), 1) / base_rate
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return [dict(level=lv if isinstance(lv, str) else float(lv), lift=float(l), lo=float(a), hi=float(b), n=int(m))
            for lv, l, a, b, m in zip(levels, lift, lo, hi, n.sum(0))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    df = load(pcs=2)  # only need the context/relative features here
    out = dict(n_shots=len(df), n_films=int(df.imdb_id.nunique()), positive_rate=float(df.in_trailer.mean()))

    df["pos_bin"] = (np.clip(df.story_pos, 0, 0.9999) * 20).astype(int) / 20 + 0.025
    out["story"] = lift_table(df, "pos_bin", a.boot, rng)
    df["scale"] = df[list(SCALES)].idxmax(axis=1).map(SCALES)
    out["scale"] = sorted(lift_table(df, "scale", a.boot, rng), key=lambda r: list(SCALES.values()).index(r["level"]))
    quint = lambda c, names: pd.cut(df[c], [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=names, include_lowest=True).astype(str)
    df["len_q"] = quint("rel_shot_len", ["shortest 20%", "short", "middle", "long", "longest 20%"])
    out["length"] = lift_table(df, "len_q", a.boot, rng)
    df["distinct_q"] = quint("rel_distinct", ["most typical 20%", "typical", "middle", "unusual", "most unusual 20%"])
    out["distinct"] = lift_table(df, "distinct_q", a.boot, rng)
    df["dialogue"] = np.where(df.dlg_has > 0, "has dialogue", "no dialogue")
    out["dialogue"] = lift_table(df, "dialogue", a.boot, rng)
    # concept: shots in the film's top 10% for that concept vs the rest
    rows = []
    for c in concepts.CONTENT:
        df["_top"] = np.where(df[f"rel_{c}"] >= 0.9, "top", "rest")
        t = {r["level"]: r for r in lift_table(df, "_top", a.boot, rng)}["top"]
        rows.append(dict(t, level=c.removeprefix("c_").replace("_", " ")))
    out["content"] = sorted(rows, key=lambda r: -r["lift"])
    for k in ["length", "distinct", "dialogue"]:
        order = {"length": ["shortest 20%", "short", "middle", "long", "longest 20%"],
                 "distinct": ["most typical 20%", "typical", "middle", "unusual", "most unusual 20%"],
                 "dialogue": ["no dialogue", "has dialogue"]}[k]
        out[k] = sorted(out[k], key=lambda r: order.index(r["level"]))

    tune, ev = FEATURES / "model_tune.json", FEATURES / "model_eval.json"
    if ev.exists():
        k0 = json.loads(ev.read_text())["k0"]
        out["models"] = [dict(name=n, lift=k0[n]["lift"]) for n in ["random", "position_only", "length_only", "clip_only", "readable", "full"]]
    if tune.exists():
        t = json.loads(tune.read_text())["variants"]
        best = max((k for k in t if k != "random"), key=lambda k: t[k]["r_prec"])
        out["best_model"] = dict(name=best, lift=t[best]["lift"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, allow_nan=False))
    print("->", OUT)


if __name__ == "__main__":
    main()
