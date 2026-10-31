"""Export what gets picked across all films, for the RQ1 page (viz/rq1.html).

Every number is a lift: the pick rate of a kind of shot divided by the pick rate of the average shot.
Intervals are 95% from resampling films (not shots), so one film can't carry a result.
Uses the same shots as the model (logos/credits at the film's edges excluded, see model.EDGE).

Every row also carries a few example shots that made a trailer (film, trailer moment), which the
page links to the Trailer View.

Writes viz/data/rq1.json (gitignored, rebuilt in under a minute)

  python -m trailercut.rq1_export [--boot 300]
"""
import argparse
import json

import numpy as np
import pandas as pd

from . import concepts
from . import movienet_io as mn
from .model import load
from .paths import FEATURES, LABELS, ROOT

OUT = ROOT / "viz" / "data" / "rq1.json"
EBM_LABEL = {"story_pos": "position in film", "shot_len": "shot length (s)", "visual_change": "visual change",
             "dlg_rate": "words per second", "scale_width": "shot width", "dlg_words": "words spoken"}
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


_times = {}


def trailer_time(imdb_id, shot):
    """When the trailer shows this film shot (its first matched trailer shot), or None."""
    if imdb_id not in _times:
        p = FEATURES / "alignment" / f"{imdb_id}.csv"
        a = pd.read_csv(p, usecols=["t_start", "film_shot", "matched"]) if p.exists() else None
        _times[imdb_id] = {} if a is None else a[a.matched].groupby("film_shot").t_start.min().to_dict()
    return _times[imdb_id].get(shot)


# hand-picked example films per row ("<section>:<level>" -> IMDb IDs), shown before the automatic picks
CURATED = LABELS / "rq1_examples.json"
_curated = json.loads(CURATED.read_text()) if CURATED.exists() else {}


def examples(df, mask, k=3, seed=0, best=None, key=None):
    """A few shots of this kind that made a trailer, from different films, with the trailer moment
    they appear at, so the RQ1 page can link a row to the Trailer View. Random by default; with `best`
    (a column), the shots that score highest on it, so a concept's examples clearly show the concept."""
    pick = df.loc[mask & (df.in_trailer == 1), ["imdb_id", "shot_idx"] + ([best] if best else [])]
    pick = pick.sort_values(best, ascending=False) if best else pick.sample(frac=1, random_state=seed)
    prefer = _curated.get(key, [])
    if prefer:  # hand-picked films first, in their order; each still shows its best shot of this kind
        rank = {i: n for n, i in enumerate(prefer)}
        pick = pick.assign(_r=pick.imdb_id.map(rank).fillna(len(prefer))).sort_values("_r", kind="stable")
    out, seen = [], set()
    for r in pick.itertuples():
        if r.imdb_id in seen:
            continue
        t = trailer_time(r.imdb_id, int(r.shot_idx))
        if t is None:
            continue
        seen.add(r.imdb_id)
        out.append(dict(id=r.imdb_id, t=round(float(t), 1), film=mn.meta(r.imdb_id)["title"]))
        if len(out) == k:
            break
    return out


def with_examples(rows, df, key, section):
    for r in rows:
        r["ex"] = examples(df, df[key] == r["level"], key=f"{section}:{r['level']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    df = load(pcs=2)  # only need the context/relative features here
    out = dict(n_shots=len(df), n_films=int(df.imdb_id.nunique()), positive_rate=float(df.in_trailer.mean()))

    df["pos_bin"] = (np.clip(df.story_pos, 0, 0.9999) * 20).astype(int) / 20 + 0.025
    out["story"] = with_examples(lift_table(df, "pos_bin", a.boot, rng), df, "pos_bin", "story")
    df["scale"] = df[list(SCALES)].idxmax(axis=1).map(SCALES)
    out["scale"] = sorted(with_examples(lift_table(df, "scale", a.boot, rng), df, "scale", "scale"), key=lambda r: list(SCALES.values()).index(r["level"]))
    quint = lambda c, names: pd.cut(df[c], [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=names, include_lowest=True).astype(str)
    df["len_q"] = quint("rel_shot_len", ["shortest 20%", "short", "middle", "long", "longest 20%"])
    out["length"] = with_examples(lift_table(df, "len_q", a.boot, rng), df, "len_q", "length")
    df["distinct_q"] = quint("rel_distinct", ["most typical 20%", "typical", "middle", "unusual", "most unusual 20%"])
    out["distinct"] = with_examples(lift_table(df, "distinct_q", a.boot, rng), df, "distinct_q", "distinct")
    df["dialogue"] = np.where(df.dlg_has > 0, "has dialogue", "no dialogue")
    out["dialogue"] = with_examples(lift_table(df, "dialogue", a.boot, rng), df, "dialogue", "dialogue")
    # concept: shots in the film's top 10% for that concept vs the rest
    rows = []
    for c in concepts.CONTENT:
        df["_top"] = np.where(df[f"rel_{c}"] >= 0.9, "top", "rest")
        t = {r["level"]: r for r in lift_table(df, "_top", a.boot, rng)}["top"]
        rows.append(dict(t, level=c.removeprefix("c_").replace("_", " "), ex=examples(df, df["_top"] == "top", best=c, key="content:" + c.removeprefix("c_").replace("_", " "))))
    out["content"] = sorted(rows, key=lambda r: -r["lift"])
    for k in ["length", "distinct", "dialogue"]:
        order = {"length": ["shortest 20%", "short", "middle", "long", "longest 20%"],
                 "distinct": ["most typical 20%", "typical", "middle", "unusual", "most unusual 20%"],
                 "dialogue": ["no dialogue", "has dialogue"]}[k]
        out[k] = sorted(out[k], key=lambda r: order.index(r["level"]))

    ev = FEATURES / "model_eval.json"
    if ev.exists():
        k0 = json.loads(ev.read_text())["k0"]
        out["models"] = [dict(name=n, lift=k0[n]["lift"]) for n in ["random", "position_only", "length_only", "clip_only", "readable", "full"]]
    ebm = FEATURES / "model_ebm.json"
    if ebm.exists():  # glass-box curves: odds of making the trailer as each feature changes
        e = json.loads(ebm.read_text())
        top = sorted(e["curves"], key=lambda n: -e["curves"][n]["importance"])[:6]
        out["ebm"] = []
        for name in top:
            c = e["curves"][name]
            edges, score = np.array(c["x"], dtype=float), np.exp(c["score"])
            lo, hi = df[name].quantile([0.01, 0.99])
            keep = (edges[:-1] >= lo) & (edges[:-1] <= hi)
            r4 = lambda a: [round(float(v), 4) for v in np.asarray(a)[keep]]
            cur = dict(name=EBM_LABEL.get(name, name.removeprefix("c_").replace("_", " ")), x=r4(edges[:-1]), y=r4(score))
            if "lo" in c:  # band (spread over the outer bags) and shots per bin (support)
                cur.update(lo=r4(np.exp(c["lo"])), hi=r4(np.exp(c["hi"])))
            if len(c.get("n", [])) == len(score):
                cur["n"] = [int(v) for v in np.asarray(c["n"])[keep]]
            out["ebm"].append(cur)
        out["ebm_lift"] = e["metrics"]["lift"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, allow_nan=False))
    print("->", OUT)


if __name__ == "__main__":
    main()
