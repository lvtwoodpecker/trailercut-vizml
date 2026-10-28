"""Compare trailerness model variants on the same film-grouped folds.

Each variant changes one thing on top of the previous best, so the report reads as a ladder:
feature groups (context, relative), learning-to-rank instead of classification, and a few
LightGBM settings. Logos/credits (first/last 1% of each film) are already excluded by model.load.

Headline metric is per-film precision at trailer length (r_prec) and its lift over random;
ap_film is reported next to it.

Writes data/features/model_tune.json (committed)

  python -m trailercut.model_tune [--folds 5]
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from .model import PARAMS, READABLE, feats, load, metrics
from .paths import FEATURES

BASE = READABLE + ["clip"]
SLOW = dict(PARAMS, n_estimators=800, learning_rate=0.03)
SMALL = dict(PARAMS, num_leaves=31, min_child_samples=200)
VARIANTS = {
    "full": (BASE, "clf", PARAMS),
    "full+context": (BASE + ["context"], "clf", PARAMS),
    "full+relative": (BASE + ["relative"], "clf", PARAMS),
    "full+context+relative": (BASE + ["context", "relative"], "clf", PARAMS),
    "rank: full+context+relative": (BASE + ["context", "relative"], "rank", PARAMS),
    "rank, slower learning": (BASE + ["context", "relative"], "rank", SLOW),
    "rank, smaller trees": (BASE + ["context", "relative"], "rank", SMALL),
}


def fit_predict(df, cols, y, tr, te, kind, params):
    if kind == "clf":
        m = lgb.LGBMClassifier(**params, random_state=0).fit(df.iloc[tr][cols], y[tr])
        return m.predict_proba(df.iloc[te][cols])[:, 1]
    # lambdarank: one query per film; rows are already contiguous per film
    sizes = df.iloc[tr].groupby("imdb_id", sort=False).size().values
    p = {k: v for k, v in params.items() if k not in ("subsample", "subsample_freq")}  # bagging breaks query groups
    m = lgb.LGBMRanker(**p, objective="lambdarank", random_state=0).fit(df.iloc[tr][cols], y[tr], group=sizes)
    return m.predict(df.iloc[te][cols])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pcs", type=int, default=64)
    a = ap.parse_args()
    df = load(a.pcs)
    y = df.in_trailer.values
    folds = list(GroupKFold(a.folds).split(df, y, df.imdb_id.values))
    scores = df[["imdb_id", "shot_idx", "shot_len", "in_trailer"]].copy()
    scores["random"] = np.random.default_rng(0).random(len(df))
    for name, (groups, kind, params) in tqdm(VARIANTS.items(), desc="variants"):
        cols = feats(groups)
        p = np.zeros(len(df))
        for tr, te in folds:
            p[te] = fit_predict(df, cols, y, tr, te, kind, params)
        scores[name] = p
    base = metrics(scores, "random")["r_prec"]
    res = {k: metrics(scores, k) for k in ["random", *VARIANTS]}
    for v in res.values():
        v["lift"] = v["r_prec"] / float(y.mean())
    report = dict(n_shots=len(df), n_films=int(df.imdb_id.nunique()), positive_rate=float(y.mean()), variants=res)
    (FEATURES / "model_tune.json").write_text(json.dumps(report, indent=2))
    print(f"{'variant':30s} {'ap_film':>8s} {'r_prec':>7s} {'lift':>5s} {'ap':>6s}")
    for k, v in report["variants"].items():
        print(f"{k:30s} {v['ap_film']:8.3f} {v['r_prec']:7.3f} {v['lift']:5.2f} {v['ap']:6.3f}")
    print(f"(random r_prec {base:.3f})")


if __name__ == "__main__":
    main()
