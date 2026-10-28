"""Score the trailerness model with a tolerance of ±k film shots.

The aligner often lands one or two shots away from the true source (shot/reverse-shot, continuous
coverage), and at shot level that counts as a full miss. Here a prediction is a hit if a used shot
is within ±k shot indices. Random gets the same tolerance, so the lift over random stays fair:
with k=3 a random shot is ~7x more likely to be "near" a used shot too.

Uses the out-of-fold scores saved by model.py, so nothing is retrained.

Metrics per model and k (k=0 is the exact-shot score from model.py)
  base_rate   share of shots within ±k of a used shot (what random scores as precision)
  ap          average precision against the ±k label, pooled
  prec_at_k   per film, precision of the top-m shots (m = number of used shots), averaged
  recall_at_k per film, share of used shots with a top-m shot within ±k, averaged
  lift        prec_at_k / base_rate

Writes data/features/model_eval.json (committed)

  python -m trailercut.model_eval [--k 0 1 3 5]
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .paths import FEATURES

MODELS = ["random", "position_only", "length_only", "clip_only", "readable", "full"]


def near(y, k):
    """1 where a positive is within ±k indices."""
    return (np.convolve(y, np.ones(2 * k + 1), mode="same") > 0).astype(int) if k else y


def film_scores(y, s, k):
    m = int(y.sum())
    top = np.argsort(-s)[:m]
    y_tol = near(y, k)
    pos = np.flatnonzero(y)
    hit = np.abs(pos[:, None] - top[None, :]).min(1) <= k
    return y_tol[top].mean(), hit.mean()


def evaluate(preds, model, k):
    y_tol, prec, rec = [], [], []
    for _, g in preds.groupby("imdb_id", sort=False):
        y = g.in_trailer.values
        y_tol.append(near(y, k))
        if y.any():
            p, r = film_scores(y, g[model].values, k)
            prec.append(p)
            rec.append(r)
    y_tol = np.concatenate(y_tol)
    base = float(y_tol.mean())
    return dict(base_rate=base, ap=float(average_precision_score(y_tol, preds[model])),
                prec_at_k=float(np.mean(prec)), recall_at_k=float(np.mean(rec)),
                lift=float(np.mean(prec) / base))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[0, 1, 3, 5])
    a = ap.parse_args()
    preds = pd.read_parquet(FEATURES / "model_preds.parquet")
    preds = preds.sort_values(["imdb_id", "shot_idx"], kind="stable")
    out = {f"k{k}": {m: evaluate(preds, m, k) for m in MODELS} for k in a.k}
    (FEATURES / "model_eval.json").write_text(json.dumps(out, indent=2))
    for k in a.k:
        print(f"\n±{k} shots")
        print(f"{'model':15s} {'base':>6s} {'ap':>6s} {'prec':>6s} {'recall':>7s} {'lift':>5s}")
        for m, v in out[f"k{k}"].items():
            print(f"{m:15s} {v['base_rate']:6.3f} {v['ap']:6.3f} {v['prec_at_k']:6.3f} {v['recall_at_k']:7.3f} {v['lift']:5.2f}")


if __name__ == "__main__":
    main()
