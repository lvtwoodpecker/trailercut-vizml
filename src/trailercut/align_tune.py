"""Compare raw and smoothed alignment against hand labels, and pick the settings.

Raw = each trailer shot takes its best film shot if the score clears a threshold. Smoothed = the
whole trailer is matched as a sequence (align.smooth). Both are scored the same way as label_eval:
  precision = predicted matches that hit the labeled source shot
  recall    = labeled source shots that are predicted, and right
  near      = like precision, but within ±3 film shots counts (right scene, wrong shot)
"same scene" and "not in film" labels count as wrong when predicted matched; "unsure" is dropped.

Writes data/features/align_tune.json (committed)

  python -m trailercut.align_tune
"""
import itertools
import json

import numpy as np
import pandas as pd

from .align import BLANK, blankness, load, shot_sim, smooth
from .label_eval import load_gold
from .paths import FEATURES

THRESHOLDS = np.round(np.arange(0.40, 0.86, 0.01), 2)
LAMS, TAUS = [0.02, 0.05, 0.1, 0.2], [3, 10, 30]


def score(pred, gold):
    """pred: film shot or -1 per labeled trailer shot; gold: label strings."""
    src = gold.str.isdigit()
    true = np.where(src, pd.to_numeric(gold, errors="coerce"), -9).astype(int)
    hit = (pred >= 0) & (pred == true)
    near = (pred >= 0) & src & (np.abs(pred - true) <= 3)
    n_pred = (pred >= 0).sum()
    p, r = hit.sum() / max(n_pred, 1), hit.sum() / max(src.sum(), 1)
    return dict(precision=float(p), recall=float(r), f1=float(2 * p * r / max(p + r, 1e-9)),
                near_precision=float(near.sum() / max(n_pred, 1)), n_pred=int(n_pred))


def main():
    gold = load_gold().drop_duplicates("key")
    if gold.empty:
        print("no labels/gold_*.json yet")
        return
    gold = gold[gold.label != "u"]
    gold["imdb_id"] = gold.key.str.split(":").str[0]
    gold["shot_idx"] = gold.key.str.split(":").str[1].astype(int)
    films = {}
    for i, g in gold.groupby("imdb_id"):
        T, F = load("trailer", i), load("film", i)
        films[i] = (shot_sim(T, F).numpy(), (blankness(T) > BLANK).numpy(), g.sort_values("shot_idx"))
    n = sum(len(g) for *_, g in films.values())
    print(f"{n} labeled shots from {len(films)} trailers")

    def run(method, t, lam=None, tau=None):
        preds, labels = [], []
        for S, blank, g in films.values():
            if method == "raw":
                shot = np.where((S.max(1) >= t) & ~blank, S.argmax(1), -1)
            else:
                shot, _ = smooth(S, blank, t, lam=lam, tau=tau)
            preds.append(shot[g.shot_idx.values])
            labels.append(g.label.values)
        return score(np.concatenate(preds), pd.Series(np.concatenate(labels)))

    rows = [dict(method="raw", threshold=float(t), **run("raw", t)) for t in THRESHOLDS]
    rows += [dict(method="smooth", threshold=float(t), lam=lam, tau=tau, **run("smooth", t, lam, tau))
             for t, lam, tau in itertools.product(THRESHOLDS, LAMS, TAUS)]
    df = pd.DataFrame(rows)
    best = {m: df[df.method == m].sort_values("f1").iloc[-1].dropna().to_dict() for m in ["raw", "smooth"]}
    impostor = json.loads((FEATURES / "alignment_threshold.json").read_text())["threshold"]
    at_impostor = df[(df.method == "raw") & (np.isclose(df.threshold, round(impostor, 2)))].iloc[0].to_dict()
    report = dict(n=int(n), trailers=len(films), current=at_impostor, best=best, grid=rows)
    (FEATURES / "align_tune.json").write_text(json.dumps(report, indent=2))
    cols = ["method", "threshold", "lam", "tau", "precision", "recall", "f1", "near_precision"]
    print("current (raw, impostor threshold):", {k: round(v, 3) if isinstance(v, float) else v for k, v in at_impostor.items() if k in cols})
    for m, b in best.items():
        print(f"best {m}:", {k: round(v, 3) if isinstance(v, float) else v for k, v in b.items() if k in cols})


if __name__ == "__main__":
    main()
