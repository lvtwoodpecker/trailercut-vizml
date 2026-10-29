"""Glass-box trailerness model: an explainable boosting machine (EBM) on the readable features.

An EBM learns one curve per feature and adds them up, so the model is its own explanation: each curve
says how the pick chance moves with that feature, with nothing approximated after the fact (unlike
SHAP on LightGBM). No pairwise interactions, so every effect is a single curve. Same film-grouped
folds and metrics as model.py, compared with LightGBM on the same readable features.

Outputs
  data/features/model_ebm.json   metrics (out of fold) + one curve per feature from a fit on all films
  figures/ebm_shapes.png         the curves of the most important features

  python -m trailercut.model_ebm [--folds 5]
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from .model import READABLE, feats, load, metrics
from .paths import FEATURES, FIGURES

PARAMS = dict(interactions=0, max_bins=64, outer_bags=4, n_jobs=8, random_state=0)
LABEL = {"story_pos": "position in film", "shot_len": "shot length (s)", "visual_change": "visual change",
         "dlg_rate": "words per second",
         "scale_width": "shot width (close to wide)", "dlg_words": "words spoken", "synopsis_pos": "synopsis position"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from model_ebm.json")
    a = ap.parse_args()
    df = load(pcs=2)
    if a.plot_only:
        return plot(json.loads((FEATURES / "model_ebm.json").read_text())["curves"], df)
    cols = feats(READABLE)
    y = df.in_trailer.values
    scores = df[["imdb_id", "shot_idx", "shot_len", "in_trailer"]].copy()
    p = np.zeros(len(df))
    for tr, te in tqdm(list(GroupKFold(a.folds).split(df, y, df.imdb_id.values)), desc="folds"):
        m = ExplainableBoostingClassifier(**PARAMS).fit(df.iloc[tr][cols], y[tr])
        p[te] = m.predict_proba(df.iloc[te][cols])[:, 1]
    scores["ebm"] = p
    res = metrics(scores, "ebm")
    res["lift"] = res["r_prec"] / float(y.mean())

    m = ExplainableBoostingClassifier(**PARAMS).fit(df[cols], y)
    g = m.explain_global().data()
    imp = dict(zip(g["names"], g["scores"]))
    curves = {}
    for k, name in enumerate(m.term_names_):
        d = m.explain_global().data(k)
        curves[name] = dict(x=[float(v) if not isinstance(v, str) else v for v in d["names"]],
                            score=[float(v) for v in d["scores"]], importance=float(imp[name]))
    lgb = json.loads((FEATURES / "model_eval.json").read_text())["k0"]["readable"]["lift"]
    (FEATURES / "model_ebm.json").write_text(json.dumps(dict(metrics=res, lightgbm_readable_lift=lgb, curves=curves), indent=2))
    print(f"EBM lift {res['lift']:.2f} (LightGBM, same features: {lgb:.2f}); ap_film {res['ap_film']:.3f}")

    plot(curves, df)


def plot(curves, df):
    """Most important curves on one shared log scale; x clipped to where 98% of shots fall
    (the outermost bins hold a handful of shots and swing wildly)."""
    top = sorted(curves, key=lambda n: -curves[n]["importance"])[:9]
    fig, axs = plt.subplots(3, 3, figsize=(9, 6.6), sharey=True)
    for ax, name in zip(axs.flat, top):
        c = curves[name]
        x, s = np.array(c["x"], dtype=float), np.exp(c["score"])
        lo, hi = df[name].quantile([0.01, 0.99])
        keep = (x[:-1] >= lo) & (x[:-1] <= hi)
        ax.step(x[:-1][keep], s[keep], where="post", color="#2a78d6", lw=1.8)
        ax.set_yscale("log")
        ax.axhline(1, color="#898781", lw=0.8, ls="--")
        ax.set_title(LABEL.get(name, name.removeprefix("c_").replace("_", " ")), loc="left", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    ticks = [0.7, 0.8, 1, 1.25, 1.5, 2]
    axs.flat[0].set_yticks(ticks, [f"{t}×" for t in ticks])
    axs.flat[0].yaxis.set_minor_formatter(plt.NullFormatter())
    fig.supylabel("odds of making the trailer", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "ebm_shapes.png", dpi=150)


if __name__ == "__main__":
    main()
