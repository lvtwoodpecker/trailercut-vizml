"""Figures for the RQ1 model, for the report.

  figures/model_pr.png         precision/recall per model, out of fold, pooled over films
  figures/model_ablation.png   lift lost when one feature group is removed from the full model
  figures/model_tolerance.png  lift at trailer length as the ±k shot tolerance grows

Lift = per-film precision at trailer length over the positive rate (see model_eval.py).

  python -m trailercut.model_figures
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .paths import FEATURES, FIGURES

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
# the models we compare, in a fixed order; baselines grey, the two main models in color
MODELS = [("random", "random", "#c3c2b7"), ("position_only", "story position only", "#898781"),
          ("length_only", "shot length only", "#52514e"), ("clip_only", "CLIP look only", "#1baf7a"),
          ("readable", "named features", "#eb6834"), ("full", "named features + CLIP", "#2a78d6")]
GROUP_LABEL = {"position": "story position", "dialogue": "dialogue", "scale": "shot scale",
               "content": "content concepts", "editing": "editing (length, change)", "clip": "CLIP look"}


def pr_curves(preds):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    base = preds.in_trailer.mean()
    for key, label, color in MODELS:
        p, r, _ = precision_recall_curve(preds.in_trailer, preds[key])
        main = key in ("readable", "full", "clip_only")
        ax.plot(r, p, color=color, lw=2 if main else 1.2, ls="-" if main else "--", label=label)
    ax.set_xlim(0, 1)
    # the curves sit near the positive rate; zoom there (precision spikes at recall ~0 are a handful of shots)
    ax.set_ylim(0, 0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(f"Which film shots end up in the trailer (positive rate {base:.1%})", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_pr.png")
    plt.close(fig)


def ablation(report):
    rate = report["positive_rate"]
    full = report["models"]["full"]["r_prec"] / rate
    drop = {g: d["r_prec"] / rate for g, d in report["ablation_drop"].items()}
    order = sorted(drop, key=drop.get)
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    ax.set_xlim(min(-0.15, min(drop.values()) - 0.1), max(drop.values()) + 0.1)
    ax.barh([GROUP_LABEL[g] for g in order], [drop[g] for g in order],
            color=["#2a78d6" if drop[g] > 0 else "#e34948" for g in order], height=0.6)
    for y, g in enumerate(order):
        ax.text(drop[g] + (0.005 if drop[g] >= 0 else -0.005), y, f"{drop[g]:+.2f}", va="center",
                ha="left" if drop[g] >= 0 else "right", fontsize=8, color="#52514e")
    ax.axvline(0, color="#898781", lw=0.8)
    ax.set_xlabel(f"lift lost when the group is removed (full model: {full:.2f}x)")
    ax.set_title("What each feature group adds", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_ablation.png")
    plt.close(fig)


def tolerance(ev):
    ks = sorted(int(k[1:]) for k in ev)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for key, label, color in MODELS:
        lift = [ev[f"k{k}"][key]["lift"] for k in ks]
        main = key in ("readable", "full", "clip_only")
        ax.plot(ks, lift, marker="o", ms=4, color=color, lw=2 if main else 1.2, ls="-" if main else "--", label=label)
    ax.axhline(1, color="#c3c2b7", lw=0.8)
    ax.set_xticks(ks, [f"±{k}" for k in ks])
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_xlabel("a pick counts if a used shot is within this many shots")
    ax.set_ylabel("lift over random")
    ax.set_title("Misses aren't near-misses: lift falls as the window grows", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "model_tolerance.png")
    plt.close(fig)


def main():
    preds = pd.read_parquet(FEATURES / "model_preds.parquet")
    pr_curves(preds)
    ablation(json.loads((FEATURES / "model_report.json").read_text()))
    tolerance(json.loads((FEATURES / "model_eval.json").read_text()))
    print("figures/model_pr.png, model_ablation.png, model_tolerance.png")


if __name__ == "__main__":
    main()
