"""Score the aligner against hand labels and pick the match threshold.

Reads labels/gold_*.json (from label_tool). Labels index shots of the trailer that was labeled
(labels/label_trailers.csv records its youtube_id), so load_gold drops a film's labels once its
trailer in trailer_status.csv differs. For a threshold t, a trailer shot is "predicted matched"
if score >= t. Then
  precision = predicted-matched shots whose top-1 is the labeled film shot
  recall    = shots with a labeled source shot that are predicted matched with the right top-1
"same scene, other shot" counts as a miss for precision and never toward recall; its count shows up
in label_mix, since for story position it is close to correct. With two labelers, Cohen's kappa on the overlap.

  python -m trailercut.label_eval
Writes figures/alignment_pr.png and data/features/alignment_eval.json
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .paths import FEATURES, FIGURES, IDS, IDS_DIR, LABELS

STATUS_CSV = IDS_DIR / "trailer_status.csv"


def load_gold():
    frames = []
    for p in sorted(LABELS.glob("gold_*.json")):
        g = json.loads(p.read_text())
        frames.append(pd.DataFrame([dict(key=k, label=v, labeler=g["labeler"]) for k, v in g["labels"].items()]))
    gold = pd.concat(frames) if frames else pd.DataFrame(columns=["key", "label", "labeler"])
    # labels index shots of the trailer that was labeled; drop films whose trailer has since been swapped
    ref = LABELS / "label_trailers.csv"
    if ref.exists() and len(gold):
        labeled = dict(pd.read_csv(ref, dtype=str).values)
        now = dict(pd.read_csv(STATUS_CSV, dtype=str)[["imdb_id", "youtube_id"]].values)
        film = gold.key.str.split(":").str[0]
        keep = film.map(lambda f: labeled.get(f) is None or labeled[f] == now.get(f))
        if (~keep).any():
            print(f"dropping {(~keep).sum()} labels from films whose trailer was swapped: {sorted(set(film[~keep]))}")
        gold = gold[keep]
    return gold


def one_per_shot(gold):
    """One label per trailer shot (first labeler wins on overlap), with imdb_id and shot_idx columns."""
    gold = gold.drop_duplicates("key").copy()
    gold["imdb_id"], gold["shot_idx"] = zip(*[(k.split(":")[0], int(k.split(":")[1])) for k in gold.key])
    return gold


def evaluate(gold, al):
    """Labels vs alignment rows (imdb_id, shot_idx, film_shot, score, blank) -> report, thresholds, P, R, best index."""
    ev = {}
    d = gold.merge(al, on=["imdb_id", "shot_idx"])
    d = d[d.label != "u"]
    d["has_source"] = d.label.str.isdigit()
    d["top1_right"] = d.has_source & (d.label == d.film_shot.astype(str))
    ev["n"] = int(len(d))
    ev["label_mix"] = {k: int(v) for k, v in d.label.where(~d.has_source, "film shot").value_counts().items()}
    ev["top1_acc_when_source"] = float(d[d.has_source].top1_right.mean())

    ts = np.linspace(0.4, 0.95, 111)
    prec, rec = [], []
    for t in ts:
        pm = (d.score >= t) & ~d.blank
        prec.append(d[pm].top1_right.mean() if pm.any() else np.nan)
        rec.append((pm & d.top1_right).sum() / max(d.has_source.sum(), 1))
    prec, rec = np.array(prec), np.array(rec)
    f1 = 2 * prec * rec / (prec + rec)
    best = int(np.nanargmax(f1))
    ok = np.where(prec >= 0.9)[0]
    ev["best_f1"] = dict(threshold=float(ts[best]), precision=float(prec[best]), recall=float(rec[best]), f1=float(f1[best]))
    if len(ok):
        ev["precision_0.9"] = dict(threshold=float(ts[ok[0]]), recall=float(rec[ok[0]]))
    return ev, ts, prec, rec, best


def main():
    gold = load_gold()
    if gold.empty:
        print("no labels/gold_*.json yet; run trailercut.label_tool and label first")
        return
    report = {}
    labelers = gold.labeler.unique()
    if len(labelers) >= 2:
        a, b = (gold[gold.labeler == l].set_index("key").label for l in labelers[:2])
        both = a.index.intersection(b.index)
        report["kappa"] = float(cohen_kappa_score(a[both], b[both])) if len(both) else None
        report["kappa_n"] = int(len(both))
    gold = one_per_shot(gold)
    al = pd.concat([pd.read_csv(FEATURES / "alignment" / f"{i}.csv").assign(imdb_id=i) for i in gold.imdb_id.unique()])
    ev, ts, prec, rec, best = evaluate(gold, al)
    report.update(ev)
    meta = json.loads((FEATURES / "alignment_threshold.json").read_text())
    impostor = meta.get("impostor_threshold", meta["threshold"])  # "threshold" is the cutoff in use, maybe the label one
    report["impostor_threshold"] = impostor

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ts, prec, label="precision")
    ax.plot(ts, rec, label="recall")
    ax.axvline(impostor, ls="--", c="gray", lw=1, label="impostor 5% FPR")
    ax.axvline(ts[best], ls=":", c="k", lw=1, label="best F1")
    ax.set_xlabel("match score threshold")
    ax.set_title(f"Alignment vs hand labels (n={report['n']})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "alignment_pr.png", dpi=150)
    (FEATURES / "alignment_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
