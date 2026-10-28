"""Exploration + alignment figures for the project update.

  python -m trailercut.figures
"""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import movienet_io as mn
from .paths import CACHE, CLIP_TAG, FEATURES, FIGURES, IDS

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def alignments():
    return pd.concat([pd.read_csv(p).assign(imdb_id=p.stem) for p in sorted((FEATURES / "alignment").glob("*.csv"))])


def funnel():
    status = list(csv.DictReader(open(IDS.parent / "trailer_status.csv")))
    steps = [
        ("MovieNet films w/ keyframes", len(mn.video_info())),
        ("+ synopsis-shot alignment", 297),
        ("+ trailer URL (core set)", len(mn.core_ids())),
        ("trailer still on YouTube", sum(r["status"] == "ok" for r in status)),
        ("trailer aligned to film", len(list((FEATURES / "alignment").glob("*.csv")))),
    ]
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.barh([s for s, _ in steps][::-1], [n for _, n in steps][::-1], color="#4c78a8")
    for y, (_, n) in enumerate(steps[::-1]):
        ax.text(n + 10, y, str(n), va="center")
    ax.set_xlabel("films")
    ax.set_title("Data funnel")
    fig.tight_layout()
    fig.savefig(FIGURES / "data_funnel.png")
    plt.close(fig)
    return dict(steps)


def scores(al):
    null = np.load(CACHE / f"alignment_null_{CLIP_TAG}.npy")
    thr = json.loads((FEATURES / "alignment_threshold.json").read_text())["threshold"]
    fig, ax = plt.subplots(figsize=(5, 3))
    bins = np.linspace(0.3, 1, 71)
    ax.hist(null, bins, density=True, alpha=0.6, label="impostor (other film)", color="#bbb")
    ax.hist(al[~al.blank].score, bins, density=True, alpha=0.6, label="own film", color="#4c78a8")
    ax.axvline(thr, c="k", ls="--", lw=1, label=f"threshold {thr:.3f}")
    ax.set_xlabel("best CLIP shot similarity")
    ax.set_title("Trailer shot to film: own film vs impostor")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "alignment_scores.png")
    plt.close(fig)


def alignment_plot(al, ids):
    fig, axes = plt.subplots(1, len(ids), figsize=(3.2 * len(ids), 3.2), sharey=True)
    for ax, imdb_id in zip(np.atleast_1d(axes), ids):
        d = al[al.imdb_id == imdb_id]
        m = d[d.matched]
        fs = mn.shots(imdb_id)
        ax.scatter(m.trailer_pos, m.film_story_pos, c=m.score, cmap="viridis", vmin=0.7, vmax=0.95, s=14)
        u = d[~d.matched]
        ax.scatter(u.trailer_pos, np.full(len(u), -0.06), marker="|", c="#c44", s=30, label="no match")
        for y in (0.25, 0.75):
            ax.axhline(y, c="#ddd", lw=0.8, zorder=0)
        ax.set_title(mn.meta(imdb_id)["title"][:28], fontsize=8)
        ax.set_xlabel("position in trailer")
        ax.set_xlim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("position in film (story)")
    np.atleast_1d(axes)[0].set_ylim(-0.1, 1.02)
    fig.tight_layout()
    fig.savefig(FIGURES / "alignment_examples.png")
    plt.close(fig)


def story_depth(al):
    m = al[al.matched]
    per = m.groupby("imdb_id").film_story_pos.agg(final_act=lambda s: (s > 0.75).mean(), depth=lambda s: s.quantile(0.9))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(m.film_story_pos, bins=20, range=(0, 1), color="#4c78a8")
    for x in (0.25, 0.75):
        axes[0].axvline(x, c="k", lw=0.8, ls=":")
    axes[0].set_xlabel("story position of matched film shot")
    axes[0].set_ylabel("trailer shots")
    axes[0].set_title("Where trailer shots come from (pooled)")
    axes[1].hist(per.final_act, bins=15, range=(0, 0.6), color="#f58518")
    axes[1].set_xlabel("share of matched shots from final quarter")
    axes[1].set_ylabel("trailers")
    axes[1].set_title("Final-act share per trailer")
    fig.tight_layout()
    fig.savefig(FIGURES / "story_depth.png")
    plt.close(fig)
    return dict(pooled_final_quarter=float((m.film_story_pos > 0.75).mean()),
                pooled_first_quarter=float((m.film_story_pos < 0.25).mean()),
                median_trailer_final_act_share=float(per.final_act.median()))


def corpus():
    meta = pd.DataFrame([mn.meta(i) for i in mn.core_ids()])
    tr = pd.concat([pd.read_csv(p).assign(imdb_id=p.stem) for p in (FEATURES / "trailer_shots").glob("*.csv")])
    per = tr.groupby("imdb_id").agg(n=("shot_idx", "size"), dur=("t_end", "max"), med=("shot_len", "median"))
    fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))
    (meta.year // 10 * 10).value_counts().sort_index().plot.bar(ax=axes[0], color="#4c78a8")
    axes[0].set_title("Films by decade")
    meta.genres.explode().value_counts().head(12).plot.barh(ax=axes[1], color="#4c78a8")
    axes[1].invert_yaxis()
    axes[1].set_title("Genres (multi-label)")
    axes[2].hist(per.dur, bins=20, color="#4c78a8")
    axes[2].set_title("Trailer length (s)")
    axes[3].hist(per.med, bins=20, color="#4c78a8")
    axes[3].set_title("Median trailer shot length (s)")
    fig.tight_layout()
    fig.savefig(FIGURES / "corpus.png")
    plt.close(fig)
    return dict(trailers=len(per), trailer_len_median=float(per.dur.median()),
                shots_per_trailer_median=float(per.n.median()), shot_len_median=float(per.med.median()))


def main():
    al = alignments()
    out = dict(funnel=funnel(), corpus=corpus())
    scores(al)
    rates = al.groupby("imdb_id").matched.mean().sort_values()
    alignment_plot(al, [rates.index[len(rates) // 4], rates.index[len(rates) // 2], rates.index[3 * len(rates) // 4]])
    out["story_depth"] = story_depth(al)
    out["alignment"] = dict(trailer_shots=int(len(al)), blank=float(al.blank.mean()), matched=float(al.matched.mean()),
                            match_rate_per_trailer_median=float(rates.median()))
    (FEATURES / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
