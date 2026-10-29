"""Where alignment and shot detection fail: three diagnostics in one figure.

  labels     match-score distributions by hand label (right shot, wrong shot, same scene, not in film)
  language   per-trailer match rate, English vs other spoken languages (Whisper's language guess)
  cuts       keyframe consistency inside a shot (lowest cosine between its 3 keyframes), trailer shots
             vs MovieNet's film shots. A trailer shot whose keyframes disagree much more than film
             shots ever do probably spans a cut TransNetV2 missed.

Writes figures/alignment_diagnostics.png and data/features/alignment_diagnostics.json (committed)

  python -m trailercut.align_diagnostics
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .align import load
from .label_eval import load_gold
from .paths import CACHE, CLIP, FEATURES, FIGURES


def consistency(E):
    """[n, 3, d] normalized -> [n]: lowest cosine between a shot's keyframes."""
    s = np.einsum("nid,njd->nij", E, E)
    return np.minimum(np.minimum(s[:, 0, 1], s[:, 0, 2]), s[:, 1, 2])


def main():
    threshold = json.loads((FEATURES / "alignment_threshold.json").read_text())["threshold"]
    gold = load_gold().drop_duplicates("key")
    gold = gold[gold.label != "u"]
    rows = []
    for key, lab in zip(gold.key, gold.label):
        i, s = key.split(":")
        r = pd.read_csv(FEATURES / "alignment" / f"{i}.csv").iloc[int(s)]
        if r.blank:
            continue
        kind = ("right shot" if int(lab) == r.film_shot else "wrong shot") if lab.isdigit() else \
               {"s": "same scene, other shot", "n": "not in film"}[lab]
        rows.append(dict(kind=kind, score=r.score))
    lab = pd.DataFrame(rows)

    lang = []
    for p in sorted((CACHE / "trailer_asr").glob("*.json")):
        al = pd.read_csv(FEATURES / "alignment" / f"{p.stem}.csv")
        segs = json.loads(p.read_text())
        l = segs["language"] if segs["segments"] else "no speech"
        lang.append(dict(imdb_id=p.stem, language=l, match=float(al.matched[~al.blank].mean())))
    lang = pd.DataFrame(lang)
    lang["group"] = np.where(lang.language == "en", "English", np.where(lang.language == "no speech", "no speech", "other language"))

    ids = sorted(p.stem for p in (CLIP / "trailer").glob("*.npy"))
    t = np.concatenate([consistency(load("trailer", i).numpy()) for i in ids])
    f = np.concatenate([consistency(load("film", i).numpy()) for i in ids[::5]])  # a fifth of the films is plenty
    floor = float(np.quantile(f, 0.01))  # 1% of real (MovieNet) shots are less consistent than this

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    bins = np.linspace(0.3, 1, 36)
    for kind, color in [("right shot", "#2a78d6"), ("wrong shot", "#e34948"), ("same scene, other shot", "#eda100"), ("not in film", "#898781")]:
        x = lab[lab.kind == kind].score
        axs[0].hist(x, bins=bins, histtype="step", lw=1.8, color=color, density=True, label=f"{kind} ({len(x)})")
    axs[0].axvline(threshold, color="#52514e", ls="--", lw=1)
    axs[0].set_title("Match score by hand label", loc="left", fontsize=10)
    axs[0].legend(frameon=False, fontsize=7)
    axs[0].set_yticks([])
    order = ["English", "other language", "no speech"]
    data = [lang[lang.group == g].match for g in order]
    axs[1].boxplot(data, widths=0.5, showfliers=False)
    for k, d in enumerate(data):
        axs[1].scatter(np.random.default_rng(k).normal(k + 1, 0.06, len(d)), d, s=6, alpha=0.4, color="#2a78d6")
    axs[1].set_xticks(range(1, 4), [f"{g}\n({len(d)})" for g, d in zip(order, data)], fontsize=8)
    axs[1].set_title("Match rate by trailer language", loc="left", fontsize=10)
    b2 = np.linspace(0, 1, 41)
    axs[2].hist(f, bins=b2, histtype="step", lw=1.8, color="#898781", density=True, label="film shots (MovieNet)")
    axs[2].hist(t, bins=b2, histtype="step", lw=1.8, color="#2a78d6", density=True, label="trailer shots (TransNetV2)")
    axs[2].axvline(floor, color="#52514e", ls="--", lw=1)
    axs[2].set_title("Keyframe consistency inside a shot", loc="left", fontsize=10)
    axs[2].legend(frameon=False, fontsize=7)
    axs[2].set_yticks([])
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "alignment_diagnostics.png", dpi=150)

    out = dict(threshold=threshold,
               labels={k: dict(n=int(len(g)), median_score=float(g.score.median()), above_cutoff=float((g.score >= threshold).mean()))
                       for k, g in lab.groupby("kind")},
               language={g: dict(trailers=int((lang.group == g).sum()), median_match=float(lang[lang.group == g].match.median())) for g in order},
               cuts=dict(film_floor_1pct=floor, trailer_shots=int(len(t)), trailer_below_floor=float((t < floor).mean())))
    (FEATURES / "alignment_diagnostics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
