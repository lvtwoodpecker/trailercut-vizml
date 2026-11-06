"""Where the RQ1 model misses: lift per slice of films (genre, decade, film and trailer size, subtitles).

Lift in a slice = mean per-film precision at trailer length / the films' mean positive rate, the same
measure as model_eval.py at k=0 (random scores 1x in every slice). Each slice is compared with the rest of the films:
  ci        95% bootstrap interval over films (2,000 resamples)
  p         permutation test on the lift difference, films shuffled between slice and rest
  q         Benjamini-Hochberg over all slices; q < 0.05 = a real gap, not noise
Genre slices use every genre a film has (IMDb lists genres alphabetically, so the first one is
not the main genre). Random gets the same treatment as a check: its lift should sit near 1x.

Writes data/features/error_slices.json and figures/error_slices.png

  python -m trailercut.error_slices [--model full]
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FEATURES, FIGURES
from .style import AXIS, INK2, MATCHED, MUTED, NEG

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
MIN_FILMS = 25


def per_film(preds, model):
    """One row per film: top-m precision, positive rate, size."""
    rows = []
    for i, g in preds.groupby("imdb_id", sort=False):
        y = g.in_trailer.values
        m = int(y.sum())
        if m:
            rows.append(dict(imdb_id=i, prec=y[np.argsort(-g[model].values)[:m]].mean(), base=y.mean(),
                             n_shots=len(y), n_used=m))
    return pd.DataFrame(rows)


def lift(prec, base, w):
    """Mean top-m precision over the films' mean positive rate (w = film weights)."""
    return prec.mean() / np.average(base, weights=w)


def slices(films):
    """name -> boolean mask over films."""
    out = {}
    genres = films.genres.fillna("").str.split("|")
    for g in sorted({x for gs in genres for x in gs if x}):
        out[("genre", g)] = genres.apply(lambda gs: g in gs).values
    dec = films.decade.astype(float)
    for lo, hi, name in [(0, 1970, "before 1970"), (1970, 1990, "1970s–80s"), (1990, 2000, "1990s"),
                         (2000, 2010, "2000s"), (2010, 3000, "2010s")]:
        out[("decade", name)] = ((dec >= lo) & (dec < hi)).values
    out[("decade", "unknown")] = dec.isna().values
    for col, label in [("n_shots", "film shots"), ("n_used", "trailer shots matched")]:
        q = pd.qcut(films[col], 4, labels=False)
        for k, name in enumerate(["fewest", "fewer", "more", "most"]):
            out[(label, f"{name} quarter")] = (q == k).values
    out[("subtitles", "has")] = films.subs.values
    out[("subtitles", "none")] = ~films.subs.values
    out = {k: np.asarray(v, dtype=bool) for k, v in out.items()}
    return {k: v for k, v in out.items() if MIN_FILMS <= v.sum() <= len(films) - MIN_FILMS}


def test(prec, base, w, mask, rng, n=2000):
    """Lift in the slice, bootstrap CI, permutation p for slice vs rest."""
    p_in, b_in, w_in = prec[mask], base[mask], w[mask]
    obs = lift(p_in, b_in, w_in) - lift(prec[~mask], base[~mask], w[~mask])
    idx = rng.integers(0, len(p_in), (n, len(p_in)))
    boot = p_in[idx].mean(1) / ((b_in * w_in)[idx].sum(1) / w_in[idx].sum(1))
    k = int(mask.sum())
    null = np.empty(n)
    for j in range(n):
        s = rng.permutation(len(prec))
        a, b = s[:k], s[k:]
        null[j] = lift(prec[a], base[a], w[a]) - lift(prec[b], base[b], w[b])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)
    return lift(p_in, b_in, w_in), *np.percentile(boot, [2.5, 97.5]), obs, p


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def plot(res, overall, path):
    # ordered slices keep their order; genres sort by lift
    res = res.assign(o=np.where(res.kind == "genre", res.lift, np.arange(len(res)) * -1.0))
    kinds = ["decade", "film shots", "trailer shots matched", "subtitles", "genre"]
    res = res.assign(k=res.kind.map(kinds.index)).sort_values(["k", "o"], ascending=[False, True])
    fig, ax = plt.subplots(figsize=(6.2, 0.19 * len(res) + 0.8))
    y = np.arange(len(res))
    for yi, (_, r) in zip(y, res.iterrows()):
        color = (MATCHED if r["diff"] > 0 else NEG) if r.q < 0.05 else MUTED
        ax.plot([r.lo, r.hi], [yi, yi], color=color, lw=1.6, solid_capstyle="round")
        ax.plot(r.lift, yi, "o", color=color, ms=4.5, mec="white", mew=1)
    ax.axvline(overall, color=AXIS, lw=1)
    ax.text(overall, len(res) - 0.3, f"all films {overall:.2f}×", fontsize=7, color=INK2, ha="center", va="bottom")
    ax.set_yticks(y, [f"{k}: {s} ({n})" for k, s, n in zip(res.kind, res.slice, res.n)], fontsize=7)
    ax.set_xlabel("lift at trailer length (× random)")
    ax.set_title("blue / red: above / below the other films (q < 0.05); grey: within noise",
                 fontsize=7, color=INK2, loc="left")
    ax.set_ylim(-0.7, len(res) + 0.3)
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="full")
    a = ap.parse_args()

    preds = pd.read_parquet(FEATURES / "model_preds.parquet")
    films = per_film(preds, a.model)
    rand = per_film(preds, "random")
    meta = pd.read_parquet(FEATURES / "trailers.parquet")[["imdb_id", "title", "genres", "decade"]]
    subs = pd.read_parquet(FEATURES / "film_shots.parquet", columns=["imdb_id", "dlg_has"])
    subs = subs.groupby("imdb_id").dlg_has.apply(lambda s: s.notna().any())
    films = films.merge(meta, on="imdb_id", how="left")
    films["subs"] = films.imdb_id.map(subs).fillna(False).astype(bool)

    rng = np.random.default_rng(0)
    w = np.ones(len(films))  # films count equally, as in model_eval
    prec, base, rprec = films.prec.values, films.base.values, rand.set_index("imdb_id").loc[films.imdb_id].prec.values
    rows = []
    for (kind, name), mask in slices(films).items():
        l, lo, hi, diff, p = test(prec, base, w, mask, rng)
        rows.append(dict(kind=kind, slice=name, n=int(mask.sum()), lift=l, lo=lo, hi=hi, diff=diff, p=p,
                         random_lift=lift(rprec[mask], base[mask], w[mask])))
    res = pd.DataFrame(rows)
    res["q"] = bh(res.p)
    overall = lift(prec, base, w)

    out = dict(model=a.model, n_films=len(films), overall_lift=overall, slices=res.round(4).to_dict("records"))
    (FEATURES / "error_slices.json").write_text(json.dumps(out, indent=2))
    plot(res, overall, FIGURES / "error_slices.png")
    print(f"{a.model}: {len(films)} films, lift {overall:.2f}×; slices with q < 0.05:")
    print(res[res.q < 0.05].sort_values("diff")[["kind", "slice", "n", "lift", "lo", "hi", "q"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
