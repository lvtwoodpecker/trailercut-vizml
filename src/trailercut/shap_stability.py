"""Are the Trailer View's reasons stable? Retrain the full model on different films, compare the reasons.

For each of the 5 grouped folds (same as model.py), train R models on different 80% subsets of the
training films (and seeds), then explain the same held-out shots with each. The reasons are the 6
SHAP groups the Trailer View shows (story position, dialogue, shot scale, content, editing, CLIP),
ranked by |value|; the panel shows the top 3. Per pair of models, on the same shot:
  top1        same strongest reason
  top1_sign   same strongest reason, pushing the same way
  top3        same top-3 set (order ignored)
  jaccard     overlap of the two top-3 sets
  spearman    rank agreement over all 6 groups
Chance = the same numbers for two random rankings. Global: each model's mean |SHAP| ranking of the
groups, compared across all models (Kendall's tau).

Shots explained: up to 3,000 used shots and 3,000 others per fold. Each fold is saved as it finishes
(data/cache/shap_stability/), so a rerun picks up where it stopped.

Writes data/features/shap_stability.json

  python -m trailercut.shap_stability [--models 3]
"""
import argparse
import itertools
import json

import lightgbm as lgb
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from .model import GROUPS, PARAMS, READABLE, feats, load
from .paths import CACHE, FEATURES
from .viewer_export import GROUP_ORDER

OUT = CACHE / "shap_stability"
PER_CLASS = 3000


def grouped(contrib, cols):
    """pred_contrib output (last column = bias) -> [n, 6] in GROUP_ORDER."""
    idx = {c: k for k, c in enumerate(cols)}
    return np.stack([contrib[:, [idx[c] for c in GROUPS[g]]].sum(1) for g in GROUP_ORDER], 1)


def fold(df, cols, tr, te, n_models, seed):
    rng = np.random.default_rng(seed)
    y = df.in_trailer.values
    te_pos, te_neg = te[y[te] == 1], te[y[te] == 0]
    ev = np.concatenate([rng.choice(te_pos, min(PER_CLASS, len(te_pos)), replace=False),
                         rng.choice(te_neg, min(PER_CLASS, len(te_neg)), replace=False)])
    films = df.imdb_id.values[tr]
    uniq = np.unique(films)
    out = []
    for r in range(n_models):
        keep = np.isin(films, rng.choice(uniq, int(0.8 * len(uniq)), replace=False))
        m = lgb.LGBMClassifier(**PARAMS, random_state=seed * 100 + r).fit(df.iloc[tr[keep]][cols], y[tr[keep]])
        out.append(grouped(m.predict(df.iloc[ev][cols], pred_contrib=True), cols))
    return np.stack(out), y[ev]


def pair_stats(a, b):
    """a, b: [n, 6] grouped SHAP for the same shots from two models."""
    ra, rb = np.argsort(-np.abs(a), 1), np.argsort(-np.abs(b), 1)
    t1 = ra[:, 0] == rb[:, 0]
    sign = np.sign(a[np.arange(len(a)), ra[:, 0]]) == np.sign(b[np.arange(len(b)), ra[:, 0]])
    sa, sb = np.sort(ra[:, :3], 1), np.sort(rb[:, :3], 1)
    inter = np.array([len(set(x) & set(z)) for x, z in zip(sa, sb)])
    rho = np.array([spearmanr(np.abs(x), np.abs(z))[0] for x, z in zip(a, b)])
    return dict(top1=t1.mean(), top1_sign=(t1 & sign).mean(), top3=(inter == 3).mean(),
                jaccard=(inter / (6 - inter)).mean(), spearman=np.nanmean(rho))


def summarize(runs):
    """runs: list of (shap [R, n, 6], y [n])."""
    rows = {"all": [], "used": [], "other": []}
    for s, y in runs:
        for i, j in itertools.combinations(range(len(s)), 2):
            rows["all"].append(pair_stats(s[i], s[j]))
            rows["used"].append(pair_stats(s[i][y == 1], s[j][y == 1]))
            rows["other"].append(pair_stats(s[i][y == 0], s[j][y == 0]))
    out = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in rows.items()}
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(20000, 6)), rng.normal(size=(20000, 6))
    out["chance"] = {k: float(v) for k, v in pair_stats(a, b).items()}
    glob = [np.abs(m).mean(0) for s, _ in runs for m in s]
    out["global_rank"] = dict(
        kendall_tau_mean=float(np.mean([kendalltau(x, z)[0] for x, z in itertools.combinations(glob, 2)])),
        order_per_model=[[GROUP_ORDER[k] for k in np.argsort(-g)] for g in glob])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=int, default=3, help="models per fold, each on 80%% of the training films")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    df = load(64)
    cols = feats(READABLE + ["clip"])
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    folds = list(GroupKFold(a.folds).split(df, df.in_trailer.values, df.imdb_id.values))
    for k, (tr, te) in enumerate(tqdm(folds, desc="folds")):
        path = OUT / f"fold{k}_m{a.models}.npz"
        if not path.exists():
            s, y = fold(df, cols, tr, te, a.models, seed=k)
            np.savez(path, shap=s, y=y)
        z = np.load(path)
        runs.append((z["shap"], z["y"]))

    res = summarize(runs)
    res.update(models_per_fold=a.models, folds=a.folds, groups=GROUP_ORDER)
    (FEATURES / "shap_stability.json").write_text(json.dumps(res, indent=2))
    print(f"{'':8s} {'top1':>6s} {'+sign':>6s} {'top3':>6s} {'jacc':>6s} {'rho':>6s}")
    for k in ["all", "used", "other", "chance"]:
        v = res[k]
        print(f"{k:8s} {v['top1']:6.2f} {v['top1_sign']:6.2f} {v['top3']:6.2f} {v['jaccard']:6.2f} {v['spearman']:6.2f}")
    print(f"global group ranking, Kendall tau between models: {res['global_rank']['kendall_tau_mean']:.2f}")


if __name__ == "__main__":
    main()
