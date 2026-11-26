"""RQ1 on a locked test set, and is story position's gain real?

model.py scores every film out-of-fold, but its settings (PARAMS) were picked by model_tune on those
same folds. Here 20% of films are set aside before anything is fit: each model trains once on the
other 80% and is scored once on the test films. (PARAMS still came from tuning on all films; with 9
settings tried and a small spread between them, that leak is small, but it is one.)

Story position: per film, precision at trailer length (the lift's numerator) with and without the
position features, as a paired test over films (t-test, Wilcoxon) and a film-bootstrap interval for
the lift difference. Run on the out-of-fold scores (all films) and on the locked test films.

Writes data/features/model_heldout.json (committed)

  python -m trailercut.model_heldout [--test 0.2] [--boot 2000]
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from .model import PARAMS, feats_for, load, metrics
from .paths import FEATURES

NAMES = ["position_only", "length_only", "clip_only", "readable", "full", "full_minus_position"]


def per_film(df, score):
    """Per film with a used shot: precision of its top-m shots (m = shots the trailer used), and its positive rate."""
    rows = []
    for i, g in df.groupby("imdb_id"):
        m = int(g.in_trailer.sum())
        if m:
            rows.append((i, g.in_trailer.values[np.argsort(-g[score].values)[:m]].mean(), g.in_trailer.mean()))
    return pd.DataFrame(rows, columns=["imdb_id", "prec", "base"]).set_index("imdb_id")


def lift(pf):
    return float(pf.prec.mean() / pf.base.mean())


def paired(df, a, b, boot, rng):
    """Model a vs b, paired by film."""
    pa, pb = per_film(df, a), per_film(df, b)
    d = pa.prec - pb.prec
    films = pa.index.values
    diffs = []
    for _ in range(boot):
        s = rng.choice(len(films), len(films))
        diffs.append(pa.prec.values[s].mean() / pa.base.values[s].mean() - pb.prec.values[s].mean() / pb.base.values[s].mean())
    return dict(films=int(len(d)), lift_a=round(lift(pa), 3), lift_b=round(lift(pb), 3),
                lift_diff=round(lift(pa) - lift(pb), 3), ci=[round(float(np.quantile(diffs, q)), 3) for q in (0.025, 0.975)],
                films_better=int((d > 0).sum()), films_worse=int((d < 0).sum()), films_tied=int((d == 0).sum()),
                t_test_p=float(ttest_rel(pa.prec, pb.prec).pvalue), wilcoxon_p=float(wilcoxon(pa.prec, pb.prec).pvalue))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--pcs", type=int, default=64)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}

    oof = pd.read_parquet(FEATURES / "model_preds.parquet")
    out["position_all_films"] = dict(full_vs_without_position=paired(oof, "full", "full_minus_position", a.boot, rng),
                                     position_only_vs_random=paired(oof, "position_only", "random", a.boot, rng))

    df = load(a.pcs)
    tr, te = next(GroupShuffleSplit(1, test_size=a.test, random_state=0).split(df, groups=df.imdb_id))
    train, test = df.iloc[tr], df.iloc[te].copy()
    test["random"] = rng.random(len(test))
    for name in tqdm(NAMES, desc="locked test"):
        cols = feats_for(name)
        m = lgb.LGBMClassifier(**PARAMS, random_state=0).fit(train[cols], train.in_trailer.values)
        test[name] = m.predict_proba(test[cols])[:, 1]
    out["locked_test"] = dict(train_films=int(train.imdb_id.nunique()), test_films=int(test.imdb_id.nunique()),
                              test_shots=int(len(test)),
                              models={n: dict(metrics(test, n), lift=round(lift(per_film(test, n)), 3)) for n in ["random", *NAMES]})
    out["position_locked_test"] = dict(full_vs_without_position=paired(test, "full", "full_minus_position", a.boot, rng),
                                       position_only_vs_random=paired(test, "position_only", "random", a.boot, rng))
    (FEATURES / "model_heldout.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
