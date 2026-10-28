"""RQ1 trailerness model: which film shots end up in the trailer?

LightGBM on the film shot table, cross-validated with folds grouped by film (no film is in both train
and test). Every model is scored out-of-fold on the same splits:

  baselines   random (AP = positive rate), position only, shot length only, CLIP only
  readable    position + dialogue + scale + content + editing
  full        readable + CLIP (PCA of the shot embedding, one "other appearance" block)
  ablations   full minus one group at a time

Metrics
  ap          average precision over all held-out shots (pooled)
  ap_film     mean per-film average precision
  r_prec      precision at trailer length: per film, take the top-k shots where k is the number of film
              shots the trailer actually used, then average over films
  ap_lenadj   mean AP within shot-length quintiles. Longer shots give a trailer more footage to sample,
              so a model can look good by ranking long shots first; this score removes that.

SHAP (TreeSHAP, out-of-fold) for the full model is saved per shot, with the CLIP PCs summed into one
"clip" column so the embedding is never shown dimension by dimension.

Outputs
  data/features/model_report.json   metrics per model (committed)
  data/features/model_preds.parquet out-of-fold scores per shot (gitignored)
  data/features/model_shap.parquet  out-of-fold SHAP per shot, readable features + clip (gitignored)

  python -m trailercut.model [--folds 5] [--pcs 64]
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from .paths import CLIP, FEATURES

GROUPS = {
    "position": ["story_pos", "act", "synopsis_pos"],
    "dialogue": ["dlg_words", "dlg_rate", "dlg_has", "dlg_sentiment"],
    "scale": ["scale_ecu", "scale_cu", "scale_ms", "scale_fs", "scale_ls", "scale_width"],
    "content": None,  # every c_* column
    "editing": ["shot_len", "visual_change"],
    "context": ["distinct", "novelty", "recurrence"],  # how the shot looks next to the rest of its film
    "relative": None,  # rel_* columns: within-film percentile of the readable/context features
    "clip": None,  # clip_pc* columns
}
# first/last 1% of each film: studio logos and credits, which match the trailer's own logos and cards
# and would teach the model "the start of the film gets picked"
EDGE = 0.01
RELATIVE = ["shot_len", "visual_change", "dlg_words", "scale_width", "distinct", "novelty", "recurrence"]
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8, verbose=-1)


def context(E):
    """E: one film's normalized shot embeddings [n, d] in shot order."""
    m = E.mean(0)
    distinct = 1 - E @ (m / np.linalg.norm(m))  # unusual for this film
    n = len(E)
    near = np.zeros(n)
    cnt = np.zeros(n)
    for o in (1, 2):  # mean similarity to the 2 shots on each side
        s = (E[o:] * E[:-o]).sum(1)
        near[o:] += s; near[:-o] += s
        cnt[o:] += 1; cnt[:-o] += 1
    novelty = 1 - near / np.maximum(cnt, 1)
    S = E @ E.T
    idx = np.arange(n)
    S[np.abs(idx[:, None] - idx[None, :]) <= 10] = -1
    recurrence = S.max(1)  # does this look come back elsewhere in the film
    return np.stack([distinct, novelty, recurrence], 1)


def load(pcs, edge=EDGE):
    df = pd.read_parquet(FEATURES / "film_shots.parquet")
    GROUPS["content"] = [c for c in df.columns if c.startswith("c_")]
    # mean of the 3 keyframes per shot; PCA is unsupervised so fitting it on all films doesn't leak labels
    embs = [np.load(CLIP / "film" / f"{i}.npy").astype(np.float32).mean(1) for i in df.imdb_id.unique()]
    embs = [e / np.linalg.norm(e, axis=1, keepdims=True) for e in embs]
    emb = np.concatenate(embs)
    assert len(emb) == len(df), "film_shots.parquet is out of date with the film embeddings"
    ctx = np.concatenate([context(e) for e in embs])
    df[GROUPS["context"]] = ctx.astype(np.float32)
    rel = RELATIVE + GROUPS["content"]
    GROUPS["relative"] = [f"rel_{c}" for c in rel]
    df[GROUPS["relative"]] = df.groupby("imdb_id")[rel].rank(pct=True).astype(np.float32).values
    pc = PCA(pcs, random_state=0).fit_transform(emb)
    GROUPS["clip"] = [f"clip_pc{k}" for k in range(pcs)]
    df = pd.concat([df, pd.DataFrame(pc.astype(np.float32), columns=GROUPS["clip"])], axis=1)
    return df[(df.story_pos >= edge) & (df.story_pos <= 1 - edge)].reset_index(drop=True)


def feats(groups):
    return [c for g in groups for c in GROUPS[g]]


READABLE = ["position", "dialogue", "scale", "content", "editing"]
MODELS = {
    "position_only": ["position"],
    "length_only": [None],  # shot_len alone, see feats_for
    "clip_only": ["clip"],
    "readable": READABLE,
    "full": READABLE + ["clip"],
    **{f"full_minus_{g}": [h for h in READABLE + ["clip"] if h != g] for g in READABLE + ["clip"]},
}


def feats_for(name):
    return ["shot_len"] if name == "length_only" else feats(MODELS[name])


def r_precision(df, score):
    out = []
    for _, g in df.groupby("imdb_id"):
        k = int(g.in_trailer.sum())
        if k:
            out.append(g.in_trailer.values[np.argsort(-g[score].values)[:k]].mean())
    return float(np.mean(out))


def metrics(df, score):
    per_film = [average_precision_score(g.in_trailer, g[score]) for _, g in df.groupby("imdb_id") if g.in_trailer.any()]
    q = pd.qcut(df.shot_len, 5, labels=False, duplicates="drop")
    lenadj = [average_precision_score(g.in_trailer, g[score]) for _, g in df.groupby(q) if g.in_trailer.any()]
    return dict(ap=float(average_precision_score(df.in_trailer, df[score])), ap_film=float(np.mean(per_film)),
                r_prec=r_precision(df, score), ap_lenadj=float(np.mean(lenadj)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pcs", type=int, default=64)
    a = ap.parse_args()

    df = load(a.pcs)
    y, groups = df.in_trailer.values, df.imdb_id.values
    folds = list(GroupKFold(a.folds).split(df, y, groups))
    print(f"{len(df)} shots, {df.imdb_id.nunique()} films, positive rate {y.mean():.4f}")

    preds = df[["imdb_id", "shot_idx", "shot_len", "in_trailer"]].copy()
    rng = np.random.default_rng(0)
    preds["random"] = rng.random(len(df))
    shap_cols = feats(READABLE)
    shap = np.zeros((len(df), len(shap_cols) + 2), dtype=np.float32)  # + clip + bias
    for name in tqdm(MODELS, desc="models"):
        cols = feats_for(name)
        p = np.zeros(len(df))
        for tr, te in folds:
            m = lgb.LGBMClassifier(**PARAMS, random_state=0).fit(df.iloc[tr][cols], y[tr])
            p[te] = m.predict_proba(df.iloc[te][cols])[:, 1]
            if name == "full":
                c = m.predict(df.iloc[te][cols], pred_contrib=True)  # log-odds, last col = bias
                n_read = len(shap_cols)
                shap[te, :n_read] = c[:, :n_read]
                shap[te, n_read] = c[:, n_read:-1].sum(1)
                shap[te, -1] = c[:, -1]
        preds[name] = p

    report = dict(n_shots=len(df), n_films=int(df.imdb_id.nunique()), positive_rate=float(y.mean()),
                  folds=a.folds, clip_pcs=a.pcs, groups={g: c for g, c in GROUPS.items() if g != "clip"},
                  models={k: metrics(preds, k) for k in ["random", *MODELS]})
    full = report["models"]["full"]
    report["ablation_drop"] = {g: {k: full[k] - report["models"][f"full_minus_{g}"][k] for k in full}
                               for g in READABLE + ["clip"]}
    gs = pd.DataFrame(shap[:, :-1], columns=shap_cols + ["clip"])
    report["shap_mean_abs_by_group"] = {g: float(gs[GROUPS[g] if g != "clip" else ["clip"]].sum(1).abs().mean())
                                        for g in READABLE + ["clip"]}
    report["shap_mean_abs_top"] = gs.abs().mean().sort_values(ascending=False).head(15).round(4).to_dict()

    (FEATURES / "model_report.json").write_text(json.dumps(report, indent=2))
    preds.to_parquet(FEATURES / "model_preds.parquet", index=False)
    pd.concat([df[["imdb_id", "shot_idx"]], pd.DataFrame(shap, columns=shap_cols + ["clip", "bias"])], axis=1) \
        .to_parquet(FEATURES / "model_shap.parquet", index=False)

    print(f"{'model':22s} {'ap':>6s} {'ap_film':>8s} {'r_prec':>7s} {'ap_lenadj':>9s}")
    for k, v in report["models"].items():
        print(f"{k:22s} {v['ap']:6.3f} {v['ap_film']:8.3f} {v['r_prec']:7.3f} {v['ap_lenadj']:9.3f}")
    print("SHAP mean |.| by group:", {k: round(v, 3) for k, v in report["shap_mean_abs_by_group"].items()})


if __name__ == "__main__":
    main()
