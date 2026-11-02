"""Trailer clustering (RQ2) with stability and metadata-association checks.

For all features combined and for each modality alone:
  - k-means for k=2..8 (silhouette) and HDBSCAN
  - bootstrap stability: refit on 80% subsamples, ARI against the full-data labels on the overlap
  - association with genre and decade (NMI), against a permutation null so "agreement" has a baseline

  python -m trailercut.cluster
Writes data/features/clusters.csv, data/features/cluster_report.json, figures/clusters_*.png
"""
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import json

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from .paths import FEATURES, FIGURES
from .trailer_features import GROUPS

RNG = np.random.default_rng(0)
N_BOOT = 50


# def prep(df, cols):
#     X = SimpleImputer(strategy="median").fit_transform(df[cols])
#     return StandardScaler().fit_transform(X)

def prep(df, cols):
    # If cols is already an array/matrix, use it directly
    if isinstance(cols, np.ndarray):
        X_raw = cols
    else:
        X_raw = df[cols].values

    X_imp = SimpleImputer(strategy="median").fit_transform(X_raw)
    return StandardScaler().fit_transform(X_imp)


def kmeans(X, k, seed=0):
    return KMeans(k, n_init=10, random_state=seed).fit_predict(X)


def stability(X, k):
    full = kmeans(X, k)
    aris = []
    for b in range(N_BOOT):
        idx = RNG.choice(len(X), int(0.8 * len(X)), replace=False)
        aris.append(adjusted_rand_score(full[idx], kmeans(X[idx], k, seed=b + 1)))
    return float(np.mean(aris)), float(np.std(aris))


def association(labels, meta, n_perm=200):
    """NMI with metadata vs. the same NMI with shuffled metadata (95th pct)."""
    ok = meta.notna().values
    lab, m = labels[ok], meta[ok].astype(str).values
    nmi = normalized_mutual_info_score(m, lab)
    null = [normalized_mutual_info_score(RNG.permutation(m), lab) for _ in range(n_perm)]
    return dict(nmi=float(nmi), null_p95=float(np.percentile(null, 95)), p=float(np.mean(np.array(null) >= nmi)))


# def analyze(df, cols, name):
#     X = prep(df, cols)
#     sil = {k: float(silhouette_score(X, kmeans(X, k))) for k in range(2, 9)}
#     k = max(sil, key=sil.get)
#     labels = kmeans(X, k)
#     hdb = HDBSCAN(min_cluster_size=max(5, len(X) // 30)).fit_predict(X)
#     stab = stability(X, k)
#     return labels, dict(
#         modality=name, n_features=len(cols), best_k=k, silhouette=sil,
#         stability_ari_mean=stab[0], stability_ari_std=stab[1],
#         hdbscan_n_clusters=int(hdb.max() + 1), hdbscan_noise_frac=float((hdb < 0).mean()),
#         genre=association(labels, df.primary_genre), decade=association(labels, df.decade),
#     )

def analyze(df, cols, name):
    X = prep(df, cols)
    sil = {k: float(silhouette_score(X, kmeans(X, k))) for k in range(2, 9)}
    k = max(sil, key=sil.get)
    labels = kmeans(X, k)
    hdb = HDBSCAN(min_cluster_size=max(5, len(X) // 30)).fit_predict(X)
    stab = stability(X, k)

    return labels, dict(
        modality=name,
        n_features=X.shape[1],
        best_k=k,
        silhouette=sil,
        stability_ari_mean=stab[0],
        stability_ari_std=stab[1],
        hdbscan_n_clusters=int(hdb.max() + 1),
        hdbscan_noise_frac=float((hdb < 0).mean()),
        genre=association(labels, df.primary_genre),
        decade=association(labels, df.decade),
    )

def residualize_features(X, meta_df, alpha=1.0):
    # One-hot encode confounds (drop first to prevent perfect multicollinearity)
    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    confounds = encoder.fit_transform(meta_df[["primary_genre", "decade"]])

    # Fit Ridge regression model to predict feature matrix X from confounds
    model = Ridge(alpha=alpha)
    model.fit(confounds, X)

    # Subtract linear metadata predictions to isolate residuals
    X_pred = model.predict(confounds)

    return X - X_pred


def plot(df, X, labels, path, title):
    import umap
    xy = umap.UMAP(random_state=0).fit_transform(X)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    top = df.primary_genre.value_counts().index[:6]
    g = df.primary_genre.fillna("unknown").where(df.primary_genre.isin(top), "other")
    for ax, (col, vals) in zip(axes, [("cluster", labels.astype(str)), ("primary genre", g), ("decade", df.decade.astype("string").fillna("unknown"))]):
        for v in sorted(pd.unique(np.asarray(vals, dtype=str))):
            msk = np.asarray(vals, dtype=str) == v
            ax.scatter(xy[msk, 0], xy[msk, 1], s=14, alpha=0.8, label=v)
        ax.set_title(f"colored by {col}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7, markerscale=0.8, frameon=False, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    # fig.savefig(path, dpi=150)
    # plt.close(fig)
    return xy


def main():
    df = pd.read_parquet(FEATURES / "trailers.parquet")

    # Standardize column types
    df["decade"] = pd.to_numeric(df["decade"], errors="coerce").astype("int64")
    df["primary_genre"] = df["primary_genre"].astype(str)

    report_baseline = []
    report_residual = []

    all_cols = [c for g in GROUPS.values() for c in g]
    out = df[["imdb_id", "title", "year", "primary_genre"]].copy()

    print("=== RUNNING BASELINE CLUSTERING ===")
    for name, cols in [("all", all_cols), *GROUPS.items()]:
        labels, r = analyze(df, cols, name)
        out[f"cluster_{name}"] = labels
        report_baseline.append(r)
        print(f"{name:9s} k={r['best_k']} sil={r['silhouette'][r['best_k']]:.3f} "
              f"stab={r['stability_ari_mean']:.2f} genreNMI={r['genre']['nmi']:.3f} (null95 {r['genre']['null_p95']:.3f}) "
              f"decadeNMI={r['decade']['nmi']:.3f} (null95 {r['decade']['null_p95']:.3f})")

    print("\n=== RUNNING RESIDUAL CLUSTERING ===")
    for name, cols in [("all", all_cols), *GROUPS.items()]:
        # Residualized prep: prep features first, then subtract genre/decade signal
        X_base = prep(df, cols)
        X_res = residualize_features(X_base, df)

        labels, r = analyze(df, X_res, name)
        out[f"cluster_res_{name}"] = labels
        report_residual.append(r)
        print(f"[RES]  {name:9s} k={r['best_k']} sil={r['silhouette'][r['best_k']]:.3f} "
              f"stab={r['stability_ari_mean']:.2f} genreNMI={r['genre']['nmi']:.3f} "
              f"decadeNMI={r['decade']['nmi']:.3f}")
        
    xy_base = plot(df, prep(df, all_cols), out.cluster_all.values, FIGURES / "clusters_all_umap.png",
              f"Trailers (n={len(df)}), all features, k-means")
    plt.show()

    xy_res = plot(df, residualize_features(prep(df, all_cols), df), out.cluster_res_all.values, FIGURES / "clusters_all_umap_res.png",
                  f"Trailers (n={len(df)}), Residualized (All Features)")
    plt.show()
    
    out["umap_x_base"], out["umap_y_base"] = xy_base[:, 0], xy_base[:, 1]
    out["umap_x_res"], out["umap_y_res"] = xy_res[:, 0], xy_res[:, 1]

    # out.to_csv(FEATURES / "clusters.csv", index=False)
    # (FEATURES / "cluster_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
