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
import umap

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import chi2_contingency
from scipy.optimize import linear_sum_assignment

from .paths import FEATURES, FIGURES, IDS
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

def compute_umap_2d(X, seed=42):
    """Deterministic 2D UMAP projection for a given feature matrix X."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(X)


def kmeans(X, k, seed=0):
    return KMeans(k, n_init=10, random_state=seed).fit_predict(X)


# def stability(X, k):
#     full = kmeans(X, k)
#     aris = []
#     for b in range(N_BOOT):
#         idx = RNG.choice(len(X), int(0.8 * len(X)), replace=False)
#         aris.append(adjusted_rand_score(full[idx], kmeans(X[idx], k, seed=b + 1)))
#     return float(np.mean(aris)), float(np.std(aris))


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

def cramers_v(labels_a, labels_b):
    """Bias-corrected Cramér's V for categorical/cluster independence (0 = orthogonal, 1 = identical)."""
    ct = pd.crosstab(labels_a, labels_b).values
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.sum()
    r, k = ct.shape
    phi2 = max(0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    v = np.sqrt(phi2 / denom) if denom > 0 else 0.0
    return float(v), float(p)


def cross_modality_agreement(out_df, prefix="cluster_res_", space_label="res"):
    """Computes pairwise ARI, NMI, and Cramér's V across all modality clusterings."""
    names = ["all", *GROUPS.keys()]
    n = len(names)
    ari_mat = np.eye(n)
    nmi_mat = np.eye(n)
    cv_mat = np.eye(n)
    results = {}

    for i in range(n):
        for j in range(i + 1, n):
            a = out_df[f"{prefix}{names[i]}"].values
            b = out_df[f"{prefix}{names[j]}"].values

            ari = float(adjusted_rand_score(a, b))
            nmi = float(normalized_mutual_info_score(a, b))
            cv, p_val = cramers_v(a, b)

            ari_mat[i, j] = ari_mat[j, i] = ari
            nmi_mat[i, j] = nmi_mat[j, i] = nmi
            cv_mat[i, j] = cv_mat[j, i] = cv

            results[f"{names[i]}_vs_{names[j]}"] = {
                "ari": round(ari, 4),
                "nmi": round(nmi, 4),
                "cramers_v": round(cv, 4),
                "chi2_p": round(p_val, 4),
            }

    print(f"\n=== CROSS-MODALITY INDEPENDENCE ({space_label.upper()}): ARI MATRIX ===")
    print(pd.DataFrame(ari_mat, index=names, columns=names).round(3).to_string())

    print(f"\n=== CROSS-MODALITY INDEPENDENCE ({space_label.upper()}): CRAMÉR'S V MATRIX ===")
    print(pd.DataFrame(cv_mat, index=names, columns=names).round(3).to_string())

    return {
        "modalities": names,
        "ari_matrix": np.round(ari_mat, 4).tolist(),
        "nmi_matrix": np.round(nmi_mat, 4).tolist(),
        "cramers_v_matrix": np.round(cv_mat, 4).tolist(),
        "pairs": results,
    }

def jaccard_cluster_stability(full_labels, sub_idx, sub_labels, k):
    """Per-cluster best-match Jaccard similarity between full clustering (restricted to sub_idx) and subsample clustering."""
    full_sub = full_labels[sub_idx]
    # Cost matrix: negative Jaccard between full cluster i and subsample cluster j
    jaccard_mat = np.zeros((k, k))
    for i in range(k):
        mask_f = full_sub == i
        if not mask_f.any():
            continue
        for j in range(k):
            mask_s = sub_labels == j
            union = np.logical_or(mask_f, mask_s).sum()
            if union > 0:
                jaccard_mat[i, j] = np.logical_and(mask_f, mask_s).sum() / union

    row_ind, col_ind = linear_sum_assignment(-jaccard_mat)
    per_cluster = np.zeros(k)
    per_cluster[row_ind] = jaccard_mat[row_ind, col_ind]
    return per_cluster

def stability(X, k):
    """Bootstrap ARI + per-cluster and overall best-match Jaccard stability on 80% subsamples."""
    full = kmeans(X, k)
    aris = []
    jaccards = np.zeros((N_BOOT, k))

    for b in range(N_BOOT):
        idx = RNG.choice(len(X), int(0.8 * len(X)), replace=False)
        sub_labels = kmeans(X[idx], k, seed=b + 1)
        aris.append(adjusted_rand_score(full[idx], sub_labels))
        jaccards[b] = jaccard_cluster_stability(full, idx, sub_labels, k)

    per_cluster_mean = np.round(jaccards.mean(axis=0), 4).tolist()
    per_cluster_std = np.round(jaccards.std(axis=0), 4).tolist()

    return {
        "ari_mean": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "jaccard_mean": float(jaccards.mean()),
        "jaccard_std": float(jaccards.mean(axis=1).std()),
        "jaccard_per_cluster_mean": per_cluster_mean,
        "jaccard_per_cluster_std": per_cluster_std,
    }

def compare_baseline_vs_residual(labels_base, labels_res, df, n_perm=500):
    """Permutation tests comparing baseline vs. residualized clusterings:
    1. Structural preservation: ARI and NMI between baseline and residual labels vs. shuffled null.
    2. Confound removal: drop in genre/decade NMI after residualization vs. random label-swap null.
    """
    obs_ari = float(adjusted_rand_score(labels_base, labels_res))
    obs_nmi = float(normalized_mutual_info_score(labels_base, labels_res))

    null_ari = np.empty(n_perm)
    null_nmi = np.empty(n_perm)
    for i in range(n_perm):
        perm = RNG.permutation(labels_res)
        null_ari[i] = adjusted_rand_score(labels_base, perm)
        null_nmi[i] = normalized_mutual_info_score(labels_base, perm)

    meta_shifts = {}
    for col in ["primary_genre", "decade"]:
        ok = df[col].notna().values
        m = df.loc[ok, col].astype(str).values
        lb, lr = labels_base[ok], labels_res[ok]

        nmi_b = float(normalized_mutual_info_score(m, lb))
        nmi_r = float(normalized_mutual_info_score(m, lr))
        obs_drop = nmi_b - nmi_r

        # Null distribution of NMI drop under random assignment swaps per trailer
        null_drops = np.empty(n_perm)
        for i in range(n_perm):
            swap = RNG.random(len(m)) < 0.5
            s1 = np.where(swap, lb, lr)
            s2 = np.where(swap, lr, lb)
            null_drops[i] = (
                normalized_mutual_info_score(m, s1)
                - normalized_mutual_info_score(m, s2)
            )

        meta_shifts[col] = {
            "nmi_baseline": round(nmi_b, 4),
            "nmi_residual": round(nmi_r, 4),
            "nmi_drop": round(obs_drop, 4),
            "drop_null_p95": round(float(np.percentile(null_drops, 95)), 4),
            "drop_p": round(float(np.mean(null_drops >= obs_drop)), 4),
        }

    return {
        "base_vs_res_ari": round(obs_ari, 4),
        "ari_null_p95": round(float(np.percentile(null_ari, 95)), 4),
        "ari_p": round(float(np.mean(null_ari >= obs_ari)), 4),
        "base_vs_res_nmi": round(obs_nmi, 4),
        "nmi_null_p95": round(float(np.percentile(null_nmi, 95)), 4),
        "nmi_p": round(float(np.mean(null_nmi >= obs_nmi)), 4),
        "confound_reduction": meta_shifts,
    }


def analyze(df, cols, name):
    X = prep(df, cols)
    sil = {k: float(silhouette_score(X, kmeans(X, k))) for k in range(2, 9)}
    k = max(sil, key=sil.get)
    labels = kmeans(X, k)
    hdb = HDBSCAN(min_cluster_size=max(5, len(X) // 30)).fit_predict(X)
    stab = stability(X, k)

    hdb_n = int(hdb.max() + 1)
    hdb_noise = float((hdb < 0).mean())
    hdb_ari_vs_km = float(adjusted_rand_score(labels, hdb)) if hdb_n > 0 else 0.0

    stab = stability(X, k)
    return labels, dict(
        modality=name,
        n_features=X.shape[1],
        best_k=k,
        silhouette=sil,
        stability_ari_mean=stab["ari_mean"],
        stability_ari_std=stab["ari_std"],
        stability_jaccard_mean=stab["jaccard_mean"],
        stability_jaccard_std=stab["jaccard_std"],
        stability_jaccard_per_cluster=stab["jaccard_per_cluster_mean"],
        hdbscan_n_clusters=hdb_n,
        hdbscan_noise_frac=hdb_noise,
        hdbscan_ari_vs_km=hdb_ari_vs_km,
        genre=association(labels, df.primary_genre),
        decade=association(labels, df.decade),
    )

def residualize_features(X, meta_df, alpha=1.0):
    meta = meta_df[["primary_genre", "decade"]].copy()

    meta["primary_genre"] = meta["primary_genre"].fillna("unknown").astype(str)
    meta["decade"] = meta["decade"].astype("string").fillna("unknown")

    encoder = OneHotEncoder(
        sparse_output=False,
        drop="first",
        handle_unknown="ignore"
    )

    confounds = encoder.fit_transform(meta)

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

def cluster_profiles(df, X_std, feature_names, labels, top_n=4):
    profiles = {}
    unique_labels = sorted(int(c) for c in np.unique(labels))

    for cid in unique_labels:
        mask = labels == cid
        n_members = int(mask.sum())
        sub_df = df.loc[mask]

        # Since X_std is standardized across all rows, column mean == z-score vs. dataset
        mean_z = X_std[mask].mean(axis=0)
        order = np.argsort(np.abs(mean_z))[::-1]

        top_features = [
            {
                "feature": feature_names[idx],
                "mean_z": round(float(mean_z[idx]), 2),
                "direction": "high" if mean_z[idx] >= 0 else "low",
            }
            for idx in order[:top_n]
        ]

        top_genres = (
            sub_df["primary_genre"]
            .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            .dropna()
            .value_counts()
            .head(3)
            .to_dict()
        )

        top_decades = (
            sub_df["decade"]
            .dropna()
            .astype(str)
            .value_counts()
            .head(2)
            .to_dict()
        )

        profiles[cid] = {
            "count": n_members,
            "pct": round(100.0 * n_members / len(df), 1),
            "top_features": top_features,
            "top_genres": top_genres,
            "top_decades": top_decades,
        }

    return profiles

def print_cluster_profiles(name, space_label, r):
    profiles = r["profiles"]
    jacc_per_c = r["stability_jaccard_per_cluster"]
    print(
        f"--- Profiles: {name} ({space_label}) | "
        f"ARI={r['stability_ari_mean']:.3f}, Jaccard={r['stability_jaccard_mean']:.3f} ---"
    )
    for cid, p in profiles.items():
        feat_str = ", ".join(
            f"{f['feature']} ({f['mean_z']:+.2f}σ)" for f in p["top_features"]
        )
        print(
            f"  C{cid} [n={p['count']}, {p['pct']}%, J={jacc_per_c[cid]:.2f}]: {feat_str}"
        )

def export_summary_tables(full_report, out_dir=FEATURES):
    """Generates a unified RQ2 summary table (Markdown & LaTeX) from full_report."""
    base_map = {r["modality"]: r for r in full_report["baseline"]}
    res_map = {r["modality"]: r for r in full_report["residual"]}
    bvr_map = full_report["baseline_vs_residual"]

    rows = []
    for mod in base_map:
        b = base_map[mod]
        r = res_map[mod]
        t = bvr_map[mod]
        g = t["confound_reduction"]["primary_genre"]
        d = t["confound_reduction"]["decade"]

        rows.append({
            "Modality": mod,
            "k (Base/Res)": f"{b['best_k']} / {r['best_k']}",
            "Sil (Base/Res)": f"{b['silhouette'][str(b['best_k']) if str(b['best_k']) in b['silhouette'] else b['best_k']]:.3f} / {r['silhouette'][str(r['best_k']) if str(r['best_k']) in r['silhouette'] else r['best_k']]:.3f}",
            "Boot ARI (Base/Res)": f"{b['stability_ari_mean']:.2f} / {r['stability_ari_mean']:.2f}",
            "Boot Jaccard (Base/Res)": f"{b['stability_jaccard_mean']:.2f} / {r['stability_jaccard_mean']:.2f}",
            "Base vs Res ARI": f"{t['base_vs_res_ari']:.3f} (p={t['ari_p']:.3f})",
            "Genre NMI (Base->Res)": f"{g['nmi_baseline']:.3f} -> {g['nmi_residual']:.3f} (p={g['drop_p']:.3f})",
            "Decade NMI (Base->Res)": f"{d['nmi_baseline']:.3f} -> {d['nmi_residual']:.3f} (p={d['drop_p']:.3f})",
        })

    summary_df = pd.DataFrame(rows)
    md_text = summary_df.to_markdown(index=False)
    tex_text = summary_df.to_latex(index=False, escape=True)

    (out_dir / "rq2_summary_table.md").write_text(md_text)
    (out_dir / "rq2_summary_table.tex").write_text(tex_text)
    print("\n=== RQ2 SUMMARY TABLE ===")
    print(md_text)
    return summary_df

def plot_rq2_diagnostics(full_report, save_path=FIGURES / "rq2_stability_and_permutation.png"):
    """Two-panel figure:
    Left: Per-cluster Bootstrap Jaccard stability (Baseline vs. Residualized) with 0.75 threshold.
    Right: Observed Base-vs-Res ARI vs. 95th-percentile permutation null across modalities.
    """
    base_list = full_report["baseline"]
    res_list = full_report["residual"]
    bvr_map = full_report["baseline_vs_residual"]
    modalities = [r["modality"] for r in base_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # --- Panel A: Per-cluster Jaccard scatter + mean bar ---
    x = np.arange(len(modalities))
    w = 0.36

    base_means = [r["stability_jaccard_mean"] for r in base_list]
    res_means = [r["stability_jaccard_mean"] for r in res_list]

    ax1.bar(x - w / 2, base_means, width=w, color="#4C78A8", alpha=0.85, label="Baseline (mean)")
    ax1.bar(x + w / 2, res_means, width=w, color="#F58518", alpha=0.85, label="Residualized (mean)")

    # Overlay individual cluster Jaccard scores so fragile sub-clusters (e.g., dialogue C2) are visible
    for i, (b, r) in enumerate(zip(base_list, res_list)):
        jb = b["stability_jaccard_per_cluster"]
        jr = r["stability_jaccard_per_cluster"]
        ax1.scatter(np.repeat(i - w / 2, len(jb)), jb, color="#1F3B5C", s=22, zorder=4)
        ax1.scatter(np.repeat(i + w / 2, len(jr)), jr, color="#8C4303", s=22, zorder=4)

    ax1.axhline(0.75, color="green", linestyle="--", linewidth=1, label="High stability (0.75)")
    ax1.axhline(0.50, color="red", linestyle=":", linewidth=1, label="Fragile threshold (0.50)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modalities, rotation=25, ha="right")
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Bootstrap Jaccard Similarity")
    ax1.set_title("A. Bootstrap Cluster Stability (Mean & Per-Cluster)")
    ax1.legend(fontsize=8, frameon=False, loc="lower left")

    # --- Panel B: Baseline vs. Residualized ARI vs. Null 95th Percentile ---
    obs_ari = [bvr_map[m]["base_vs_res_ari"] for m in modalities]
    null_ari = [bvr_map[m]["ari_null_p95"] for m in modalities]

    ax2.barh(x, obs_ari, color="#54A24B", alpha=0.88, height=0.55, label="Observed ARI (Base vs. Res)")
    ax2.scatter(null_ari, x, color="crimson", marker="|", s=220, linewidths=2.5, zorder=5, label="Permutation Null (95th pct)")

    for i, val in enumerate(obs_ari):
        p_val = bvr_map[modalities[i]]["ari_p"]
        sig = "***" if p_val < 0.001 else ("*" if p_val < 0.05 else "ns")
        ax2.text(val + 0.015, i, f"{val:.2f} ({sig})", va="center", fontsize=8.5)

    ax2.set_yticks(x)
    ax2.set_yticklabels(modalities)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 0.98)
    ax2.set_xlabel("Adjusted Rand Index (ARI)")
    ax2.set_title("B. Archetype Persistence After Confound Removal")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved diagnostic figure -> {save_path}")


def main():
    df = pd.read_parquet(FEATURES / "trailers.parquet")

    # Standardize column types
    df["decade"] = pd.to_numeric(df["decade"], errors="coerce").astype("Int64")
    df["primary_genre"] = df["primary_genre"].astype(str)

    report_baseline = []
    report_residual = []

    all_cols = [c for g in GROUPS.values() for c in g]

    # Load the trailer status manifest alongside trailers.parquet
    status_df = pd.read_csv(IDS.parent / "trailer_status.csv")[["imdb_id", "youtube_id"]]

    # Include youtube_id when creating the export dataframe
    # out = df[["imdb_id", "title", "year", "primary_genre"]].merge(status_df, on="imdb_id", how="left")
    out = df.merge(status_df, on="imdb_id", how="left")

    print("=== RUNNING BASELINE CLUSTERING ===")
    for name, cols in [("all", all_cols), *GROUPS.items()]:
        X_base = prep(df, cols)
        labels, r = analyze(df, cols, name)
        out[f"cluster_{name}"] = labels
        r["profiles"] = cluster_profiles(df, X_base, cols, labels)
        report_baseline.append(r)
        print_cluster_profiles(name, "base", r)

        xy = compute_umap_2d(X_base)
        out[f"umap_base_{name}_x"] = np.round(xy[:, 0], 4)
        out[f"umap_base_{name}_y"] = np.round(xy[:, 1], 4)

        # labels, r = analyze(df, cols, name)
        # out[f"cluster_{name}"] = labels
        # report_baseline.append(r)
        # # print(f"{name:9s} k={r['best_k']} sil={r['silhouette'][r['best_k']]:.3f} "
        # #       f"stab={r['stability_ari_mean']:.2f} genreNMI={r['genre']['nmi']:.3f} (null95 {r['genre']['null_p95']:.3f}) "
        # #       f"decadeNMI={r['decade']['nmi']:.3f} (null95 {r['decade']['null_p95']:.3f})")
        # print(f"{name:9s} k={r['best_k']} sil={r['silhouette'][r['best_k']]:.3f} "
        #       f"stab={r['stability_ari_mean']:.2f} hdb_k={r['hdbscan_n_clusters']} "
        #       f"hdb_noise={r['hdbscan_noise_frac']:.2f} hdb_ari={r['hdbscan_ari_vs_km']:.2f} "
        #       f"genreNMI={r['genre']['nmi']:.3f} decadeNMI={r['decade']['nmi']:.3f}")

    print("\n=== RUNNING RESIDUAL CLUSTERING ===")
    for name, cols in [("all", all_cols), *GROUPS.items()]:
        X_base = prep(df, cols)
        X_res = residualize_features(X_base, df)

        labels, r = analyze(df, X_res, name)
        out[f"cluster_res_{name}"] = labels
        r["profiles"] = cluster_profiles(df, X_res, cols, labels)
        report_residual.append(r)
        print_cluster_profiles(name, "res", r)

        xy_res_mod = compute_umap_2d(X_res)
        out[f"umap_res_{name}_x"] = np.round(xy_res_mod[:, 0], 4)
        out[f"umap_res_{name}_y"] = np.round(xy_res_mod[:, 1], 4)

        # Residualized prep: prep features first, then subtract genre/decade signal
        # X_base = prep(df, cols)
        # X_res = residualize_features(X_base, df)

        # labels, r = analyze(df, X_res, name)
        # out[f"cluster_res_{name}"] = labels
        # report_residual.append(r)
        # # print(f"[RES]  {name:9s} k={r['best_k']} sil={r['silhouette'][r['best_k']]:.3f} "
        # #       f"stab={r['stability_ari_mean']:.2f} genreNMI={r['genre']['nmi']:.3f} "
        # #       f"decadeNMI={r['decade']['nmi']:.3f}")
        # print(f"{name:9s} k={r['best_k']} sil={r['silhouette'][r['best_k']]:.3f} "
        #       f"stab={r['stability_ari_mean']:.2f} hdb_k={r['hdbscan_n_clusters']} "
        #       f"hdb_noise={r['hdbscan_noise_frac']:.2f} hdb_ari={r['hdbscan_ari_vs_km']:.2f} "
        #       f"genreNMI={r['genre']['nmi']:.3f} decadeNMI={r['decade']['nmi']:.3f}")
        
    base_vs_res_tests = {}
    for name in ["all", *GROUPS.keys()]:
        base_vs_res_tests[name] = compare_baseline_vs_residual(
            out[f"cluster_{name}"].values,
            out[f"cluster_res_{name}"].values,
            df,
        )
        res_t = base_vs_res_tests[name]
        g = res_t["confound_reduction"]["primary_genre"]
        d = res_t["confound_reduction"]["decade"]
        print(
            f"[BASE vs RES] {name:9s} | "
            f"ARI={res_t['base_vs_res_ari']:.3f} (null95={res_t['ari_null_p95']:.3f}, p={res_t['ari_p']:.3f}) | "
            f"Genre NMI: {g['nmi_baseline']:.3f}->{g['nmi_residual']:.3f} (drop={g['nmi_drop']:+.3f}, p={g['drop_p']:.3f}) | "
            f"Decade NMI: {d['nmi_baseline']:.3f}->{d['nmi_residual']:.3f} (drop={d['nmi_drop']:+.3f}, p={d['drop_p']:.3f})"
        )
    
    xy_base = plot(df, prep(df, all_cols), out.cluster_all.values, FIGURES / "clusters_all_umap.png",
              f"Trailers (n={len(df)}), all features, k-means")
    # plt.show()

    xy_res = plot(df, residualize_features(prep(df, all_cols), df), out.cluster_res_all.values, FIGURES / "clusters_all_umap_res.png",
                  f"Trailers (n={len(df)}), Residualized (All Features)")
    # plt.show()
    
    out["umap_x_base"], out["umap_y_base"] = xy_base[:, 0], xy_base[:, 1]
    out["umap_x_res"], out["umap_y_res"] = xy_res[:, 0], xy_res[:, 1]

    agreement_base = cross_modality_agreement(out, prefix="cluster_", space_label="base")
    agreement_res = cross_modality_agreement(out, prefix="cluster_res_", space_label="res")

    out.to_csv(FEATURES / "clusters.csv", index=False)
    full_report = {
        "baseline": report_baseline,
        "residual": report_residual,
        "cross_modality_baseline": agreement_base,
        "cross_modality_residual": agreement_res,
        "baseline_vs_residual": base_vs_res_tests
    }
    (FEATURES / "cluster_report.json").write_text(json.dumps(full_report, indent=2))
    export_summary_tables(full_report)
    plot_rq2_diagnostics(full_report)


if __name__ == "__main__":
    main()
