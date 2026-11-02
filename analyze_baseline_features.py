import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_parquet("data/features/trailers.parquet")

# 2. Metadata vs. Feature separation
meta_cols = [
    "imdb_id",
    "title",
    "year",
    "genres",
    "primary_genre",
    "decade",
]

# Select all numeric feature columns
feature_cols = [
    c
    for c in df.select_dtypes(include=[np.number]).columns
    if c not in meta_cols
]

df["decade"] = df["decade"].astype("int64")

# Standardize feature matrix
X = df[feature_cols].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Principal Component Analysis
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]

# Extract feature weights (loadings)
loadings = pd.DataFrame(
    pca.components_.T, index=feature_cols, columns=["PC1_weight", "PC2_weight"]
)
loadings["PC1_abs"] = loadings["PC1_weight"].abs()
loadings["PC2_abs"] = loadings["PC2_weight"].abs()

print("=== TOP 5 FEATURES DRIVING PC1 (Horizontal Axis) ===")
print(loadings.sort_values(by="PC1_abs", ascending=False)[["PC1_weight"]].head(5))

print("\n=== TOP 5 FEATURES DRIVING PC2 (Vertical Axis) ===")
print(loadings.sort_values(by="PC2_abs", ascending=False)[["PC2_weight"]].head(5))

# 4. Generate visual plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.scatterplot(
    data=df,
    x="pca_1",
    y="pca_2",
    hue="primary_genre",
    alpha=0.75,
    ax=axes[0],
    palette="tab10",
)
axes[0].set_title(
    f"Baseline Clusters by Genre (PC1: {pca.explained_variance_ratio_[0]:.1%} var)"
)

sns.scatterplot(
    data=df,
    x="pca_1",
    y="pca_2",
    hue="decade",
    alpha=0.75,
    ax=axes[1],
    palette="viridis",
)
axes[1].set_title(
    f"Baseline Clusters by Decade (PC2: {pca.explained_variance_ratio_[1]:.1%} var)"
)

plt.tight_layout()
plt.show()
# plt.savefig("baseline_pca_features.png", dpi=300)
# print("\nSaved scatter plot to baseline_pca_features.png")