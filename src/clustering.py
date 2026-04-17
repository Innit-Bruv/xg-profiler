"""
clustering.py — K-Means shot clustering pipeline.

Runs K=3 to 10, plots elbow + silhouette, fits optimal K,
and saves cluster labels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUTPUTS_DIR = Path(__file__).parent.parent / "data" / "outputs"

# 15 Understat-primary clustering features — matches DATA_STRATEGY Phase 1
CLUSTER_FEATURES = [
    "distance",
    "angle",
    "is_header",
    "is_right_foot",
    "is_left_foot",
    "from_corner",
    "from_set_piece",
    "from_freekick",
    "preceded_by_cross",
    "preceded_by_aerial",
    "preceded_by_dribble",
    "preceded_by_throughball",
    "preceded_by_rebound",
    "preceded_by_layoff",
    "shot_in_box",
]


def _get_available_features(df: pd.DataFrame) -> list[str]:
    return [f for f in CLUSTER_FEATURES if f in df.columns]


def elbow_silhouette(df: pd.DataFrame,
                     k_range: range = range(3, 11),
                     save_path: Path | None = None) -> dict:
    """
    Compute inertia and silhouette score for each K.
    Plots elbow curve and silhouette scores side by side.
    Returns dict mapping K -> {"inertia": ..., "silhouette": ...}.
    """
    features = _get_available_features(df)
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels, sample_size=min(10_000, len(X_scaled)))
        results[k] = {"inertia": km.inertia_, "silhouette": sil}
        print(f"  K={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.4f}")

    ks = list(results.keys())
    inertias = [results[k]["inertia"] for k in ks]
    silhouettes = [results[k]["silhouette"] for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, inertias, "bo-")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, silhouettes, "ro-")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Scores")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return results


def fit_kmeans(df: pd.DataFrame, k: int) -> tuple[KMeans, StandardScaler, np.ndarray]:
    """
    Fit K-Means with the chosen K.
    Returns (fitted KMeans, fitted scaler, cluster label array).
    """
    features = _get_available_features(df)
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, scaler, labels


def inspect_centroids(km: KMeans,
                      scaler: StandardScaler,
                      feature_names: list[str]) -> pd.DataFrame:
    """
    Return centroid values in original (unscaled) units for human inspection.
    Use this output to manually assign cluster names.
    """
    centroids_scaled = km.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    return pd.DataFrame(centroids_orig, columns=feature_names).round(3)


def assign_cluster_names(k: int) -> dict[int, str]:
    """
    Placeholder — fill in manually after inspecting centroids.
    Maps cluster int ID -> descriptive name.
    """
    default = {i: f"cluster_{i}" for i in range(k)}
    # Example (edit after inspection):
    # default = {
    #     0: "close_range_first_time",
    #     1: "header_from_cross",
    #     2: "long_range_strike",
    #     3: "set_piece_header",
    #     4: "one_on_one",
    #     5: "cutback_tap_in",
    # }
    return default


def run_clustering(df: pd.DataFrame,
                   k: int,
                   save: bool = True) -> pd.DataFrame:
    """
    Full clustering run:
    - Fit K-Means with given K
    - Attach cluster_id and cluster_name columns to df
    - Save cluster_labels.csv
    Returns augmented DataFrame.
    """
    features = _get_available_features(df)
    km, scaler, labels = fit_kmeans(df, k)

    centroids = inspect_centroids(km, scaler, features)
    print("\nCentroid inspection (original scale):")
    print(centroids.to_string())

    cluster_names = assign_cluster_names(k)
    df = df.copy()
    df["cluster_id"] = labels
    df["cluster_name"] = df["cluster_id"].map(cluster_names)

    final_sil = silhouette_score(
        scaler.transform(df[features].values),
        labels,
        sample_size=min(10_000, len(df))
    )
    print(f"\nFinal silhouette score (K={k}): {final_sil:.4f}")

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out = df[["player_id", "cluster_id", "cluster_name"]].copy()
        out.to_csv(OUTPUTS_DIR / "cluster_labels.csv", index=False)
        print(f"Saved cluster_labels.csv ({len(out)} rows)")

    return df
