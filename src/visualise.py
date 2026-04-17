"""
visualise.py — Visualisation utilities.

Key output: player × cluster finishing residual heatmap.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUTS_DIR = Path(__file__).parent.parent / "data" / "outputs"


def finishing_heatmap(residuals: pd.DataFrame,
                      top_n_players: int = 40,
                      xg_col_suffix: str = "xg_xgb",
                      save_path: Path | None = None) -> None:
    """
    Plot player × cluster finishing residual heatmap.

    Shows the top_n_players with the highest absolute residual across all clusters.
    White = neutral, blue = over-performs, red = under-performs.
    Cells with insufficient data (below min_shots threshold) are shown as grey.
    """
    residual_col = "residual"
    if residual_col not in residuals.columns:
        raise ValueError("residuals DataFrame must have a 'residual' column")

    # Pivot to player × cluster matrix
    pivot = residuals.pivot_table(
        index="player_name",
        columns="cluster_name",
        values=residual_col,
        aggfunc="sum"
    )

    # Select players with highest total absolute residual
    player_magnitude = pivot.abs().sum(axis=1).nlargest(top_n_players)
    pivot = pivot.loc[player_magnitude.index]

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.5), max(12, top_n_players * 0.35)))

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)].max()),
               abs(pivot.values[~np.isnan(pivot.values)].min()),
               0.5)

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdBu",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.3,
        linecolor="lightgray",
        cbar_kws={"label": "Finishing Residual (Goals Above/Below xG)"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        mask=pivot.isna(),
    )

    ax.set_title(
        "Shot-Type Finishing Profiles\n(Residual = Actual Goals − xG, per player per cluster)",
        fontsize=14, pad=15
    )
    ax.set_xlabel("Shot Cluster", fontsize=11)
    ax.set_ylabel("Player", fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved heatmap to {save_path}")
    plt.show()


def cluster_distribution(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Bar chart of shot count per cluster."""
    counts = df["cluster_name"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Shot Distribution Across Clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Shots")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def player_profile(residuals: pd.DataFrame, player_name: str) -> None:
    """Bar chart of a single player's residual across all clusters."""
    row = residuals[residuals["player_name"] == player_name]
    if row.empty:
        print(f"No residual data found for '{player_name}' (check min_shots threshold).")
        return

    row = row.set_index("cluster_name")["residual"].sort_values(ascending=False)

    colors = ["steelblue" if v >= 0 else "tomato" for v in row]
    fig, ax = plt.subplots(figsize=(8, 4))
    row.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Finishing Profile: {player_name}")
    ax.set_xlabel("Shot Cluster")
    ax.set_ylabel("Residual (Goals Above/Below xG)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()


def xg_comparison_scatter(df: pd.DataFrame,
                           col_a: str = "xg_xgb",
                           col_b: str = "xg_understat",
                           save_path: Path | None = None) -> None:
    """Scatter plot comparing two xG baselines shot by shot."""
    sub = df[[col_a, col_b]].dropna()
    if sub.empty:
        print("No overlapping rows with both xG columns. Skipping scatter.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(sub[col_a], sub[col_b], alpha=0.2, s=5, color="steelblue")
    lim = max(sub.max().max(), 0.05)
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8)
    ax.set_xlabel(col_a)
    ax.set_ylabel(col_b)
    ax.set_title(f"xG Comparison: {col_a} vs {col_b}")
    corr = sub.corr().iloc[0, 1]
    ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
