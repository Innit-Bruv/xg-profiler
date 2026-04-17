"""
residuals.py — Per-player per-cluster finishing residuals.

Residual = actual goals − sum(xG)  [per player per cluster]
Minimum 15 shots per player per cluster to report a residual.
"""

from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path(__file__).parent.parent / "data" / "outputs"
MIN_SHOTS = 15  # methodological threshold — see CLAUDE.md


def compute_residuals(df: pd.DataFrame,
                      xg_col: str = "xg_xgb",
                      min_shots: int = MIN_SHOTS) -> pd.DataFrame:
    """
    Compute per-player per-cluster finishing residuals.

    Parameters
    ----------
    df : DataFrame with columns player_id, player_name, cluster_id, cluster_name,
         goal, and the chosen xg_col.
    xg_col : which xG column to use for residual computation.
    min_shots : minimum shots per player-cluster cell to report residual.

    Returns
    -------
    DataFrame indexed by (player_id, player_name, cluster_id, cluster_name) with:
        shots, actual_goals, expected_goals, residual, residual_per_shot
    Cells below min_shots are excluded (not imputed).
    """
    required = {"player_id", "player_name", "cluster_id", "cluster_name", "goal", xg_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    grouped = df.groupby(["player_id", "player_name", "cluster_id", "cluster_name"], observed=True)

    agg = grouped.agg(
        shots=("goal", "count"),
        actual_goals=("goal", "sum"),
        expected_goals=(xg_col, "sum"),
    ).reset_index()

    # Apply minimum shot threshold
    agg = agg[agg["shots"] >= min_shots].copy()

    agg["residual"] = agg["actual_goals"] - agg["expected_goals"]
    agg["residual_per_shot"] = agg["residual"] / agg["shots"]

    agg = agg.sort_values("residual", ascending=False)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUTPUTS_DIR / "residuals.csv", index=False)
    print(f"Saved residuals.csv ({len(agg)} player-cluster pairs, min_shots={min_shots})")

    return agg


def robustness_check(df: pd.DataFrame,
                     min_shots: int = MIN_SHOTS) -> pd.DataFrame:
    """
    Compute residuals for both xG baselines (Understat and XGBoost)
    and compare rankings per cluster.

    Returns wide DataFrame with both residual columns for shared player-cluster pairs.
    """
    available_xg_cols = [c for c in ["xg_understat", "xg_xgb", "xg_lr"] if c in df.columns]
    if len(available_xg_cols) < 2:
        raise ValueError(f"Need at least two xG columns for robustness check, found: {available_xg_cols}")

    frames = {}
    for col in available_xg_cols:
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        res = compute_residuals(sub, xg_col=col, min_shots=min_shots)
        res = res.set_index(["player_id", "player_name", "cluster_id", "cluster_name"])
        frames[col] = res[["residual"]].rename(columns={"residual": f"residual_{col}"})

    combined = pd.concat(frames.values(), axis=1, join="inner").reset_index()
    combined = combined.sort_values(combined.columns[-1], ascending=False)
    return combined


def top_finishers(residuals: pd.DataFrame,
                  cluster: str | int,
                  n: int = 10,
                  cluster_col: str = "cluster_name") -> pd.DataFrame:
    """Return top N finishers (positive residual) for a given cluster."""
    mask = residuals[cluster_col] == cluster
    return residuals[mask].nlargest(n, "residual")[
        ["player_name", "cluster_name", "shots", "actual_goals", "expected_goals", "residual"]
    ]


def bottom_finishers(residuals: pd.DataFrame,
                     cluster: str | int,
                     n: int = 10,
                     cluster_col: str = "cluster_name") -> pd.DataFrame:
    """Return bottom N finishers (negative residual) for a given cluster."""
    mask = residuals[cluster_col] == cluster
    return residuals[mask].nsmallest(n, "residual")[
        ["player_name", "cluster_name", "shots", "actual_goals", "expected_goals", "residual"]
    ]
