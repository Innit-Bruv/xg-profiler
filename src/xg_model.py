"""
xg_model.py — Logistic Regression and XGBoost xG models.

Both evaluated with 5-fold cross-validation.
Metrics: Brier score (primary), ROC-AUC, log-loss, calibration plot.
SHAP values computed for the XGBoost model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

OUTPUTS_DIR = Path(__file__).parent.parent / "data" / "outputs"

# Full feature set for XGBoost (15 cluster features + minute)
MODEL_FEATURES = [
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
    "minute",
]

# Simpler LR baseline — fewer, most interpretable features
LR_FEATURES = [
    "distance",
    "angle",
    "is_header",
    "is_right_foot",
    "is_left_foot",
    "from_corner",
    "from_set_piece",
    "from_freekick",
    "preceded_by_cross",
]

TARGET = "goal"


def _get_X_y(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    feat_list = features if features is not None else MODEL_FEATURES
    available = [f for f in feat_list if f in df.columns]
    X = df[available].copy()
    y = df[TARGET].copy()
    return X, y


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic(df: pd.DataFrame) -> tuple[LogisticRegression, StandardScaler, dict]:
    """
    Train logistic regression xG model with 5-fold CV.
    Returns (fitted model, fitted scaler, cv_metrics dict).
    """
    X, y = _get_X_y(df, features=LR_FEATURES)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_scaled, y, cv=cv,
        scoring=["roc_auc", "neg_brier_score", "neg_log_loss"],
        return_train_score=False
    )

    metrics = {
        "roc_auc_mean": cv_results["test_roc_auc"].mean(),
        "roc_auc_std": cv_results["test_roc_auc"].std(),
        "brier_mean": -cv_results["test_neg_brier_score"].mean(),
        "brier_std": cv_results["test_neg_brier_score"].std(),
        "log_loss_mean": -cv_results["test_neg_log_loss"].mean(),
        "log_loss_std": cv_results["test_neg_log_loss"].std(),
    }

    # Fit on full data
    model.fit(X_scaled, y)
    print("Logistic Regression CV results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, scaler, metrics


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(df: pd.DataFrame) -> tuple[XGBClassifier, dict]:
    """
    Train XGBoost xG model with 5-fold CV.
    Returns (fitted model, cv_metrics dict).
    No separate scaler needed — tree models are scale-invariant.
    """
    X, y = _get_X_y(df)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=cv,
        scoring=["roc_auc", "neg_brier_score", "neg_log_loss"],
        return_train_score=False
    )

    metrics = {
        "roc_auc_mean": cv_results["test_roc_auc"].mean(),
        "roc_auc_std": cv_results["test_roc_auc"].std(),
        "brier_mean": -cv_results["test_neg_brier_score"].mean(),
        "brier_std": cv_results["test_neg_brier_score"].std(),
        "log_loss_mean": -cv_results["test_neg_log_loss"].mean(),
        "log_loss_std": cv_results["test_neg_log_loss"].std(),
    }

    model.fit(X, y)
    print("XGBoost CV results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, metrics


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def predict_xg(df: pd.DataFrame,
               lr_model: LogisticRegression,
               lr_scaler: StandardScaler,
               xgb_model: XGBClassifier) -> pd.DataFrame:
    """
    Attach xg_lr and xg_xgb columns to df.
    Saves xg_predictions.csv.
    """
    X_lr, _ = _get_X_y(df, features=LR_FEATURES)
    X_xgb, _ = _get_X_y(df, features=MODEL_FEATURES)
    df = df.copy()
    df["xg_lr"] = lr_model.predict_proba(lr_scaler.transform(X_lr))[:, 1]
    df["xg_xgb"] = xgb_model.predict_proba(X_xgb)[:, 1]

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df[["player_id", "player_name", "goal", "xg_understat", "xg_lr", "xg_xgb",
        "cluster_id", "cluster_name"]].to_csv(
        OUTPUTS_DIR / "xg_predictions.csv", index=False
    )
    print(f"Saved xg_predictions.csv ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Evaluation plots
# ---------------------------------------------------------------------------

def calibration_plot(df: pd.DataFrame,
                     xg_col: str = "xg_xgb",
                     save_path: Path | None = None) -> None:
    """Plot calibration curve: predicted xG vs observed goal rate."""
    prob_true, prob_pred = calibration_curve(df[TARGET], df[xg_col], n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, "bo-", label=f"Model ({xg_col})")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Predicted xG")
    plt.ylabel("Observed goal rate")
    plt.title("Calibration Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def shap_importance(xgb_model: XGBClassifier,
                    df: pd.DataFrame,
                    save_path: Path | None = None) -> None:
    """Compute and plot SHAP feature importance for XGBoost model."""
    X, _ = _get_X_y(df)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (XGBoost xG)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
