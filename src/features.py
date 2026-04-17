"""
features.py — Feature engineering for shot-level data.

Handles both Understat (coordinate system 0-1) and StatsBomb (120x80 yards).
Outputs a unified feature matrix ready for StandardScaler + ML.
"""

import json
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

# StatsBomb: pitch 120x80 yards. Goal mouth at x=120, y=36–44.
SB_GOAL_X = 120.0
SB_GOAL_Y = 40.0
SB_POST_Y_LEFT = 36.0
SB_POST_Y_RIGHT = 44.0

# Understat: percentage of pitch length/width (0-100 scale). Goal at x≈100, y≈50.
US_GOAL_X = 100.0
US_GOAL_Y = 50.0


def _sb_distance(x: float, y: float) -> float:
    return math.sqrt((SB_GOAL_X - x) ** 2 + (SB_GOAL_Y - y) ** 2)


def _sb_angle(x: float, y: float) -> float:
    """Angle in degrees subtended by goal posts from shot location."""
    a = math.sqrt((SB_GOAL_X - x) ** 2 + (SB_POST_Y_LEFT - y) ** 2)
    b = math.sqrt((SB_GOAL_X - x) ** 2 + (SB_POST_Y_RIGHT - y) ** 2)
    c = abs(SB_POST_Y_RIGHT - SB_POST_Y_LEFT)
    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b + 1e-9)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _us_distance(x: float, y: float) -> float:
    """Euclidean distance to goal centre in Understat's 0-100 scale."""
    return math.sqrt((US_GOAL_X - x) ** 2 + (US_GOAL_Y - y) ** 2)


def _us_angle(x: float, y: float) -> float:
    """Shot angle for Understat coordinates. Goal width ≈ 10 units on the 0-100 scale."""
    post_y1, post_y2 = 45.0, 55.0
    a = math.sqrt((US_GOAL_X - x) ** 2 + (post_y1 - y) ** 2)
    b = math.sqrt((US_GOAL_X - x) ** 2 + (post_y2 - y) ** 2)
    c = abs(post_y2 - post_y1)
    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b + 1e-9)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


# ---------------------------------------------------------------------------
# Freeze frame parsing (StatsBomb only)
# ---------------------------------------------------------------------------

def parse_freeze_frame(raw) -> dict:
    """
    Extract aggregate features from a StatsBomb freeze frame.
    Returns dict with: defenders_in_triangle, gk_distance_to_goal.
    """
    result = {"defenders_in_triangle": 0, "gk_distance_to_goal": np.nan}

    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return result

    if isinstance(raw, str):
        try:
            frame = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return result
    else:
        frame = raw

    if not isinstance(frame, list):
        return result

    for player in frame:
        loc = player.get("location", [None, None])
        if player.get("keeper"):
            if loc[0] is not None and loc[1] is not None:
                result["gk_distance_to_goal"] = math.sqrt(
                    (SB_GOAL_X - loc[0]) ** 2 + (SB_GOAL_Y - loc[1]) ** 2
                )

    return result


# ---------------------------------------------------------------------------
# Game state reconstruction (Understat)
# ---------------------------------------------------------------------------

def add_game_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct real-time game state from shot sequence within each match.
    Adds 'game_state' column: 'winning' / 'drawing' / 'losing' from shooter's perspective.

    Uses the result column to accumulate goals as shots happen (ordered by minute).
    Only open-play goals recorded as shots are captured; own goals may be missed.
    """
    df = df.copy()
    df = df.sort_values(["match_id", "minute"]).reset_index(drop=True)

    is_goal = (df["result"] == "Goal").astype(int)
    df["_hg"] = is_goal * (df["h_a"] == "h").astype(int)
    df["_ag"] = is_goal * (df["h_a"] == "a").astype(int)

    # Cumulative goals up to but NOT including the current shot
    df["_cumh"] = df.groupby("match_id")["_hg"].cumsum() - df["_hg"]
    df["_cuma"] = df.groupby("match_id")["_ag"].cumsum() - df["_ag"]

    h_mask = df["h_a"] == "h"
    diff = np.where(h_mask, df["_cumh"] - df["_cuma"], df["_cuma"] - df["_cumh"])
    df["game_state"] = np.where(diff > 0, "winning", np.where(diff < 0, "losing", "drawing"))

    df = df.drop(columns=["_hg", "_ag", "_cumh", "_cuma"])
    return df


# ---------------------------------------------------------------------------
# Understat feature engineering
# ---------------------------------------------------------------------------

def engineer_understat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from raw Understat shot DataFrame.

    Returns DataFrame with shot_id, player metadata, 15 clustering features,
    xG baseline, binary goal target, and match context columns.
    """
    out = pd.DataFrame()

    out["shot_id"] = df["id"].astype(str)
    out["player_id"] = df["player_id"].astype(str)
    out["player_name"] = df["player"]
    out["league"] = df.get("league", pd.Series(["unknown"] * len(df)))
    out["season"] = df.get("season", pd.Series([0] * len(df)))
    out["match_id"] = df.get("match_id", pd.Series([None] * len(df)))
    out["minute"] = pd.to_numeric(df.get("minute"), errors="coerce")
    out["h_a"] = df.get("h_a", pd.Series(["unknown"] * len(df)))

    x = pd.to_numeric(df["X"], errors="coerce") * 100  # Understat stores 0-1; scale to 0-100
    y = pd.to_numeric(df["Y"], errors="coerce") * 100

    out["distance"] = [_us_distance(xi, yi) if pd.notna(xi) and pd.notna(yi) else np.nan
                       for xi, yi in zip(x, y)]
    out["angle"] = [_us_angle(xi, yi) if pd.notna(xi) and pd.notna(yi) else np.nan
                    for xi, yi in zip(x, y)]

    # 18-yard box: X > 83, Y between 21 and 79 (on 0-100 scale)
    out["shot_in_box"] = ((x > 83) & (y > 21) & (y < 79)).astype(int)

    # 18-zone grid features (zones 13-18 cover 98% of all shots)
    # zone_row = floor(x / (100/6)), zone_col = floor(y / (100/3))
    # zone_id  = zone_row * 3 + zone_col + 1
    zone_row = (x * 6 / 100).apply(math.floor).clip(0, 5)
    zone_col = (y * 3 / 100).apply(math.floor).clip(0, 2)
    zone_id  = zone_row * 3 + zone_col + 1
    for z in [13, 14, 15, 16, 17, 18]:
        out[f"zone_{z}"] = (zone_id == z).astype(int)

    # Body part
    shot_type = df.get("shotType", df.get("shot_type", pd.Series(["unknown"] * len(df))))
    shot_type_lower = shot_type.str.lower()
    out["is_header"] = (shot_type_lower == "head").astype(int)
    out["is_left_foot"] = (shot_type_lower == "leftfoot").astype(int)
    out["is_right_foot"] = (shot_type_lower == "rightfoot").astype(int)

    # Situation
    situation = df.get("situation", pd.Series(["unknown"] * len(df))).str.lower()
    out["from_corner"] = (situation == "fromcorner").astype(int)
    out["from_set_piece"] = (situation == "setpiece").astype(int)
    out["from_freekick"] = (situation == "directfreekick").astype(int)
    out["open_play"] = (situation == "openplay").astype(int)

    # lastAction proxy features — key clustering signals for Understat
    last_action = df.get("lastAction", pd.Series(["unknown"] * len(df))).str.lower()
    out["preceded_by_cross"] = (last_action == "cross").astype(int)
    out["preceded_by_aerial"] = (last_action == "aerial").astype(int)
    out["preceded_by_dribble"] = (last_action == "takeon").astype(int)
    out["preceded_by_throughball"] = (last_action == "throughball").astype(int)
    out["preceded_by_rebound"] = (last_action == "rebound").astype(int)
    out["preceded_by_layoff"] = (last_action == "layoff").astype(int)

    # StatsBomb-only features — not available in Understat, filled as 0
    out["first_touch"] = 0
    out["one_on_one"] = 0

    # xG baseline
    out["xg_understat"] = pd.to_numeric(df.get("xG", df.get("xg", None)), errors="coerce")
    out["xg_statsbomb"] = np.nan

    # Target
    result_col = df.get("result", pd.Series(["unknown"] * len(df)))
    out["goal"] = result_col.map(lambda r: 1 if str(r) == "Goal" else 0)
    # Keep raw result for game_state reconstruction
    out["result"] = result_col

    out["source"] = "understat"

    return out


# ---------------------------------------------------------------------------
# StatsBomb feature engineering
# ---------------------------------------------------------------------------

def engineer_statsbomb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from pre-processed StatsBomb shot CSV.

    Column names match the per-competition CSVs in data/raw/statsbomb/
    (not raw statsbombpy output).
    """
    out = pd.DataFrame()

    out["shot_id"] = df.get("event_id", pd.Series([f"sb_{i}" for i in range(len(df))])).astype(str)
    out["player_id"] = df.get("player_id", pd.Series(["unknown"] * len(df))).astype(str)
    out["player_name"] = df.get("player_name", pd.Series(["unknown"] * len(df)))
    out["league"] = df.get("competition_name", pd.Series(["unknown"] * len(df)))
    out["season"] = df.get("season_name", pd.Series(["unknown"] * len(df)))
    out["match_id"] = df.get("match_id", pd.Series([None] * len(df)))
    out["minute"] = pd.to_numeric(df.get("minute"), errors="coerce")

    xs = pd.to_numeric(df.get("shot_x"), errors="coerce")
    ys = pd.to_numeric(df.get("shot_y"), errors="coerce")

    out["distance"] = [_sb_distance(xi, yi) if pd.notna(xi) and pd.notna(yi) else np.nan
                       for xi, yi in zip(xs, ys)]
    out["angle"] = [_sb_angle(xi, yi) if pd.notna(xi) and pd.notna(yi) else np.nan
                    for xi, yi in zip(xs, ys)]

    # 18-yard box in StatsBomb coords: x > 102, 18 < y < 62
    out["shot_in_box"] = ((xs > 102) & (ys > 18) & (ys < 62)).astype(int)

    # Body part
    body_part = df.get("body_part_name", pd.Series(["unknown"] * len(df))).str.lower()
    out["is_header"] = body_part.str.contains("head", na=False).astype(int)
    out["is_left_foot"] = body_part.str.contains("left", na=False).astype(int)
    out["is_right_foot"] = body_part.str.contains("right", na=False).astype(int)

    # Situation (from shot_type_name: "Open Play", "Corner", "Free Kick", "Penalty")
    shot_type_name = df.get("shot_type_name", pd.Series(["unknown"] * len(df))).str.lower()
    out["from_corner"] = shot_type_name.str.contains("corner", na=False).astype(int)
    out["from_set_piece"] = shot_type_name.str.contains("free kick", na=False).astype(int)
    out["from_freekick"] = out["from_set_piece"]  # StatsBomb doesn't distinguish indirect vs direct
    out["open_play"] = shot_type_name.str.contains("open play", na=False).astype(int)

    # lastAction proxies — use assist columns where available
    out["preceded_by_cross"] = df.get("assist_cross", pd.Series([False] * len(df))).fillna(False).astype(int)
    out["preceded_by_layoff"] = df.get("assist_cut_back", pd.Series([False] * len(df))).fillna(False).astype(int)
    out["preceded_by_throughball"] = df.get("assist_through_ball", pd.Series([False] * len(df))).fillna(False).astype(int)
    out["preceded_by_dribble"] = df.get("follows_dribble", pd.Series([False] * len(df))).fillna(False).astype(int)
    # No direct proxy for aerial or rebound in StatsBomb — default 0
    out["preceded_by_aerial"] = 0
    out["preceded_by_rebound"] = 0

    # StatsBomb-rich features
    out["first_touch"] = df.get("first_time", pd.Series([False] * len(df))).fillna(False).astype(int)
    out["one_on_one"] = df.get("one_on_one", pd.Series([False] * len(df))).fillna(False).astype(int)

    # Freeze frame derived features (optional enrichment)
    if "freeze_frame" in df.columns:
        ff_features = df["freeze_frame"].apply(parse_freeze_frame).apply(pd.Series)
        out["defenders_in_triangle"] = ff_features["defenders_in_triangle"]
        out["gk_distance_to_goal"] = ff_features["gk_distance_to_goal"]
    else:
        out["defenders_in_triangle"] = np.nan
        out["gk_distance_to_goal"] = np.nan

    out["xg_understat"] = np.nan
    out["xg_statsbomb"] = pd.to_numeric(df.get("xg_statsbomb"), errors="coerce")

    outcome = df.get("outcome_name", pd.Series(["unknown"] * len(df))).str.lower()
    out["goal"] = outcome.str.contains("goal", na=False).astype(int)
    out["result"] = df.get("outcome_name", pd.Series(["unknown"] * len(df)))

    out["source"] = "statsbomb"

    return out


# ---------------------------------------------------------------------------
# Cluster feature list — 15 Understat-primary features (DATA_STRATEGY Phase 1)
# ---------------------------------------------------------------------------

# V1: original 15-feature set (includes body part — preserves right/left foot split)
CLUSTER_FEATURES_V1 = [
    "distance", "angle",
    "is_header", "is_right_foot", "is_left_foot",
    "from_corner", "from_set_piece", "from_freekick",
    "preceded_by_cross", "preceded_by_aerial", "preceded_by_dribble",
    "preceded_by_throughball", "preceded_by_rebound", "preceded_by_layoff",
    "shot_in_box",
]

# V2: situational features — drops right/left foot, adds 6 zone flags
# K-Means clusters by shot context and pitch location, not execution mechanism
CLUSTER_FEATURES_V2 = [
    "distance", "angle",
    "is_header",
    "from_corner", "from_set_piece", "from_freekick",
    "preceded_by_cross", "preceded_by_aerial", "preceded_by_dribble",
    "preceded_by_throughball", "preceded_by_rebound", "preceded_by_layoff",
    "shot_in_box",
    "zone_13", "zone_14", "zone_15", "zone_16", "zone_17", "zone_18",
]

# Default alias used by clustering.py and xg_model.py
CLUSTER_FEATURES = CLUSTER_FEATURES_V1

# xG model features: clustering features + minute (game_state added separately if available)
XG_FEATURES_LR = [
    "distance", "angle",
    "is_header", "is_right_foot", "is_left_foot",
    "from_corner", "from_set_piece", "from_freekick",
    "preceded_by_cross",
]

XG_FEATURES_XGB = CLUSTER_FEATURES + ["minute"]


# ---------------------------------------------------------------------------
# Build & save combined feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(understat_df: pd.DataFrame | None = None,
                         statsbomb_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Combine engineered features from both sources into one DataFrame.
    Missing numeric values are imputed with column medians.
    """
    parts = []
    if understat_df is not None:
        parts.append(engineer_understat(understat_df))
    if statsbomb_df is not None:
        parts.append(engineer_statsbomb(statsbomb_df))

    if not parts:
        raise ValueError("Must supply at least one of understat_df or statsbomb_df.")

    combined = pd.concat(parts, ignore_index=True)

    numeric_cols = combined.select_dtypes(include="number").columns
    for col in numeric_cols:
        median = combined[col].median()
        combined[col] = combined[col].fillna(median)

    return combined
