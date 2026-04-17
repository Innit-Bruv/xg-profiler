"""
scraper.py — Fetch shot-level data from Understat and StatsBomb.

Understat: async scraper via the `understat` PyPI package.
StatsBomb: synchronous via statsbombpy.
"""

import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
UNDERSTAT_DIR = RAW_DIR / "understat"
STATSBOMB_DIR = RAW_DIR / "statsbomb"

LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1"]
SEASONS = list(range(2014, 2025))  # 2014/15 through 2024/25


# ---------------------------------------------------------------------------
# Understat
# ---------------------------------------------------------------------------

async def _fetch_league_shots_async(league: str, season: int) -> pd.DataFrame:
    """
    Fetch all shots for a league/season from Understat.
    Uses get_league_results → match IDs → get_match_shots per match.
    This correctly scopes shots to the given season.
    """
    import aiohttp
    import understat as us

    async with aiohttp.ClientSession() as session:
        u = us.Understat(session)
        matches = await u.get_league_results(league, season)
        rows = []
        for match in tqdm(matches, desc=f"{league} {season}", leave=False):
            match_id = match["id"]
            try:
                shots = await u.get_match_shots(match_id)
                # shots is {"h": [...], "a": [...]}
                all_shots = shots.get("h", []) + shots.get("a", [])
                for shot in all_shots:
                    shot["league"] = league
                rows.extend(all_shots)
            except Exception as e:
                print(f"  Warning: match {match_id} failed: {e}")
        df = pd.DataFrame(rows)
        if not df.empty:
            df["season"] = season  # ensure season column is consistent int
        return df


def fetch_understat(leagues=None, seasons=None, overwrite=False) -> pd.DataFrame:
    """
    Scrape Understat for all specified leagues/seasons.
    Saves per-league-season CSVs to data/raw/understat/ and returns combined DataFrame.
    Excludes penalties.
    """
    UNDERSTAT_DIR.mkdir(parents=True, exist_ok=True)
    leagues = leagues or LEAGUES
    seasons = seasons or SEASONS

    frames = []
    for league in leagues:
        for season in seasons:
            out_path = UNDERSTAT_DIR / f"{league}_{season}.csv"
            if out_path.exists() and not overwrite:
                df = pd.read_csv(out_path)
                frames.append(df)
                continue
            print(f"Fetching {league} {season}...")
            df = asyncio.run(_fetch_league_shots_async(league, season))
            df.to_csv(out_path, index=False)
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Exclude penalties — fixed xG situation, distorts clustering
    combined = combined[combined["situation"] != "Penalty"]
    return combined


def load_understat_cache() -> pd.DataFrame:
    """Load all cached Understat CSVs without re-scraping."""
    files = list(UNDERSTAT_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No cached Understat data found. Run fetch_understat() first.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["situation"] != "Penalty"]
    return df


def load_statsbomb_cache() -> pd.DataFrame:
    """
    Load all cached StatsBomb shot CSVs from data/raw/statsbomb/.

    Files are named {competition_id}_{season_id}.csv (one per competition-season).
    Excludes penalty shots.
    """
    files = list(STATSBOMB_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No cached StatsBomb data found. Run fetch_statsbomb() first.")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"  Warning: could not load {f.name}: {e}")
    if not frames:
        raise RuntimeError("All StatsBomb CSVs failed to load.")
    df = pd.concat(frames, ignore_index=True)
    # Exclude penalties
    if "shot_type_name" in df.columns:
        df = df[df["shot_type_name"].str.lower() != "penalty"]
    return df


# ---------------------------------------------------------------------------
# StatsBomb
# ---------------------------------------------------------------------------

def fetch_statsbomb(overwrite=False) -> pd.DataFrame:
    """
    Load all open-data StatsBomb matches that contain shot events.
    Saves to data/raw/statsbomb/shots_raw.csv.
    Returns shot-level DataFrame with freeze-frame info.
    """
    from statsbombpy import sb

    STATSBOMB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATSBOMB_DIR / "shots_raw.csv"

    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    competitions = sb.competitions()
    # Use free open-data competitions only
    all_shots = []

    for _, comp in tqdm(competitions.iterrows(), total=len(competitions), desc="Competitions"):
        competition_id = comp["competition_id"]
        season_id = comp["season_id"]
        try:
            matches = sb.matches(competition_id=competition_id, season_id=season_id)
            for _, match in matches.iterrows():
                match_id = match["match_id"]
                try:
                    events = sb.events(match_id=match_id)
                    shots = events[events["type"] == "Shot"].copy()
                    if shots.empty:
                        continue
                    shots["competition_id"] = competition_id
                    shots["season_id"] = season_id
                    shots["competition_name"] = comp["competition_name"]
                    shots["season_name"] = comp["season_name"]
                    all_shots.append(shots)
                except Exception:
                    pass
        except Exception:
            pass

    if not all_shots:
        raise RuntimeError("No StatsBomb shots fetched — check statsbombpy credentials or open-data availability.")

    df = pd.concat(all_shots, ignore_index=True)
    # Exclude penalties
    df = df[df["shot_type"] != "Penalty"]
    # Serialise freeze frames as JSON strings for CSV storage
    if "shot_freeze_frame" in df.columns:
        df["shot_freeze_frame"] = df["shot_freeze_frame"].apply(
            lambda x: json.dumps(x) if isinstance(x, list) else x
        )
    df.to_csv(out_path, index=False)
    return df
