"""Frozen v1 pipeline: raw JSON -> app CSV, in one command.

This replaces phase1.ipynb, phase2.ipynb and phase3_fixed_model.py. It reproduces
data/app/players_scored.csv byte-for-byte in its numbers. It is deliberately NOT
improved: the scoring formula and the model are kept exactly as v1 had them, so
v1 stays reproducible as the baseline v2 has to beat. The known problems are
listed in AUDIT.md (sections 2 and 3); do not fix them here, fix them in v2.

Run from the repo root:
    python pipeline/v1_pipeline.py          # rebuild everything
    python pipeline/verify_data.py          # check the outputs match MANIFEST.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
RAW_JSON = ROOT / "data/raw/player_performance.json"
ROLES_TEAMS = ROOT / "data/reference/player_roles_teams_2025.csv"
SEASONS_CSV = ROOT / "data/processed/ipl_2023_2025_player_seasons.csv"
SCORES_CSV = ROOT / "data/processed/player_scores_2023_2025.csv"
APP_CSV = ROOT / "data/app/players_scored.csv"

SEASONS = (2023, 2025)            # inclusive range v1 used; 2016-2022 are in the JSON but unused
TARGET_SEASON = 2025              # v1 trains on 2023-24 features -> 2025 score

# v1 scoring constants (not any real platform's rules; see AUDIT.md 2.1)
POINTS_PER_RUN, POINTS_PER_FOUR, POINTS_PER_SIX, POINTS_PER_WICKET = 1, 1, 2, 25
MATCH_THRESHOLD = 10.0            # "full confidence" after 10 matches

FEATURES = [
    "Total_Matches", "Total_Bt_Runs", "Total_Bt_Balls", "Total_Bw_Runs", "Total_Bw_Balls",
    "Total_Wkts", "Strike_Rate", "Economy_Rate", "SR_Bonus_Points", "Econ_Bonus_Points",
]


# ---------------------------------------------------------------- phase 1: flatten
def load_player_seasons() -> pd.DataFrame:
    """One row per (player, season). Per-match lists are kept as Python lists."""
    data = json.loads(RAW_JSON.read_text(encoding="utf-8"))
    rows = [{"player_name": p, "season": int(s), **stats} for p, seasons in data.items() for s, stats in seasons.items()]
    df = pd.DataFrame(rows)
    df = df[df["season"].between(*SEASONS)]
    df = df[df["Total_Matches"] > 0]
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].fillna(0)
    # v1 title-cases names, which mangles initials ("RG Sharma" -> "Rg Sharma"). Kept for v1 parity.
    df["player_name"] = df["player_name"].str.strip().str.title()
    return df


# ---------------------------------------------------------------- phase 2: score
def score(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per player over the given rows and compute v1's Enhanced_FPS."""
    df = df.copy()
    df["Total_4s"] = df["Bt_four"].apply(sum)
    df["Total_6s"] = df["Bt_six"].apply(sum)
    df["Total_Wkts"] = df["Bw_Wickets"].apply(sum)
    agg = df.groupby("player_name", as_index=False).agg({
        "Total_Matches": "sum", "Total_Bt_Runs": "sum", "Total_Bt_Balls": "sum",
        "Total_Bw_Runs": "sum", "Total_Bw_Balls": "sum",
        "Total_4s": "sum", "Total_6s": "sum", "Total_Wkts": "sum",
    })
    agg["Batting_Points"] = agg["Total_Bt_Runs"] * POINTS_PER_RUN + agg["Total_4s"] * POINTS_PER_FOUR + agg["Total_6s"] * POINTS_PER_SIX
    agg["Bowling_Points"] = agg["Total_Wkts"] * POINTS_PER_WICKET
    agg["Total_Fantasy_Points"] = agg["Batting_Points"] + agg["Bowling_Points"]
    agg["FPS_per_Match"] = np.where(agg["Total_Matches"] > 0, agg["Total_Fantasy_Points"] / agg["Total_Matches"], 0)
    agg["Match_Factor"] = np.clip(agg["Total_Matches"] / MATCH_THRESHOLD, 0, 1)
    agg["Adjusted_FPS"] = agg["FPS_per_Match"] * agg["Match_Factor"]

    sr = np.where(agg["Total_Bt_Balls"] > 0, agg["Total_Bt_Runs"] / agg["Total_Bt_Balls"] * 100, 0)
    agg["Strike_Rate"] = sr
    agg["SR_Bonus_Points"] = np.select(
        [sr > 170, (sr >= 150) & (sr <= 170), (sr >= 130) & (sr < 150), (sr >= 60) & (sr < 70), (sr >= 50) & (sr < 60), sr < 50],
        [6, 4, 2, -2, -4, -6], default=0)

    econ = pd.Series(np.where(agg["Total_Bw_Balls"] > 0, agg["Total_Bw_Runs"] / (agg["Total_Bw_Balls"] / 6), np.nan))
    agg["Economy_Rate"] = econ
    agg["Econ_Bonus_Points"] = np.select(
        [econ.isna(), econ < 5, (econ >= 5) & (econ < 6), (econ >= 6) & (econ < 7),
         (econ >= 10) & (econ < 11), (econ >= 11) & (econ < 12), econ >= 12],
        [0, 6, 4, 2, -2, -4, -6], default=0)

    agg["Enhanced_FPS"] = agg["Adjusted_FPS"] + agg["SR_Bonus_Points"] + agg["Econ_Bonus_Points"]
    return agg


# ---------------------------------------------------------------- phase 3: model
def train_model(seasons: pd.DataFrame) -> RandomForestRegressor:
    hist = score(seasons[seasons["season"] < TARGET_SEASON])
    target = score(seasons[seasons["season"] == TARGET_SEASON])
    train = hist.merge(target[["player_name", "Enhanced_FPS"]], on="player_name", suffixes=("", "_target"))
    X, y = train[FEATURES].fillna(0), train["Enhanced_FPS_target"]
    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
    # In-sample R2 only. It is NOT a measure of predictive skill (AUDIT.md 3.1).
    print(f"  trained on {len(X)} players; in-sample R2 = {r2_score(y, rf.predict(X)):.3f} (not a test score)")
    return rf


def main() -> None:
    print("1/4 flatten raw JSON")
    seasons = load_player_seasons()
    seasons.to_csv(SEASONS_CSV, index=False, lineterminator="\n")

    print("2/4 score 2023-2025 aggregate")
    scores = score(seasons)
    scores.to_csv(SCORES_CSV, index=False, lineterminator="\n")

    print("3/4 train v1 Random Forest")
    rf = train_model(seasons)

    print("4/4 attach roles/teams and predict")
    ref = pd.read_csv(ROLES_TEAMS)
    app = scores.merge(ref, on="player_name", how="inner")   # inner join drops 9 players, incl. R Ashwin (AUDIT.md 2.4)
    # v1 serves predictions from the 2023-25 aggregate, while training used 2023-24 (AUDIT.md 3.2)
    app["Predicted_Fantasy_Score"] = rf.predict(app[FEATURES].fillna(0))
    app = app.sort_values("Predicted_Fantasy_Score", ascending=False, kind="stable")
    app.to_csv(APP_CSV, index=False, lineterminator="\n")
    print(f"done: {len(app)} players -> {APP_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
