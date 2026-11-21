"""Pick the highest-scoring valid fantasy XI with an exact integer program.

v1 picked the best player per role and then the next best overall, which ignored
role limits and team limits (it could return 9 batters) and crashed when fewer than
11 players were available. This solves the actual selection problem instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

ROLE_GROUPS = {"WK": ["BAT WK", "WK"], "BAT": ["BAT", "BATTER"], "AR": ["AR", "ALL-ROUNDER"], "BOWL": ["BOWL", "BOWLER"]}
ROLE_ORDER = {r: i for i, r in enumerate(ROLE_GROUPS)}


@dataclass
class Rules:
    """Team-composition rules. Defaults follow Dream11's current 1-8 per role; check your platform."""
    squad_size: int = 11
    role_min: dict = field(default_factory=lambda: dict.fromkeys(ROLE_GROUPS, 1))
    role_max: dict = field(default_factory=lambda: dict.fromkeys(ROLE_GROUPS, 8))
    max_per_team: int = 10


def role_group(role: str) -> str | None:
    role = str(role).strip().upper()
    return next((g for g, names in ROLE_GROUPS.items() if role in names), None)


def optimize_xi(players: pd.DataFrame, rules: Rules = Rules(), score_col: str = "Predicted_Fantasy_Score",
                team_col: str = "Team_2025") -> tuple[pd.DataFrame | None, str]:
    """Return (xi, message). xi is None when no valid XI exists under the rules."""
    df = players.copy()
    df["group"] = df["Role"].map(role_group)
    df = df.dropna(subset=["group"]).reset_index(drop=True)
    n = len(df)
    if n < rules.squad_size:
        return None, f"Only {n} players match the filters; {rules.squad_size} are needed. Loosen the filters."

    rows, lo, hi = [np.ones(n)], [rules.squad_size], [rules.squad_size]
    for g in ROLE_GROUPS:
        rows.append((df["group"] == g).to_numpy(float)); lo.append(rules.role_min[g]); hi.append(rules.role_max[g])
    for t in df[team_col].unique():
        rows.append((df[team_col] == t).to_numpy(float)); lo.append(0); hi.append(rules.max_per_team)

    res = milp(c=-df[score_col].to_numpy(float), integrality=np.ones(n),
               bounds=Bounds(0, 1), constraints=LinearConstraint(np.vstack(rows), lo, hi))
    if not res.success:
        have = df["group"].value_counts().to_dict()
        return None, f"No valid XI under these rules with the filtered players (available by role: {have})."

    xi = df[res.x > 0.5].sort_values(score_col, ascending=False).reset_index(drop=True)
    xi["Team_Role"] = ["Captain", "Vice Captain"] + ["Player"] * (len(xi) - 2)
    xi = xi.sort_values(["group", score_col], key=lambda s: s.map(ROLE_ORDER) if s.name == "group" else -s).drop(columns="group")
    return xi.reset_index(drop=True), "ok"
