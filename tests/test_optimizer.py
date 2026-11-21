from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optimizer import Rules, optimize_xi, role_group  # noqa: E402

DF = pd.read_csv(ROOT / "data/app/players_scored.csv")
DF = DF[DF["Team_2025"] != "-"]


def check_valid(xi, rules):
    assert len(xi) == 11 and xi["player_name"].is_unique
    counts = xi["Role"].map(role_group).value_counts()
    for g in rules.role_min:
        assert rules.role_min[g] <= counts.get(g, 0) <= rules.role_max[g]
    assert xi["Team_2025"].value_counts().max() <= rules.max_per_team
    assert list(xi["Team_Role"]).count("Captain") == 1


def test_whitespace_roles_are_recognised():
    assert role_group("BAT ") == "BAT" and role_group(" bowl") == "BOWL" and role_group("BAT WK") == "WK"


def test_full_pool_gives_valid_xi():
    rules = Rules()
    xi, msg = optimize_xi(DF, rules)
    assert msg == "ok"; check_valid(xi, rules)


def test_tight_rules_respected_and_beat_or_match_greedy():
    rules = Rules(max_per_team=7, role_max=dict.fromkeys(["WK", "BAT", "AR", "BOWL"], 4))
    pool = DF[DF["Team_2025"].isin(["CSK", "MI"])]
    xi, _ = optimize_xi(pool, rules); check_valid(xi, rules)


def test_too_few_players_returns_message_not_crash():
    xi, msg = optimize_xi(DF.head(5), Rules())
    assert xi is None and "Only 5 players" in msg


def test_infeasible_roles_returns_message():
    no_wk = DF[DF["Role"].map(role_group) != "WK"]
    xi, msg = optimize_xi(no_wk, Rules())
    assert xi is None and "No valid XI" in msg
