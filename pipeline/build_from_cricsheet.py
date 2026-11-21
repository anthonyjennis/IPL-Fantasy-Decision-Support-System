"""Build a per-player, per-match table from Cricsheet's IPL ball-by-ball JSON.

This is the v2 data source. It replaces the hand-obtained player_performance.json
with a reproducible build from a public, licensed dataset, and it covers every
season Cricsheet has (2008 onward, including 2026).

Get the data (Open Data Commons Attribution License, credit "Cricsheet"):
    https://cricsheet.org/downloads/ipl_json.zip  ->  save to data/raw/ipl_json.zip

Then, from the repo root:
    python pipeline/build_from_cricsheet.py                    # writes data/processed/player_matches.csv
    python pipeline/build_from_cricsheet.py --legacy-check     # also compares against v1's JSON

Output: one row per player per match they were named in (XI or substitute). Super
overs are excluded, so a batter can be dismissed at most once per match.

Counting rules (documented so the numbers can be defended):
  balls_faced     every delivery to the batter except wides (no-balls count)
  balls_bowled    legal deliveries only (no wides, no no-balls)
  runs_conceded   batter runs + wides + no-balls (byes and leg-byes are not the bowler's)
  wickets         dismissals credited to the bowler: excludes run out, retired hurt,
                  retired out, obstructing the field
  maidens         overs of 6 legal balls, all by one bowler, conceding 0 runs
  catches         'caught' -> each listed fielder; 'caught and bowled' -> the bowler
  stumpings       'stumped' -> the listed fielder
  run_outs        'run out' -> each listed fielder (Cricsheet does not mark direct hits)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "data/raw/ipl_json.zip"
OUT_CSV = ROOT / "data/processed/player_matches.csv"
LEGACY_JSON = ROOT / "data/raw/player_performance.json"

NOT_BOWLER_WICKETS = {"run out", "retired hurt", "retired out", "obstructing the field", "retired not out"}

COLUMNS = [
    "match_id", "date", "season", "match_number", "stage", "venue", "city", "team", "opponent", "player", "player_id",
    "batted", "runs", "balls_faced", "fours", "sixes", "dismissed", "dismissal_kind", "batting_position",
    "balls_bowled", "runs_conceded", "wickets", "lbw_bowled_wickets", "maidens", "dot_balls_bowled",
    "catches", "stumpings", "run_outs", "won", "no_result",
]


def _blank(match_meta: dict, team: str, opponent: str, player: str, pid: str | None) -> dict:
    row = dict.fromkeys(COLUMNS, 0)
    row.update(match_meta, team=team, opponent=opponent, player=player, player_id=pid,
               batted=False, dismissed=False, dismissal_kind="", batting_position=None)
    return row


def parse_match(match_id: str, m: dict) -> list[dict]:
    info = m["info"]
    teams = info["teams"]
    date = info["dates"][0]
    outcome = info.get("outcome", {})
    event = info.get("event", {})
    meta = {
        "match_id": match_id, "date": date, "season": int(date[:4]),  # Cricsheet 'season' can be "2020/21"; use the year played
        "match_number": event.get("match_number"), "stage": event.get("stage", "League"),
        "venue": info.get("venue"), "city": info.get("city"),
        "no_result": "result" in outcome and outcome.get("result") == "no result",
    }
    registry = info.get("registry", {}).get("people", {})
    team_of: dict[str, str] = {}
    for t, names in info.get("players", {}).items():
        for n in names:
            team_of[n] = t
    other = {teams[0]: teams[1], teams[1]: teams[0]}
    rows: dict[str, dict] = {}

    def get(player: str, team_hint: str) -> dict:
        if player not in rows:
            team = team_of.get(player, team_hint)
            rows[player] = _blank(meta, team, other.get(team, ""), player, registry.get(player))
        return rows[player]

    for name, t in team_of.items():
        get(name, t)

    for inn in m.get("innings", []):
        if inn.get("super_over"):
            continue
        bat_team = inn["team"]
        bowl_team = other[bat_team]
        order: list[str] = []
        over_stats: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])  # (over, bowler) -> [legal, runs]
        over_bowlers: dict[int, set[str]] = defaultdict(set)
        for over in inn.get("overs", []):
            o = over["over"]
            for d in over["deliveries"]:
                batter, bowler = d["batter"], d["bowler"]
                for p in (batter, d["non_striker"]):
                    if p not in order:
                        order.append(p)
                extras = d.get("extras", {})
                wide, noball = extras.get("wides", 0), extras.get("noballs", 0)
                bat_runs = d["runs"]["batter"]

                b = get(batter, bat_team)
                b["batted"] = True
                b["runs"] += bat_runs
                if not wide:
                    b["balls_faced"] += 1
                if bat_runs == 4 and not d["runs"].get("non_boundary"):
                    b["fours"] += 1
                if bat_runs == 6 and not d["runs"].get("non_boundary"):
                    b["sixes"] += 1
                get(d["non_striker"], bat_team)["batted"] = True

                bw = get(bowler, bowl_team)
                legal = not wide and not noball
                conceded = bat_runs + wide + noball
                bw["runs_conceded"] += conceded
                if legal:
                    bw["balls_bowled"] += 1
                    if conceded == 0:
                        bw["dot_balls_bowled"] += 1
                over_bowlers[o].add(bowler)
                s = over_stats[(o, bowler)]
                s[0] += int(legal)
                s[1] += conceded

                for w in d.get("wickets", []):
                    out = get(w["player_out"], bat_team)
                    out["batted"] = True
                    if w["kind"] not in ("retired hurt", "retired not out"):
                        out["dismissed"] = True
                        out["dismissal_kind"] = w["kind"]
                    if w["kind"] not in NOT_BOWLER_WICKETS:
                        bw["wickets"] += 1
                        if w["kind"] in ("lbw", "bowled"):
                            bw["lbw_bowled_wickets"] += 1
                    fielders = [f["name"] for f in w.get("fielders", []) if "name" in f]
                    if w["kind"] == "caught":
                        for f in fielders:
                            get(f, bowl_team)["catches"] += 1
                    elif w["kind"] == "caught and bowled":
                        bw["catches"] += 1
                    elif w["kind"] == "stumped":
                        for f in fielders:
                            get(f, bowl_team)["stumpings"] += 1
                    elif w["kind"] == "run out":
                        for f in fielders:
                            get(f, bowl_team)["run_outs"] += 1
        for (o, bowler), (legal, runs) in over_stats.items():
            if legal == 6 and runs == 0 and over_bowlers[o] == {bowler}:
                rows[bowler]["maidens"] += 1
        for pos, p in enumerate(order, start=1):
            rows[p]["batting_position"] = pos

    winner = outcome.get("winner")
    for r in rows.values():
        r["won"] = winner == r["team"]
    return list(rows.values())


def iter_matches(source: Path):
    """Yield (match_id, match_dict) from a Cricsheet zip or a folder of JSON files."""
    if source.is_dir():
        for f in sorted(source.glob("*.json")):
            yield f.stem, json.loads(f.read_text(encoding="utf-8"))
        return
    with zipfile.ZipFile(source) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".json")):
            yield Path(name).stem, json.load(io.TextIOWrapper(z.open(name), encoding="utf-8"))


def build(source: Path) -> pd.DataFrame:
    rows = [r for mid, m in iter_matches(source) for r in parse_match(mid, m)]
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(["date", "match_id", "team", "player"], kind="stable").reset_index(drop=True)


def legacy_check(df: pd.DataFrame) -> pd.DataFrame:
    """Compare per-season totals with v1's player_performance.json, for the seasons both cover.

    v1 counted a match only if the player batted or bowled, and it included super overs,
    so small differences in Total_Matches and dismissals are expected. Large ones mean a bug.
    """
    legacy = json.loads(LEGACY_JSON.read_text(encoding="utf-8"))
    act = df[df["batted"] | (df["balls_bowled"] > 0) | (df["runs_conceded"] > 0)]
    g = act.groupby(["player", "season"]).agg(
        Total_Matches=("match_id", "nunique"), Total_Bt_Runs=("runs", "sum"), Total_Bt_Balls=("balls_faced", "sum"),
        Total_Bw_Runs=("runs_conceded", "sum"), Total_Bw_Balls=("balls_bowled", "sum"),
        fours=("fours", "sum"), sixes=("sixes", "sum"), wkts=("wickets", "sum"))
    out = []
    for (player, season), new in g.iterrows():
        old = legacy.get(player, {}).get(str(season))
        if old is None:
            continue
        o = {"Total_Matches": old["Total_Matches"], "Total_Bt_Runs": old["Total_Bt_Runs"], "Total_Bt_Balls": old["Total_Bt_Balls"],
             "Total_Bw_Runs": old["Total_Bw_Runs"], "Total_Bw_Balls": old["Total_Bw_Balls"],
             "fours": sum(old["Bt_four"]), "sixes": sum(old["Bt_six"]), "wkts": sum(old["Bw_Wickets"])}
        out.append({"player": player, "season": season, **{f"{k}_diff": int(new[k] - v) for k, v in o.items()}})
    rep = pd.DataFrame(out)
    legacy_pairs = sum(len(v) for v in legacy.values())
    print(f"legacy player-seasons: {legacy_pairs}; matched in Cricsheet build: {len(rep)}")
    for c in [c for c in rep.columns if c.endswith("_diff")]:
        exact = (rep[c] == 0).mean()
        print(f"  {c[:-5]:15s} exact match {exact:6.1%}   max |diff| {rep[c].abs().max()}")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=DEFAULT_ZIP, help="Cricsheet ipl_json.zip or a folder of match JSONs")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--legacy-check", action="store_true", help="compare totals with data/raw/player_performance.json")
    a = ap.parse_args(argv)
    if not a.source.exists():
        print(f"missing {a.source}. Download https://cricsheet.org/downloads/ipl_json.zip to data/raw/", file=sys.stderr)
        return 1
    df = build(a.source)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, lineterminator="\n")
    per_season = df.groupby("season")["match_id"].nunique()
    print(f"{df['match_id'].nunique()} matches, {len(df)} player-match rows -> {a.out}")
    print("matches per season:", per_season.to_dict())
    if a.legacy_check:
        legacy_check(df).to_csv(a.out.with_name("legacy_check.csv"), index=False, lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
