# Data: what exists, where it came from, what uses it

Every file in `data/` is listed in `MANIFEST.json` with its SHA-256 and row count.
`python pipeline/verify_data.py` fails if any file changes or a new file appears
without an entry. CI runs it on every push.

## Lineage

```
data/raw/player_performance.json ──┐
data/reference/player_roles_teams_2025.csv ──┤
                                   ▼
                     pipeline/v1_pipeline.py  (frozen v1)
                                   │
       ┌───────────────────────────┼─────────────────────────────┐
       ▼                           ▼                             ▼
processed/ipl_2023_2025_    processed/player_scores_     app/players_scored.csv ──► streamlit_app.py
player_seasons.csv          2023_2025.csv ──► notebooks/eda.ipynb

data/raw/ipl_json.zip (Cricsheet, 2008-2026) ──► pipeline/build_from_cricsheet.py ──► processed/player_matches.csv ──► v2
```

## Files

| File | What it is | Rows | Old name |
|---|---|---|---|
| `raw/player_performance.json` | v1 source. 486 players, seasons 2016–2025, per-match lists per season. Origin undocumented; names match Cricsheet's. | n/a | `player_performance.json` |
| `reference/player_roles_teams_2025.csv` | Role (`WK`/`BAT`/`AR`/`BOWL` as `BAT WK`, `BAT`, `AR`, `BOWL`) and 2025 franchise per player. Hand-curated; the build script was never committed. | 306 | columns of `phase3_with_roles_teams.csv` |
| `processed/ipl_2023_2025_player_seasons.csv` | One row per player-season, 2023–2025. | 618 | `legacy_version/final_clean_ipl_2023_2025.csv` |
| `processed/player_scores_2023_2025.csv` | v1 `Enhanced_FPS` over the 2023–25 aggregate. | 315 | `legacy_version/phase2.1_enhanced_fps_no_roles.csv` |
| `app/players_scored.csv` | Scores, roles, teams and v1 predictions. The only file the app reads. | 306 | `phase3_with_roles_teams.csv` |
| `raw/ipl_json.zip` | Cricsheet IPL ball-by-ball (not committed; download it). | n/a | new |
| `processed/player_matches.csv` | One row per player per match, all seasons, with fielding and match context. | built later | new |

## Deleted as clutter (still in git history before this change)

| File | Why it went |
|---|---|
| `legacy_version/phase3.ipynb` | Random-split model whose target is a formula of its own features (leakage). Superseded. |
| `legacy_version/phase3_predicted_fantasy_scores.csv` | Output of the leaky model above. Nothing read it. |
| `legacy_version/phase2_adjusted_player_fps.csv` | Intermediate that only phase 2 read back, immediately overwritten in memory. |
| `phase1.ipynb`, `phase2.ipynb`, `phase3_fixed_model.py` | Folded into `pipeline/v1_pipeline.py`. The model script also crashed on a fresh clone (its input had been moved into `legacy_version/`). |

## Known quirks in the v1 source

- **Super overs are merged into the match.** 16 player-matches show 2 dismissals in one innings (for example YBK Jaiswal and R Parag, 2025).
- **"Matches" means "batted or bowled".** Players who only fielded aren't counted.
- **There are no match IDs, dates, opponents, venues, or fielding stats.** So catches, stumpings and run-outs can't be scored, and nothing can be split by date.
- **A 2025 abandoned match is counted.** P Simran Singh and Priyansh Arya each show 18 matches.
- **Names are title-cased by v1, which mangles initials.** "RG Sharma" becomes "Rg Sharma". v2 should keep Cricsheet names and IDs.

## Licence

The Cricsheet data is under the Open Data Commons Attribution License. Credit "Cricsheet (cricsheet.org)" in the README and in any published report.
