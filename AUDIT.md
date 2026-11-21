# Audit of v1: IPL Fantasy Decision Support System

Audited commit `469eae4` ("version 2.0"), 22 Nov 2026. Every number below was
measured by re-running the code, not read off the notebooks. The evaluation
script is reproduced in Appendix A.

## 1. Verdict in four lines

1. **The model has no demonstrated skill.** v1 reports R² = 0.899, but that is the model scored on its own training rows. On a real forward test, it does *worse* than assuming each player repeats his last two seasons.
2. **The target, `Enhanced_FPS`, isn't a real fantasy score.** Per-match rules are applied once to a three-season total, fielding is missing, and bowlers are penalised for their batting. As a result the optimizer fills the XI with batters.
3. **The app serves predictions from inputs the model never saw in training.** It trains on 2-season totals and predicts from 3-season totals that include the target season.
4. **The engineering was fragile.** The model script crashed on a fresh clone, the roles/teams step had no code, there were no tests and no pinned versions, and the optimizer crashed when filters left fewer than 11 players.

Sections 2–4 explain each. Section 5 lists what this branch fixed and what is left for v2 (see `PLAN.md`).

## 2. Data

### 2.1 The target (`Enhanced_FPS`) is not well defined

`Enhanced_FPS = (points per match × min(matches/10, 1)) + SR bonus + economy bonus`.

- **Per-match rules applied to aggregates.** Real platforms award the strike-rate and economy bonuses per innings. v1 computes one strike rate over three seasons and adds a single ±6. A player who scored 0 and 100 in two innings is treated like one who scored 50 twice.
- **Mixed units.** The base score is per match, and the bonus is one flat number. The bonus is worth about 15% for a 40-point batter and about 25% for a 25-point bowler.
- **Bowlers get batting penalties.** 53% of bowlers with 20+ matches lose points for a low batting strike rate, often from a handful of balls faced at No. 10. Dream11 exempts bowlers from strike-rate points and requires at least 10 balls faced.
- **Scoring is missing.** v1 has no catches, stumpings, run-outs, 30/50/100 milestones, ducks, maidens, LBW/bowled bonuses or wicket hauls. The source JSON has no fielding data, so this can't be fixed without a new source.
- **The shrinkage goes the wrong way.** `Match_Factor` multiplies a 5-match player's average by 0.5. That shrinks him toward *zero*, not toward a typical player, which ranks every newcomer as a poor pick.
- **Result:** among players with 20+ matches, the median predicted score is 38.0 for batters, 37.5 for keepers, 33.0 for all-rounders and **22.8 for bowlers**. JJ Bumrah's own v1 score is 40.5, and the model predicts 26.8.
- **The rules aren't tied to a platform or season.** v1 gives 25 per wicket, +1 per four and +2 per six, but doesn't say which platform's table that is or when. So "fantasy points" can't be checked against a real game.

### 2.2 Provenance

- **The source of `player_performance.json` is undocumented.** Its player names (`YBK Jaiswal`, `RG Sharma`) match Cricsheet's naming, so it was almost certainly derived from Cricsheet.
- **The README says "ball-by-ball". It isn't.** It holds per-match innings figures grouped by season, with no match IDs, dates, opponents or venues.
- **v1 used 3 of the 10 seasons available.** It discarded 2016–2022.

### 2.3 Quirks in the source

- **Super overs are merged into the main match.** 16 player-matches show 2 dismissals in one innings (KL Rahul twice in 2020; YBK Jaiswal, R Parag and MS Dhoni in 2025).
- **"Matches" counts only games where the player batted or bowled.**
- **An abandoned 2025 match seems to be counted.** P Simran Singh and Priyansh Arya each show 18 matches. PBKS's 14 league games plus 3 playoffs make 17. The most likely extra is the PBKS v DC match abandoned mid-game in Dharamsala and later replayed, in which both openers batted.
- **`str.title()` mangles names.** `RG Sharma` becomes `Rg Sharma` and `MR Marsh` becomes `Mr Marsh`. That's harmless for joins inside v1, but breaks matching against any outside list.

### 2.4 Roles and teams: an undocumented, lossy join

- **No committed code produces the roles/teams file.** `phase3_with_roles_teams.csv` came out of a step that was never saved. Its leftover columns (`clean_name_x`, `clean_name_y`) and the unused `rapidfuzz` in `requirements.txt` point to a fuzzy name match.
- **The join dropped 9 players.** Most had retired, but **R Ashwin played 2025 for CSK**.
- **7 players who did play 2025 were marked team `-`**, which the app hides: A Mhatre, C Sakariya, M Tiwari, MM Ali, Mujeeb Ur Rahman, PWA Mulder and SN Thakur. Several were mid-season replacements. The team list looks like auction squads, not the players who actually turned out.
- **15 players shown in the app didn't play a 2025 match.** They were squad members who never got a game.
- **Two roles had trailing spaces** (`"BAT "`, `"BOWL "`). The app's exact match dropped those players from the role tabs.

### 2.5 Freshness

- **The data stops at 2025.** IPL 2026 finished on 31 May 2026, and Cricsheet lists 1,243 IPL matches, which looks like a full season beyond 2025.

## 3. Model

### 3.1 Evaluation

| Setup | R² | MAE | Spearman ρ (ranking) |
|---|---|---|---|
| RF scored on its own 158 training rows (what v1 prints) | **0.899** | 4.46 | 0.949 |
| RF, 5-fold CV on the same 158 players | 0.295 | 11.79 | 0.546 |
| **Forward test:** RF trained 2022-23 → 2024, tested 2023-24 → 2025 | 0.130 | 12.72 | 0.427 |
| No model: "next season = past two seasons' score" | **0.154** | **12.39** | **0.525** |
| No model: "next season = last season's score" | 0.100 | 12.74 | 0.531 |
| Constant (training mean) | 0.000 | 13.70 | n/a |

- **The forward test is the honest one**: a model trained only on the past, scored on a season it never saw. The Random Forest loses to the no-model baseline on all three metrics.
- **The gap is probably real.** Over 2,000 bootstrap resamples, the baseline ranked players better in 94% of them. The 95% interval for the gap in ρ runs from −0.03 to +0.23, so this is "no evidence the model helps", not proof that it hurts.
- **Ranking (ρ) is the metric that matters here.** Picking a team is a ranking problem.
- **No baseline existed, so this was never visible.**
- **The earlier `legacy_version/phase3.ipynb` was worse still.** It used a random split, and its target is computed from its own input features, so its R² mostly measured how well a forest can reverse a formula.

### 3.2 Train/serve skew

- **Training and serving inputs differ.** The model trains on **2023–24 totals** (max 30 matches) to predict 2025, then the app feeds it **2023–25 totals** (max 46 matches).
- **The inputs are raw totals**, such as runs, balls and wickets, and a tree model can't extrapolate beyond the range it trained on.
- **So predictions are squashed.** They span −3.1 to 49.4, while the scores they stand in for span −12.0 to 56.8.
- **It's also unclear what's being predicted.** Serving inputs include 2025, so "Predicted Fantasy Score" is neither a 2025 forecast (the inputs include 2025) nor a 2026 forecast (the model was never trained to look one season ahead from three seasons of data).

### 3.3 Sample size and features

- **158 training rows from one season pairing.** That's too few to tune a 300-tree forest or to estimate its error.
- **The features are totals, so they largely measure matches played.** Feature importances: runs 0.28, balls faced 0.18, strike rate 0.17, matches 0.11. Rates per match or per ball, plus role, would separate skill from availability.

### 3.4 Team selection

- **Never evaluated.** Not against random valid XIs, nor against "pick last season's top 11".
- **Greedy pick ignored the rules.** v1 took the best player per role, then the best remaining overall. It ignored role maximums and per-team caps, and its default XI had **9 batters/keepers, 1 all-rounder and 1 bowler**.
- **It crashed** (`ValueError: Length of values (11) does not match length of index (2)`) whenever filters left fewer than 11 players.
- **Captain and vice-captain** were simply the top two scores. That's reasonable, but not stated.

## 4. Code quality and engineering

| Issue | Where | Severity |
|---|---|---|
| Model script crashes on a fresh clone: it reads `final_clean_ipl_2023_2025.csv` from the root, but the file was moved to `legacy_version/` | `phase3_fixed_model.py` | Major |
| Script overwrites its own input file in place, so a failed run corrupts the app's data | `phase3_fixed_model.py` | Major |
| Roles/teams step has no code, so the app's data can't be rebuilt | missing | Major |
| Optimizer crashes when fewer than 11 players match the filters, and ignores role/team limits | `streamlit_app.py` | Major |
| No tests, no CI, unpinned dependencies; `rapidfuzz` listed but unused | repo | Medium |
| Logic duplicated across `phase2.ipynb`, `phase3_fixed_model.py` and the legacy notebook | repo | Medium |
| `phase4.ipynb` was 498 KB of embedded images and showed stale output (`"=== PHASE 4 ... Loaded 315 player records"`) from code no longer in the cell | notebook | Minor |
| Bare `except:`; unused `session_state`; a role sort immediately overwritten by a second sort; `SettingWithCopy` on `user_df` | various | Minor |
| Sliders allowed 100 matches and a 600 strike rate (data max 46 and 300); tiny samples (1 ball faced) are unfiltered | `streamlit_app.py` | Minor |
| README overclaims: "ball-by-ball", "prevents data leakage", "robust, real-world predictive power" | `README.md` | Medium (credibility) |

## 5. What this branch did

**Fixed now (no change to v1's numbers):**

- **Cleanup.** Removed 8 clutter files: the legacy folder, superseded notebooks, unused intermediates, and the crashing script.
- **One pipeline.** `pipeline/v1_pipeline.py` rebuilds every data file from the raw JSON in one command and reproduces the committed app data to within 1e-9. The roles/teams table is now an explicit, frozen reference file.
- **Data lock.** `data/MANIFEST.json` records the SHA-256, row count, producer and consumers of every data file. `pipeline/verify_data.py` fails on any changed or untracked file. CI runs the lock check, rebuilds, checks again, and runs the tests. Data files are exempt from line-ending conversion, so checksums match on Windows.
- **Optimizer.** It's now an exact integer program (`optimizer.py`) with role limits and a per-team cap. It returns a message instead of crashing.
- **App fixes.** Roles are normalised, sliders are bounded by the data, dead code is removed, and a visible caveat explains what the score is.
- **Dependencies and tests.** Dependencies are pinned (Python 3.12+), and 13 tests cover the builder, the optimizer and the app's former crash cases.
- **Cricsheet builder.** `pipeline/build_from_cricsheet.py` builds a per-player, per-match table for every season (2008–2026), with fielding, dates, venues, opponents and results, and excludes super overs. Its `--legacy-check` option reconciles against v1's JSON. It has been tested on synthetic matches, and still needs the real `ipl_json.zip` (see `PLAN.md`, Stage 1).

**Deliberately not fixed here (v2 work, see `PLAN.md`):** the scoring formula, the model, the train/serve skew, and the 7 hidden 2025 players and missing R Ashwin. These are modelling decisions to make and defend yourself, and v1 has to stay reproducible as the baseline v2 must beat.

## Appendix A: reproduce the evaluation

`python -I scripts/evaluate_v1.py .` prints every number in section 3, including the bootstrap. It reuses v1's own feature code unchanged, so the comparison is like for like.
