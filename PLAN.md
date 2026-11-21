# Plan: v1 → v2

The stages run in order. Each one has a **gate**: don't start the next stage until it passes.
Findings referenced as "AUDIT x.y" are in `AUDIT.md`.

**Decision to log first.** The November plan proposed replacing this project with a
ball-by-ball win-probability model. You now want to extend this one. Both can
share `pipeline/build_from_cricsheet.py`.

My recommendation: make this project's v2 the ML flagship. It needs the same rigour
(leakage-safe splits, baselines, uncertainty, a drift case study around the Impact
Player rule from 2023), and it already has users and a dashboard. Win probability
can follow as a second model on the same data. Whichever you choose, write it in the decision log.

---

## Stage 0: Land the cleanup (this branch)

- **Review the PR.** Before merging, tag the old main so v1 is preserved: `git tag v1-original 469eae4 && git push --tags`.
- **Set Python 3.12 on Streamlit Community Cloud.** Go to the app's settings, then Python version, then 3.12 or newer. The pinned packages need it. Redeploy and check that the live URL loads.
- **Gate:** CI is green, the live app loads, and "Optimize XI" with "Minimum matches" set to 45 shows a warning instead of crashing.

## Stage 1: Data v2, with 2026 included

1. **Get the data.** Download `https://cricsheet.org/downloads/ipl_json.zip` to `data/raw/`. Cricsheet is blocked from my sandbox, so either attach the zip in chat and I'll run this stage, or run it yourself.
2. **Build and reconcile.** Run `python pipeline/build_from_cricsheet.py --legacy-check`. This prints matches per season and how closely the new build agrees with v1's JSON for 2016–2025.
   - Runs, balls and boundaries should match almost exactly.
   - `Total_Matches` and dismissals will differ slightly. v1 merged super overs into the main match and counted only games where a player batted or bowled (AUDIT 2.3).
   - Write down every other difference and explain it before going on. This reconciliation is the provenance story for the README.
3. **Lock it.** Run `python pipeline/verify_data.py --update` and commit `player_matches.csv` together with `MANIFEST.json`.
4. **Teams per season from the data.** For 2026, the team is the side a player actually played for in Cricsheet, which fixes the hidden replacements (AUDIT 2.4).
5. **Roles: decide the source and write it down.** Cricsheet has no roles. Options:
   - Copy the official 2026 squad lists.
   - Derive roles from behaviour: bowled at least N balls per match means bowler; has stumpings means keeper.
   - Do both, and report where they disagree.
6. **Gate:** 2026 shows the expected number of matches (about 74). `--legacy-check` differences are all explained. `verify_data.py` passes. Every 2026 player has a team and a role.

## Stage 2: A real target

1. **Pick one rule set and version it.** Use Dream11 T20 as it stood for the 2026 season, and save a copy of its points table with the date you read it.
2. **Score each player per match** from `player_matches.csv`:
   - batting: runs, boundaries, 30/50/100 milestones, ducks, and strike rate (minimum 10 balls, not for bowlers)
   - bowling: wickets, LBW/bowled bonus, hauls, maidens, and economy (minimum 2 overs)
   - fielding: catches, stumpings, run-outs
   - the playing-XI bonus
3. **Unit-test every rule** on hand-worked innings, the same way `tests/test_cricsheet_builder.py` does.
4. **Season target = mean points per match played.** Keep the per-match table too; Stage 3 can use it.
5. **Gate:** on a sample of 2026 matches, your per-match points match published Dream11 scores closely enough. Write down how close, and why any differences remain.

## Stage 3: A model that's tested honestly (you write this code)

1. **Frame it.** Predict a player's points per match next season from everything up to the end of this season. The 2027 forecast is trained on data up to 2026.
2. **Rolling-origin backtest.** For each season *t* from 2019 to 2026, train on seasons up to *t*−1 and test on *t*. Report each season separately, not just the mean.
3. **Model ladder.** Each rung must beat the one before it on ranking (Spearman ρ, and the top-30 hit rate), or you write down why not:
   1. last season's points per match
   2. a weighted average of recent seasons
   3. **shrinkage toward the role average.** This is the correct version of v1's `Match_Factor`: a player with few matches is pulled toward a typical player of his role, not toward zero.
   4. gradient-boosted trees on *rates* (per ball, per match, per over), role, age of form, and phase of innings if you add it
4. **Leakage rules, written into the code.** Features come only from seasons before the target season. Training and serving use the same window length (AUDIT 3.2). Squad and team membership come from the target season's squads, which are known before it starts.
5. **Uncertainty.** Bootstrap confidence intervals on every headline number, and paired comparisons between rungs.
6. **Drift.** Does a model trained up to 2022 go miscalibrated on 2023 and later, once the Impact Player rule arrived? That's your "data drift vs concept drift" interview answer.
7. **Gate:** the final model beats the strongest baseline in most backtest seasons, with intervals. If it doesn't, ship the baseline and say so. That's still a good result to write up.

## Stage 4: Check that the team picker adds value

1. Keep `optimizer.py`. Add credits only if you can find a reliable source.
2. Backtest on 2026 matches. For each match, build the XI from pre-match predictions and score it with real points. Compare against:
   - random valid XIs
   - "last season's top 11"
   - the best possible XI in hindsight (the ceiling)
3. **Gate:** report the mean points and percentile against random XIs, with intervals.

## Stage 5: Public dashboard

- **Stay on Streamlit Community Cloud.** It's free, it's already linked from the README, and it deploys from `main`.
- **Add an "About this model" page.** Include the data source with Cricsheet credit, the season the data runs to, the backtest table, and the limitations. Put the honest caveat right on the page.
- **Switch the app to 2026 data**, and later to the next season's squads once they're announced.
- **If cold starts annoy users,** Hugging Face Spaces (Streamlit SDK) is the fallback host. Don't move unless there's a reason.
- **Gate:** a friend can open the link on a phone, understand what the score means, and build an XI without help.

## Stage 6: Report and publish

1. **Technical write-up, in the repo** (README "Lessons from v1" plus `docs/report.md`). This is the version for recruiters and interviews. The story:
   - the in-sample R² of 0.899 that turned out to be no skill
   - the forward test and the baseline that won
   - the new data and target
   - the backtest
   - what still doesn't work
2. **Reader-facing piece on The Fifth Day** (thefifthday.substack.com). Something like "I built a fantasy model and a spreadsheet beat it; here's what I fixed". Use charts from the data, link the live dashboard, and credit Cricsheet. This is the data-backed cricket writing you identified as your edge.
3. **A LinkedIn post** linking both.
4. **Framing:** present it as analytics for fans and free-to-play fantasy, not a tool for paid contests. India banned online real-money games in 2025.
5. **Gate:** every number in both write-ups can be regenerated from a committed script.

---

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-11-22 | v1 frozen as the baseline; cleanup, data lock and app fixes landed | AUDIT.md |
| | This project's v2 as the ML flagship, or the win-probability model? | |
| | Fantasy rule set and version | |
| | Role source for 2026 | |
