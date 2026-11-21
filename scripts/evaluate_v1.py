"""Honest evaluation of the v1 model framing (AUDIT.md section 3).

Reuses v1's own scoring code from pipeline/v1_pipeline.py, so the comparison is like
for like. Usage from the repo root:  python -I scripts/evaluate_v1.py .
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(REPO / "pipeline"))
import v1_pipeline as v1  # noqa: E402

data = json.loads((REPO / "data/raw/player_performance.json").read_text(encoding="utf-8"))
raw = pd.DataFrame([{"player_name": p.strip().title(), "season": int(s), **st} for p, ss in data.items() for s, st in ss.items()])
raw = raw[raw.Total_Matches > 0]
F = v1.FEATURES


def pair(feat_seasons, tgt):
    X = v1.score(raw[raw.season.isin(feat_seasons)])
    y = v1.score(raw[raw.season == tgt])[["player_name", "Enhanced_FPS"]]
    return X.merge(y, on="player_name", suffixes=("", "_target"))


def rf():
    return RandomForestRegressor(n_estimators=300, random_state=42)


def row(name, y, p):
    rho = spearmanr(y, p).statistic if np.std(p) > 0 else float("nan")
    print(f"{name:62s} R2={r2_score(y, p):6.3f}  MAE={mean_absolute_error(y, p):6.2f}  rho={rho:6.3f}")


tr = pair([2023, 2024], 2025)
m0 = rf().fit(tr[F].fillna(0), tr.Enhanced_FPS_target)
row("RF scored on its own training rows (what v1 prints)", tr.Enhanced_FPS_target, m0.predict(tr[F].fillna(0)))

cv = np.zeros(len(tr))
for a, b in KFold(5, shuffle=True, random_state=0).split(tr):
    cv[b] = rf().fit(tr.iloc[a][F].fillna(0), tr.iloc[a].Enhanced_FPS_target).predict(tr.iloc[b][F].fillna(0))
row("RF, 5-fold CV on the same players", tr.Enhanced_FPS_target, cv)

trn, tst = pair([2022, 2023], 2024), pair([2023, 2024], 2025)
m = rf().fit(trn[F].fillna(0), trn.Enhanced_FPS_target)
y, prf, naive = tst.Enhanced_FPS_target.values, m.predict(tst[F].fillna(0)), tst.Enhanced_FPS.values
row("Forward: RF trained 2022-23->2024, tested 2023-24->2025", y, prf)
row("Baseline: past two seasons' score", y, naive)
last = v1.score(raw[raw.season == 2024]).set_index("player_name")["Enhanced_FPS"]
row("Baseline: last season's score", y, tst.player_name.map(last).fillna(tst.Enhanced_FPS).values)
row("Constant: training mean", y, np.full(len(y), trn.Enhanced_FPS_target.mean()))

rng = np.random.default_rng(0)
d = []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    d.append(spearmanr(y[i], naive[i]).statistic - spearmanr(y[i], prf[i]).statistic)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"rho(baseline) - rho(RF): 95% CI [{lo:.3f}, {hi:.3f}], baseline better in {np.mean(np.array(d) > 0):.0%} of resamples")
print(f"train rows {len(tr)} | forward train/test {len(trn)}/{len(tst)} | max matches train {tr.Total_Matches.max()} vs served "
      f"{pd.read_csv(REPO / 'data/app/players_scored.csv').Total_Matches.max()}")
