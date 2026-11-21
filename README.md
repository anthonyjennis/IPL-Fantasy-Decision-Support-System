# 🏏 IPL Fantasy Decision Support System 📊

An analytics dashboard that ranks IPL players for fantasy cricket and builds a valid fantasy XI.

### Live App
👉 **[View Dashboard on Streamlit Cloud](https://ipl-fantasy.streamlit.app/)**

---

## Status: v1 audited, v2 in progress

v1 combined a custom fantasy score (`Enhanced_FPS`) built from 2023–2025 per-match data with a Random Forest that predicted it.

An audit (**[AUDIT.md](AUDIT.md)**) found that the model's reported R² of 0.899 was measured on its own training rows. On a true forward test, a no-model baseline ("each player repeats his last two seasons") ranked players better. Here is the comparison (ρ is the Spearman rank correlation; higher means better ranking):

| Forward test: train ≤2024, predict 2025 | R² | Spearman ρ |
|---|---|---|
| v1 Random Forest | 0.130 | 0.427 |
| Baseline: past two seasons | 0.154 | 0.525 |

v2 (**[PLAN.md](PLAN.md)**) will make these changes:
- rebuild the data from Cricsheet ball-by-ball files through 2026
- score players with a real, versioned fantasy rule set
- test models with a rolling, season-by-season backtest against baselines

Until then, the dashboard runs on v1's numbers, with a caveat shown in the app.

## Project layout

```
streamlit_app.py            dashboard (reads data/app/players_scored.csv only)
optimizer.py                exact best-XI selection under role and team limits
pipeline/
  v1_pipeline.py            frozen v1: raw JSON -> every data file, one command
  build_from_cricsheet.py   v2 data: Cricsheet IPL JSON -> one row per player per match
  verify_data.py            data lock: checks every file against data/MANIFEST.json
data/
  DATA.md                   what each file is, where it came from, what uses it
  MANIFEST.json             SHA-256, row count, producer and consumers for every data file
  raw/ reference/ processed/ app/
scripts/evaluate_v1.py      reproduces every number in AUDIT.md
notebooks/eda.ipynb         exploratory charts
tests/                      builder, optimizer and app smoke tests
```

## Run locally (Python 3.12+)

```bash
git clone https://github.com/anthonyjennis/IPL-Fantasy-Decision-Support-System.git
cd IPL-Fantasy-Decision-Support-System
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt

python pipeline/verify_data.py     # data lock holds
python pipeline/v1_pipeline.py     # rebuild all v1 data (identical output)
python -m pytest -q tests          # tests
streamlit run streamlit_app.py     # dashboard
```

### Building v2 data (includes 2026)

1. Download [`ipl_json.zip`](https://cricsheet.org/downloads/ipl_json.zip) into `data/raw/`.
2. Run:

```bash
python pipeline/build_from_cricsheet.py --legacy-check
python pipeline/verify_data.py --update   # record the new checksums, then commit
```

## Data lock

Every file in `data/` is recorded in `data/MANIFEST.json`, along with its checksum and the script that produced it. `verify_data.py` fails if a file changes or an unrecorded file appears. CI rebuilds the data on every push and checks that the output is byte-for-byte identical.

## Credits

Match data: [Cricsheet](https://cricsheet.org), Open Data Commons Attribution License. v1's source JSON was derived from an undocumented download; see `data/DATA.md`.
