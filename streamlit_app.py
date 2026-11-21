from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from optimizer import ROLE_GROUPS, ROLE_ORDER, Rules, optimize_xi, role_group

DATA = Path(__file__).parent / "data/app/players_scored.csv"
SCORE = "Predicted_Fantasy_Score"
SHOW = ["player_name", "Team_2025", "Role", "Total_Matches", "Strike_Rate", "Economy_Rate", SCORE]


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    num = df.select_dtypes(include=np.number).columns
    df[num] = df[num].fillna(0)
    df["Role"] = df["Role"].str.strip().str.upper()
    return df[df["Team_2025"] != "-"].reset_index(drop=True)


df = load_data()
st.title("🏏 Fantasy Cricket Decision Support System")
st.caption("IPL 2023–2025 player data | Interactive team builder")
st.info("Scores come from the v1 model, which has not yet been shown to beat a simple "
        "'last two seasons' baseline. Treat them as a ranking aid, not a forecast. Details: AUDIT.md in the repo.")

# ---------------------------------------------------------------- filters
st.sidebar.header("⚙️ Filter Controls")
teams = sorted(df["Team_2025"].unique())
selected_teams = st.sidebar.multiselect("Select team(s) for a match", teams, max_selections=2)
min_matches = st.sidebar.slider("Minimum matches played (2023–25)", 0, int(df["Total_Matches"].max()), 0)
min_sr = st.sidebar.slider("Minimum batting strike rate", 0.0, float(np.ceil(df["Strike_Rate"].max())), 0.0)
max_econ = st.sidebar.slider("Maximum bowling economy (non-bowlers always pass)", 0.0, float(np.ceil(df["Economy_Rate"].max())),
                             float(np.ceil(df["Economy_Rate"].max())))

st.sidebar.header("📋 Team rules")
max_per_team = st.sidebar.number_input("Max players from one team", 1, 11, Rules().max_per_team)
role_max = st.sidebar.number_input("Max players per role", 1, 8, 8)

filtered = df[(df["Total_Matches"] >= min_matches) & (df["Strike_Rate"] >= min_sr) & (df["Economy_Rate"] <= max_econ)]
if selected_teams:
    filtered = filtered[filtered["Team_2025"].isin(selected_teams)]
filtered = filtered.sort_values(SCORE, ascending=False)

st.subheader(f"🎯 Filtered Players ({len(filtered)} results)")
st.dataframe(filtered[SHOW], hide_index=True)

st.subheader("📊 Top 10 Players by Predicted Fantasy Score")
top10 = filtered.head(10)
if len(top10):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(y="player_name", x=SCORE, data=top10, hue="Team_2025", dodge=False, palette="crest", ax=ax)
    ax.set_xlabel("Predicted Fantasy Score"); ax.set_ylabel("Player")
    ax.legend(title="Team", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.write("No players match the filters.")

# ---------------------------------------------------------------- team builder
st.subheader("🏆 Build & Optimize Your XI")
labels = {"WK": "🧤 Wicketkeepers", "BAT": "🏏 Batters", "AR": "🌀 All-Rounders", "BOWL": "🎯 Bowlers"}
tabs = st.tabs([labels[g] for g in ROLE_GROUPS] + ["Summary / Optimize"])
groups = filtered["Role"].map(role_group)
chosen: list[str] = []
for tab, g in zip(tabs, ROLE_GROUPS):
    with tab:
        sub = filtered[groups == g]
        st.write(f"Available: {len(sub)}")
        chosen += st.multiselect(f"Select {labels[g][2:]}", sub["player_name"].tolist(), key=f"pick_{g}")
        st.dataframe(sub[["player_name", "Team_2025", SCORE]], hide_index=True)

with tabs[-1]:
    rules = Rules(max_per_team=int(max_per_team), role_max=dict.fromkeys(ROLE_GROUPS, int(role_max)))
    if st.button("🚀 Optimize XI Automatically"):
        xi, msg = optimize_xi(filtered, rules)
        if xi is None:
            st.warning(msg)
        else:
            st.success("✅ Best valid XI under the team rules")
            st.dataframe(xi[["player_name", "Role", "Team_2025", SCORE, "Team_Role"]], hide_index=True)
            st.info(f"**Total: {xi[SCORE].sum():.2f} | Avg: {xi[SCORE].mean():.2f}**")
    else:
        mine = filtered[filtered["player_name"].isin(chosen)].copy()
        if len(mine) > 11:
            st.error(f"⚠️ You selected {len(mine)} players. The limit is 11; deselect {len(mine) - 11}.")
        elif len(mine):
            mine["order"] = mine["Role"].map(role_group).map(ROLE_ORDER)
            st.write(f"**You selected {len(mine)} players.**")
            st.dataframe(mine.sort_values(["order", SCORE], ascending=[True, False])[["player_name", "Role", "Team_2025", SCORE]],
                         hide_index=True)
            st.info(f"**Total: {mine[SCORE].sum():.2f} | Avg: {mine[SCORE].mean():.2f}**")
        else:
            st.warning("Select up to 11 players in the tabs above, or click Optimize XI.")
