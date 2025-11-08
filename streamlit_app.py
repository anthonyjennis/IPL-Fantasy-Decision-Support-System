import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    df = pd.read_csv("phase3_with_roles_teams.csv")
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df = df[df["Team_2025"] != "-"]
    return df

df = load_data()
st.title("🏏 Fantasy Cricket Decision Support System")
st.caption("Built on IPL 2025 Player Data | Interactive Team Builder")

st.sidebar.header("⚙️ Filter Controls")

#Filtering options
teams = sorted(df["Team_2025"].dropna().unique().tolist())
selected_teams = st.sidebar.multiselect("Select Team(s)", teams, max_selections=2)

min_matches = int(st.sidebar.slider("Minimum matches played", 0, 20, 5))
min_sr = float(st.sidebar.slider("Minimum batting strike rate", 50.0, 200.0, 100.0))
max_econ = float(st.sidebar.slider("Maximum bowling economy rate", 4.0, 15.0, 10.0))

filtered_df = df[
    (df["Total_Matches"] >= min_matches)
    & (df["Strike_Rate"] >= min_sr)
    & (df["Economy_Rate"] <= max_econ)
]

if selected_teams:
    filtered_df = filtered_df[filtered_df["Team_2025"].isin(selected_teams)]

st.subheader(f"🎯 Filtered Players ({len(filtered_df)} results)")
st.dataframe(
    filtered_df[
        [
            "player_name",
            "Team_2025",
            "Role",
            "Total_Matches",
            "Strike_Rate",
            "Economy_Rate",
            "Predicted_Fantasy_Score",
        ]
    ].sort_values("Predicted_Fantasy_Score", ascending=False)
)

st.subheader("📊 Top 10 Players by Predicted Fantasy Score")

top10 = filtered_df.sort_values("Predicted_Fantasy_Score", ascending=False).head(10)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    y="player_name",
    x="Predicted_Fantasy_Score",
    data=top10,
    hue="Team_2025",
    dodge=False,
    palette="crest",
    ax=ax,
)
ax.set_title("Top 10 Players – Combined Teams")
ax.set_xlabel("Predicted Fantasy Score")
ax.set_ylabel("Player")
st.pyplot(fig)

st.subheader("🏆 Build & Optimize Your XI")

role_map = {
    "WK": ["BAT WK", "WK"],
    "BAT": ["BAT", "BATTER"],
    "AR": ["AR", "ALL-ROUNDER"],
    "BOWL": ["BOWL", "BOWLER"],
}

tab_wk, tab_bat, tab_ar, tab_bowl, tab_summary = st.tabs(
    ["🧤 Wicketkeepers", "🏏 Batters", "🌀 All-Rounders", "🎯 Bowlers", "Summary / Optimize"]
)

# persistent selection across tabs
if "user_selection" not in st.session_state:
    st.session_state.user_selection = []

# helper function
def update_selection(role_df, label):
    chosen = st.multiselect(label, role_df["player_name"].tolist())
    return chosen

with tab_wk:
    wk_df = filtered_df[filtered_df["Role"].str.upper().isin(role_map["WK"])]
    st.write(f"Available Wicketkeepers: {len(wk_df)}")
    chosen_wk = update_selection(wk_df, "Select Wicketkeepers")
    st.dataframe(wk_df[["player_name", "Team_2025", "Predicted_Fantasy_Score"]])

with tab_bat:
    bat_df = filtered_df[filtered_df["Role"].str.upper().isin(role_map["BAT"])]
    st.write(f"Available Batters: {len(bat_df)}")
    chosen_bat = update_selection(bat_df, "Select Batters")
    st.dataframe(bat_df[["player_name", "Team_2025", "Predicted_Fantasy_Score"]])

with tab_ar:
    ar_df = filtered_df[filtered_df["Role"].str.upper().isin(role_map["AR"])]
    st.write(f"Available All-Rounders: {len(ar_df)}")
    chosen_ar = update_selection(ar_df, "Select All-Rounders")
    st.dataframe(ar_df[["player_name", "Team_2025", "Predicted_Fantasy_Score"]])

with tab_bowl:
    bowl_df = filtered_df[filtered_df["Role"].str.upper().isin(role_map["BOWL"])]
    st.write(f"Available Bowlers: {len(bowl_df)}")
    chosen_bowl = update_selection(bowl_df, "Select Bowlers")
    st.dataframe(bowl_df[["player_name", "Team_2025", "Predicted_Fantasy_Score"]])

with tab_summary:
    st.write("Optimal XI Summary")

    # combine selections
    user_selection = list(set(chosen_wk + chosen_bat + chosen_ar + chosen_bowl))
    user_df = filtered_df[filtered_df["player_name"].isin(user_selection)]

    # Restrict to 11 players
    if len(user_selection) > 11:
        st.error(f"⚠️ You have selected {len(user_selection)} players. Limit is 11. Please deselect some players.")
        user_df = user_df.head(11)  # truncate extra players

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**You selected {len(user_df)} players.**")
    with col2:
        optimize = st.button("🚀 Optimize XI Automatically")

    if optimize:
        # Optimization Logic
        auto_team = pd.DataFrame()
        for role, keywords in role_map.items():
            sub = filtered_df[filtered_df["Role"].str.upper().isin(keywords)]
            if len(sub) > 0:
                best = sub.sort_values("Predicted_Fantasy_Score", ascending=False).head(1)
                auto_team = pd.concat([auto_team, best])

        if len(auto_team) < 11:
            remaining = filtered_df[
                ~filtered_df["player_name"].isin(auto_team["player_name"])
            ]
            add = remaining.sort_values("Predicted_Fantasy_Score", ascending=False).head(
                11 - len(auto_team)
            )
            auto_team = pd.concat([auto_team, add])

        # sort by role order
        role_priority = {"WK": 1, "BAT": 2, "AR": 3, "BOWL": 4}
        auto_team["role_order"] = auto_team["Role"].apply(
            lambda r: next((role_priority[k] for k in role_priority if any(k in r.upper() for k in role_map[k])), 99)
        )
        auto_team = auto_team.sort_values("role_order").drop(columns=["role_order"])

        auto_team = auto_team.sort_values("Predicted_Fantasy_Score", ascending=False).reset_index(drop=True)
        auto_team["Team_Role"] = [
            "Captain" if i == 0 else "Vice Captain" if i == 1 else "Player"
            for i in range(11)
        ]

        st.success("✅ Optimized XI Generated!")
        st.dataframe(
            auto_team[
                [
                    "player_name",
                    "Role",
                    "Team_2025",
                    "Predicted_Fantasy_Score",
                    "Team_Role",
                ]
            ]
        )

        total_score = auto_team["Predicted_Fantasy_Score"].sum()
        st.info(f"**Total Score: {total_score:.2f} | Avg: {total_score/11:.2f}**")

    else:
        # Manual team summary
        if len(user_df) > 0:
            role_priority = {"WK": 1, "BAT": 2, "AR": 3, "BOWL": 4}
            user_df["role_order"] = user_df["Role"].apply(
                lambda r: next((role_priority[k] for k in role_priority if any(k in r.upper() for k in role_map[k])), 99)
            )
            user_df = user_df.sort_values("role_order").drop(columns=["role_order"])

            st.dataframe(
                user_df[
                    [
                        "player_name",
                        "Role",
                        "Team_2025",
                        "Predicted_Fantasy_Score",
                    ]
                ]
            )

            total_score = user_df["Predicted_Fantasy_Score"].sum()
            st.info(f"**Total Score: {total_score:.2f} | Avg: {total_score/len(user_df):.2f}**")
        else:
            st.warning("Select up to 11 players in the tabs above or click Optimize XI.")