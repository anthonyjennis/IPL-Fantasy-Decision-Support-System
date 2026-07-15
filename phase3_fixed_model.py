import pandas as pd
import numpy as np
import ast
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Function to parse list string and sum
def parse_and_sum(list_str):
    try:
        data = ast.literal_eval(list_str)
        if isinstance(data, list):
            return sum(pd.to_numeric(item, errors='coerce') for item in data)
        return 0
    except:
        return 0

# Helper to calculate features
def calc_features(df_subset):
    df_subset = df_subset.copy()
    df_subset['Total_4s'] = df_subset['Bt_four'].apply(parse_and_sum)
    df_subset['Total_6s'] = df_subset['Bt_six'].apply(parse_and_sum)
    df_subset['Total_Wkts'] = df_subset['Bw_Wickets'].apply(parse_and_sum)
    
    agg = df_subset.groupby('player_name', as_index=False).agg({
        'Total_Matches': 'sum', 'Total_Bt_Runs': 'sum', 'Total_Bt_Balls': 'sum',
        'Total_Bw_Runs': 'sum', 'Total_Bw_Balls': 'sum', 'Total_4s': 'sum',
        'Total_6s': 'sum', 'Total_Wkts': 'sum'
    })
    
    agg['Batting_Points'] = agg['Total_Bt_Runs'] + agg['Total_4s'] + (agg['Total_6s'] * 2)
    agg['Bowling_Points'] = agg['Total_Wkts'] * 25
    agg['Total_Fantasy_Points'] = agg['Batting_Points'] + agg['Bowling_Points']
    
    agg['FPS_per_Match'] = np.where(agg['Total_Matches'] > 0, agg['Total_Fantasy_Points'] / agg['Total_Matches'], 0)
    agg['Match_Factor'] = np.clip(agg['Total_Matches'] / 10.0, 0, 1)
    agg['Adjusted_FPS'] = agg['FPS_per_Match'] * agg['Match_Factor']
    
    agg["Strike_Rate"] = np.where(agg["Total_Bt_Balls"] > 0, (agg["Total_Bt_Runs"] / agg["Total_Bt_Balls"]) * 100, 0)
    agg["SR_Bonus_Points"] = np.select(
        [agg["Strike_Rate"] > 170, (agg["Strike_Rate"] >= 150) & (agg["Strike_Rate"] <= 170),
         (agg["Strike_Rate"] >= 130) & (agg["Strike_Rate"] < 150), (agg["Strike_Rate"] >= 60) & (agg["Strike_Rate"] < 70),
         (agg["Strike_Rate"] >= 50) & (agg["Strike_Rate"] < 60), agg["Strike_Rate"] < 50],
        [6, 4, 2, -2, -4, -6], default=0
    )
    
    agg["Economy_Rate"] = np.where(agg["Total_Bw_Balls"] > 0, agg["Total_Bw_Runs"] / (agg["Total_Bw_Balls"] / 6), np.nan)
    agg["Econ_Bonus_Points"] = np.select(
        [agg["Economy_Rate"].isna(), agg["Economy_Rate"] < 5, (agg["Economy_Rate"] >= 5) & (agg["Economy_Rate"] < 6),
         (agg["Economy_Rate"] >= 6) & (agg["Economy_Rate"] < 7), (agg["Economy_Rate"] >= 10) & (agg["Economy_Rate"] < 11),
         (agg["Economy_Rate"] >= 11) & (agg["Economy_Rate"] < 12), agg["Economy_Rate"] >= 12],
        [0, 6, 4, 2, -2, -4, -6], default=0
    )
    
    agg["Enhanced_FPS"] = agg["Adjusted_FPS"] + agg["SR_Bonus_Points"] + agg["Econ_Bonus_Points"]
    return agg

print("Loading data...")
raw_df = pd.read_csv("final_clean_ipl_2023_2025.csv")

# Create historical (2023-2024) and target (2025) datasets
historical_df = raw_df[raw_df["season"] < 2025]
target_df = raw_df[raw_df["season"] == 2025]

print("Calculating features...")
hist_features = calc_features(historical_df)
target_features = calc_features(target_df)

# Merge to create training set (only players present in both)
train_data = pd.merge(hist_features, target_features[['player_name', 'Enhanced_FPS']], on='player_name', suffixes=('', '_target'))
train_data = train_data.dropna(subset=['Enhanced_FPS_target'])

features = ["Total_Matches", "Total_Bt_Runs", "Total_Bt_Balls", "Total_Bw_Runs", "Total_Bw_Balls", "Total_Wkts", "Strike_Rate", "Economy_Rate", "SR_Bonus_Points", "Econ_Bonus_Points"]

X_train = train_data[features].fillna(0)
y_train = train_data['Enhanced_FPS_target']

print(f"Training model on {len(X_train)} samples to fix data leakage...")
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

# Calculate training R2 for reporting
y_pred_train = rf.predict(X_train)
print(f"Training R2 (without leakage): {r2_score(y_train, y_pred_train):.3f}")

# Now predict on the FULL dataset (to maintain all players in the Streamlit app)
# Using their overall stats (from phase3_with_roles_teams.csv) as input to the trained model
print("Applying fixed model to full dataset...")
final_df = pd.read_csv("phase3_with_roles_teams.csv")
X_full = final_df[features].fillna(0)
final_df["Predicted_Fantasy_Score"] = rf.predict(X_full)

# Save the updated file
final_df = final_df.sort_values("Predicted_Fantasy_Score", ascending=False)
final_df.to_csv("phase3_with_roles_teams.csv", index=False)
print("Fix applied successfully. phase3_with_roles_teams.csv has been updated.")
