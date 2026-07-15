# 🏏 IPL Fantasy Decision Support System 📊

**An End-to-End Machine Learning-powered analytics dashboard to optimize IPL Fantasy team selection.**

### Live App
👉 **[View Dashboard on Streamlit Cloud](https://ipl-fantasy.streamlit.app/)**

---

## 🎯 Project Overview
The **IPL Fantasy Decision Support System** is a data analytics and machine learning project designed to identify high-value players for fantasy cricket leagues. Unlike basic stat trackers that only show past aggregates, this system uses a **predictive modeling pipeline** to forecast a player's future "Fantasy Potential Score", identifying the best players to maximize points for a Fantasy Team.

This project demonstrates a complete **Full-Stack Data Science & ML workflow**: from raw JSON extraction and ETL to feature engineering, Machine Learning modeling (Random Forest), and a deployed interactive dashboard.

### 🛠️ Tech Stack & Skills Demonstrated
* **Languages:** Python (Pandas, NumPy)
* **Machine Learning:** Scikit-Learn (Random Forest Regressor), Predictive Modeling, Time-Series Data Splits
* **Data Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit
* **Concepts:** ETL Pipelines, Feature Engineering, Regression, Data Leakage Prevention, Interactive Dashboards

---

## 🏗️ System Flow & Architecture

The project processes data in four distinct phases:

1. **Phase 1: Data Pipeline (ETL)**
   * **Extraction:** Ingests raw, nested JSON data containing ball-by-ball performance for IPL seasons (2023-2025).
   * **Transformation:** Flattens the hierarchical structures into a tabular CSV format.
   * **Cleaning:** Handles missing values, standardizes player names, and filters out inactive players.
2. **Phase 2: Feature Engineering & Scoring**
   * Computes custom metrics, including a dynamic "Strike Rate Bonus" and "Economy Rate Bonus".
   * Normalizes match counts to ensure that players with a small sample size do not arbitrarily skew the rankings.
3. **Phase 3: Predictive Modeling (Machine Learning)**
   * **Goal:** Predict a player's future fantasy output based on historical traits.
   * **Methodology:** The data is split temporally. Stats from previous seasons (2023-2024) act as the input features (X), while the actual fantasy points earned in the next season (2025) act as the target (Y).
   * **Model:** A **Random Forest Regressor** is trained to capture non-linear relationships (e.g., the trade-off between wicket-taking risk and economy rate). This prevents data leakage and ensures robust, real-world predictive power.
4. **Phase 4: Visualization & Dashboarding**
   * Built an interactive **Streamlit** dashboard.
   * Users can filter players by team, minimum matches, strike rate, and economy.
   * The app includes an "Optimize XI Automatically" feature that programmatically selects the highest-value team while adhering to role constraints.

---

## 📂 File-by-File Explanation

If you are exploring the codebase, here is a guide to what each file does:

* `player_performance.json`: The raw data source containing nested ball-by-ball and match stats for players across multiple seasons.
* `phase1.ipynb`: The ETL notebook. Reads the raw JSON, flattens it into a Pandas DataFrame, cleans up string names, handles NaNs, and exports `final_clean_ipl_2023_2025.csv`.
* `phase2.ipynb`: The Feature Engineering notebook. Aggregates the cleaned data, defines the fantasy point scoring system, and calculates standard metrics like Strike Rate and Economy.
* `phase3.ipynb` / `phase3_fixed_model.py`: The Machine Learning modeling script. Splits the data temporally to prevent data leakage, trains the Random Forest Regressor on historical stats to predict future fantasy scores, and generates the final predictive dataset.
* `phase4.ipynb`: Exploratory Data Analysis (EDA). Contains Matplotlib and Seaborn visualizations detailing correlations, player consistency, and top performers.
* `streamlit_app.py`: The frontend application code. Loads the final ML-predicted dataset and renders the interactive UI, charts, and team optimization logic using Streamlit.
* `requirements.txt`: Defines all the Python dependencies required to run the project locally.

---

## 🚀 Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/anthonyjennis/IPL-Fantasy-Decision-Support-System.git
   cd IPL-Fantasy-Decision-Support-System
   ```

2. **Set up environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the App**
   ```bash
   streamlit run streamlit_app.py
   ```