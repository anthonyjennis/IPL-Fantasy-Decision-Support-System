# 🏏IPL Fantasy Decision Support System 📊

**A Machine Learning-powered analytics dashboard to optimize IPL Fantasy team selection.**

### Live App
👉 **[View Dashboard on Streamlit Cloud](https://ipl-fantasy.streamlit.app/)**

---

## Project Overview
The **IPL Fantasy Decision Support System** is a data analytics project designed to identify high-value players for fantasy leagues. Unlike basic stat trackers, this system uses **predictive modeling** to calculate an "Enhanced Fantasy Potential Score" (FPS), identifying the best players to maximize points for a Fantasy Team.

### Tech Stack
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_Dashboard-FF4B4B.svg)

---

## How It Works

This project was built in four distinct phases:

### Data Pipeline (ETL)
*   **Extraction**: Parsed nested JSON data containing ball-by-ball performance for IPL seasons 2023-2025.
*   **Transformation**: Flattened hierarchical structures into tabular format using Python.
*   **Cleaning**: Standardized player names, handled missing values for non-batting/bowling players, and filtered for active players.

### Feature Engineering & Scoring
*   **Custom Metrics**: Introduced "Strike Rate Bonus" and "Economy Rate Bonus" features.
*   **Confidence Scoring**: Created a `Match_Factor` weight to normalize scores, ensuring players with small sample sizes don't skew the rankings.

### Predictive Modeling (Machine Learning)
*   **Goal**: Predict a player's future fantasy output based on historical traits.
*   **Model Selection**: Compared **Linear Regression** vs. **Random Forest Regressor**.
*   **Result**: The Random Forest model achieved a higher R² score, successfully capturing non-linear relationships (e.g., the trade-off between wicket-taking risk and economy rate).

### Visualization & Dashboarding
*   **Interactive UI**: Built a Streamlit app that allows users to filter by teams, stike rate, economy and create a Fantasy XI.
*   **Visual insights**: Bubble charts and ranking bars to visualize the "Fantasy Potential" vs. "Consistency".

---

## Run Locally

1.  **Clone the repository**
    ```bash
    git clone https://github.com/anthonyjennis/IPL-Fantasy-Decision-Support-System.git
    cd IPL-Fantasy-Decision-Support-System
    ```

2.  **Set up environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the App**
    ```bash
    streamlit run streamlit_app.py
    ```