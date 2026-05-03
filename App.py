import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# Load data
df = pd.read_csv("final_dataset.csv")

avg_home_goals = df['FTHG'].mean()
avg_away_goals = df['FTAG'].mean()

teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

attack_strength = {}
defense_strength = {}

for team in teams:
    home = df[df['HomeTeam'] == team]
    away = df[df['AwayTeam'] == team]

    attack = (home['FTHG'].mean() + away['FTAG'].mean()) / 2
    defense = (home['FTAG'].mean() + away['FTHG'].mean()) / 2

    attack_strength[team] = attack / avg_home_goals
    defense_strength[team] = defense / avg_away_goals


def predict(home_team, away_team):

    home_lambda = attack_strength[home_team] * defense_strength[away_team] * avg_home_goals
    away_lambda = attack_strength[away_team] * defense_strength[home_team] * avg_away_goals

    home_lambda = max(0.01, home_lambda)
    away_lambda = max(0.01, away_lambda)

    max_goals = 5
    matrix = np.zeros((max_goals+1, max_goals+1))

    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i][j] = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)

    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))

    return {
        "home_goals": round(home_lambda, 2),
        "away_goals": round(away_lambda, 2),
        "home_win": round(home_win, 2),
        "draw": round(draw, 2),
        "away_win": round(away_win, 2)
    }


# ================= APP UI =================
st.title("⚽ Football Prediction App")

home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", teams)

if st.button("Predict Match"):
    result = predict(home_team, away_team)

    st.subheader("📊 Prediction Result")
    st.write(f"Expected Goals: {result['home_goals']} - {result['away_goals']}")
    st.write(f"Home Win: {result['home_win']*100:.1f}%")
    st.write(f"Draw: {result['draw']*100:.1f}%")
    st.write(f"Away Win: {result['away_win']*100:.1f}%")