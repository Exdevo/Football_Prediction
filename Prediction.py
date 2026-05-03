import pandas as pd
import numpy as np
from scipy.stats import poisson

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("final_dataset.csv")

# League averages
avg_home_goals = df['FTHG'].mean()
avg_away_goals = df['FTAG'].mean()

# Teams
teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

attack_strength = {}
defense_strength = {}

# =========================
# CALCULATE TEAM STRENGTHS
# =========================
for team in teams:
    home = df[df['HomeTeam'] == team]
    away = df[df['AwayTeam'] == team]

    attack = (home['FTHG'].mean() + away['FTAG'].mean()) / 2
    defense = (home['FTAG'].mean() + away['FTHG'].mean()) / 2

    attack_strength[team] = attack / avg_home_goals
    defense_strength[team] = defense / avg_away_goals


# =========================
# PREDICTION FUNCTION
# =========================
def predict(home_team, away_team):

    # safety check (important)
    if home_team not in attack_strength or away_team not in attack_strength:
        return None

    home_lambda = attack_strength[home_team] * defense_strength[away_team] * avg_home_goals
    away_lambda = attack_strength[away_team] * defense_strength[home_team] * avg_away_goals

    home_lambda = max(0.01, home_lambda)
    away_lambda = max(0.01, away_lambda)

    max_goals = 5
    matrix = np.zeros((max_goals + 1, max_goals + 1))

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i][j] = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)

    home_win = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away_win = np.sum(np.triu(matrix, 1))

    # FINAL RESULT (THIS WAS MISSING BEFORE)
    result = {
        "home_goals_exp": round(home_lambda, 2),
        "away_goals_exp": round(away_lambda, 2),
        "home_win_prob": round(home_win, 2),
        "draw_prob": round(draw, 2),
        "away_win_prob": round(away_win, 2)
    }

    return result


# =========================
# RUN PREDICTION (OUTSIDE FUNCTION)
# =========================
result = predict("Arsenal", "Chelsea")

if result is None:
    print("❌ ERROR: Team not found in dataset or prediction failed.")
else:
    print("\n=== MATCH PREDICTION ===")
    print(f"Expected Goals: {result['home_goals_exp']} - {result['away_goals_exp']}")
    print(f"Home Win: {result['home_win_prob']*100:.1f}%")
    print(f"Draw: {result['draw_prob']*100:.1f}%")
    print(f"Away Win: {result['away_win_prob']*100:.1f}%")