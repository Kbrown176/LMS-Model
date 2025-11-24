import math
import pandas as pd

HOME_ADVANTAGE = 100  # Elo points

def load_elo_table(path="data/elo.csv"):
    """Loads a CSV with columns: Team, Rating"""
    df = pd.read_csv(path)
    return {row["Team"]: row["Rating"] for _, row in df.iterrows()}

def elo_win_prob(R_home, R_away):
    """Return win probabilities (no draw modelling yet)."""

    # Home team advantage
    R_home_adj = R_home + HOME_ADVANTAGE

    exp_home = 1 / (1 + 10 ** ((R_away - R_home_adj) / 400))
    exp_away = 1 - exp_home

    return exp_home, exp_away
