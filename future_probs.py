# future_probs.py

import pandas as pd


def fill_missing_probs_with_elo(
    merged: pd.DataFrame,
    elo_table: dict,
    home_advantage: float = 100.0,
) -> pd.DataFrame:
    """
    For fixtures missing probabilities from odds, fill using Elo ratings.
    Adds/updates a 'prob_source' column to indicate 'elo' vs 'odds'.
    """

    # If the column doesn't exist, assume all pre-filled probs came from odds
    if "prob_source" not in merged.columns:
        merged["prob_source"] = "odds"

    for idx, row in merged.iterrows():
        if pd.isna(row.get("p_home_win")):
            home = row["home_clean"]
            away = row["away_clean"]

            if home in elo_table and away in elo_table:
                R_home = elo_table[home] + home_advantage
                R_away = elo_table[away]

                p_home = 1 / (1 + 10 ** ((R_away - R_home) / 400))
                p_away = 1 - p_home

                merged.at[idx, "p_home_win"] = p_home
                merged.at[idx, "p_away_win"] = p_away
                merged.at[idx, "prob_source"] = "elo"

    return merged
