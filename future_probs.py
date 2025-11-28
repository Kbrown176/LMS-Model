# future_probs.py

import pandas as pd


def _build_elo_map(elo_table: pd.DataFrame) -> dict:
    """
    Build a dict {team_name: elo} from a DataFrame.
    Expects either:
      - columns 'team' and 'elo', OR
      - first column = team, second column = elo
    """
    df = elo_table.copy()

    if "team" in df.columns and "elo" in df.columns:
        team_col = "team"
        elo_col = "elo"
    else:
        # fallback: assume first col is team, second is elo
        team_col = df.columns[0]
        elo_col = df.columns[1]

    df["team_clean"] = df[team_col].astype(str).str.strip()
    return dict(zip(df["team_clean"], df[elo_col].astype(float)))


def fill_missing_probs_with_elo(
    merged: pd.DataFrame,
    elo_table: pd.DataFrame,
    home_advantage: float = 100.0,
) -> pd.DataFrame:
    """
    For fixtures missing probabilities from odds, fill using Elo ratings.

    - merged: DataFrame after odds merge, with columns:
        - 'home_clean' / 'away_clean' (preferred)
          or 'HomeTeam' / 'AwayTeam' as fallback
        - 'p_home_win', 'p_away_win' (may contain NaN)
    - elo_table: DataFrame loaded from data/elo.csv

    Adds/updates:
      - 'p_home_win', 'p_away_win'
      - 'prob_source' = 'elo' where Elo was used
    """

    df = merged.copy()

    # Ensure prob_source column exists
    if "prob_source" not in df.columns:
        df["prob_source"] = pd.NA

    elo_map = _build_elo_map(elo_table)

    def get_teams(row):
        """
        Prefer cleaned names if present; otherwise raw HomeTeam/AwayTeam.
        """
        home = row.get("home_clean")
        away = row.get("away_clean")

        if not isinstance(home, str):
            home = row.get("HomeTeam")
        if not isinstance(away, str):
            away = row.get("AwayTeam")

        if isinstance(home, str):
            home = home.strip()
        if isinstance(away, str):
            away = away.strip()

        return home, away

    for idx, row in df.iterrows():
        # Skip rows where we already have bookmaker probabilities
        if pd.notna(row.get("p_home_win")):
            continue

        home, away = get_teams(row)
        if not isinstance(home, str) or not isinstance(away, str):
            continue

        if home not in elo_map or away not in elo_map:
            continue

        R_home = elo_map[home] + home_advantage
        R_away = elo_map[away]

        p_home = 1.0 / (1.0 + 10.0 ** ((R_away - R_home) / 400.0))
        p_away = 1.0 - p_home

        df.at[idx, "p_home_win"] = p_home
        df.at[idx, "p_away_win"] = p_away
        df.at[idx, "prob_source"] = "elo"

    return df
