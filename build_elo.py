import math
import pandas as pd

RESULTS_PATH = "data/Results.csv"
ELO_OUTPUT_PATH = "data/elo.csv"

# Same cleaning as in main.py so names line up
TEAM_CLEAN = {
    # Spurs
    "Tottenham Hotspurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspur FC": "Tottenham",
    "Spurs": "Tottenham",

    # Man United
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Manchester United": "Manchester United",
    "Manchester United FC": "Manchester United",

    # Man City
    "Man City": "Manchester City",
    "Manchester City": "Manchester City",
    "Manchester City FC": "Manchester City",

    # Wolves
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Wolverhampton W.": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",

    # West Ham
    "West Ham": "West Ham",
    "West Ham ": "West Ham",
    "West Ham Utd": "West Ham",
    "West Ham United": "West Ham",
    "West Ham United FC": "West Ham",

    # Leeds
    "Leeds": "Leeds United",
    "Leeds Utd": "Leeds United",
    "Leeds United": "Leeds United",
    "Leeds United FC": "Leeds United",


    # Brighton
    "Brighton": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion FC": "Brighton",

    # Nottingham Forest
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
    "Nottingham Forest FC": "Nottingham Forest",

    # Newcastle
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Newcastle United": "Newcastle United",
    "Newcastle United FC": "Newcastle United",

    # Others from your sample
    "Sunderland AFC": "Sunderland",
    "Brentford FC": "Brentford",
    "Everton FC": "Everton",
    "Crystal Palace FC": "Crystal Palace",
    "Aston Villa FC": "Aston Villa",
    "Chelsea FC": "Chelsea",
    "Arsenal FC": "Arsenal",
    "AFC Bournemouth": "Bournemouth",
    "Fulham FC": "Fulham",
    "Burnley FC": "Burnley",
    "Everton FC": "Everton",
    "Liverpool FC": "Liverpool",
}

HOME_ADVANTAGE = 80      # Elo points
BASE_RATING = 1500.0
K = 40.0                  # higher = more reactive


def clean_team(name: str) -> str:
    if isinstance(name, str):
        name = name.strip()
        return TEAM_CLEAN.get(name, name)
    return name


def margin_multiplier(goal_diff: int, rating_diff: float) -> float:
    """
    Standard Elo margin-of-victory multiplier:
    more than 1-goal wins move ratings more, but capped.
    """
    if goal_diff <= 0:
        return 1.0
    # rating_diff is absolute difference between teams
    return math.log(goal_diff + 1.0) * (2.2 / ((rating_diff * 0.001) + 2.2))


def build_elo():
    df = pd.read_csv(RESULTS_PATH)

    # Ensure date sorted (oldest first)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df = df.sort_values("Date")

    ratings: dict[str, float] = {}

    def get_rating(team: str) -> float:
        team = clean_team(team)
        if team not in ratings:
            ratings[team] = BASE_RATING
        return ratings[team]

    for _, row in df.iterrows():
        home_raw = row["HomeTeam"]
        away_raw = row["AwayTeam"]
        home = clean_team(home_raw)
        away = clean_team(away_raw)

        # Goals
        hg = int(row["FTHG"])
        ag = int(row["FTAG"])

        # Init ratings if new
        Rh = get_rating(home)
        Ra = get_rating(away)

        # Home advantage
        Rh_eff = Rh + HOME_ADVANTAGE
        Ra_eff = Ra

        # Expected result (home side perspective)
        exp_home = 1.0 / (1.0 + 10 ** ((Ra_eff - Rh_eff) / 400.0))
        exp_away = 1.0 - exp_home

        # Actual scores (1 win, 0.5 draw, 0 loss)
        if hg > ag:
            act_home, act_away = 1.0, 0.0
        elif hg < ag:
            act_home, act_away = 0.0, 1.0
        else:
            act_home, act_away = 0.5, 0.5

        goal_diff = abs(hg - ag)
        rating_diff = abs(Rh_eff - Ra_eff)
        mult = margin_multiplier(goal_diff, rating_diff)

        # Elo updates
        delta_home = K * mult * (act_home - exp_home)
        delta_away = K * mult * (act_away - exp_away)

        ratings[home] = Rh + delta_home
        ratings[away] = Ra + delta_away

    # Write out final ratings to CSV
    out_rows = [{"Team": team, "Rating": rating} for team, rating in ratings.items()]
    out_df = pd.DataFrame(out_rows)
    out_df = out_df.sort_values("Rating", ascending=False)

    out_df.to_csv(ELO_OUTPUT_PATH, index=False)
    print(f"Saved ELO ratings to {ELO_OUTPUT_PATH}")
    print(out_df.head(10))


if __name__ == "__main__":
    build_elo()
