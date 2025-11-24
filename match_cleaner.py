import pandas as pd
from datetime import datetime

TEAM_CLEAN = {
    "Tottenham Hotspurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Man City": "Manchester City",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "West Ham Utd": "West Ham United",
    # Add more once we see inconsistencies
}

def clean_team(name):
    name = name.strip()
    return TEAM_CLEAN.get(name, name)

def clean_and_merge(odds_path, fixtures_path):
    odds = pd.read_csv(odds_path)
    fixtures = pd.read_csv(fixtures_path)

    # Clean team names
    odds["home_clean"] = odds["home_team"].apply(clean_team)
    odds["away_clean"] = odds["away_team"].apply(clean_team)
    fixtures["home_clean"] = fixtures["HomeTeam"].apply(clean_team)
    fixtures["away_clean"] = fixtures["AwayTeam"].apply(clean_team)

    # Clean and match dates
    fixtures["date_clean"] = pd.to_datetime(fixtures["Date"], format="%d/%m/%Y").dt.date.astype(str)
    odds["date_clean"] = pd.to_datetime(odds["date_utc"]).dt.date.astype(str)

    # Merge to attach Round (gameweek)
    merged = odds.merge(
        fixtures[["Round", "date_clean", "home_clean", "away_clean"]],
        on=["date_clean", "home_clean", "away_clean"],
        how="left"
    )

    return merged
