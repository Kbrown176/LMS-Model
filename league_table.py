from dotenv import load_dotenv
load_dotenv()

import os
import requests
import pandas as pd

BASE_URL = "https://api.football-data.org/v4"
API_KEY = os.getenv("TABLE_API_KEY")
HEADERS = {"X-Auth-Token": API_KEY}


def _get(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params or {})
    resp.raise_for_status()
    return resp.json()


def get_prem_table():
    data = _get("/competitions/PL/standings")

    table = []
    for row in data["standings"][0]["table"]:
        table.append({
            "position": row["position"],
            "team": row["team"]["name"],
            "played": row["playedGames"],
            "won": row["won"],
            "draw": row["draw"],
            "lost": row["lost"],
            "goals_for": row["goalsFor"],
            "goals_against": row["goalsAgainst"],
            "goal_diff": row["goalDifference"],
            "points": row["points"]
        })

    return pd.DataFrame(table)


def get_prem_fixtures(status="SCHEDULED"):
    data = _get("/competitions/PL/matches", params={"status": status})

    matches = []
    for m in data["matches"]:
        matches.append({
            "match_id": m["id"],
            "gameweek": m["matchday"],
            "utc_date": m["utcDate"],
            "status": m["status"],
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
            "home_score": m["score"]["fullTime"]["home"],
            "away_score": m["score"]["fullTime"]["away"]
        })

    return pd.DataFrame(matches)


def get_prem_results():
    return get_prem_fixtures(status="FINISHED")

def refresh_csvs():
    """Fetch fresh data from API and save to CSV files that main.py expects."""
    # League table & results – you might use later for Elo, etc.
    table = get_prem_table()
    fixtures_api = get_prem_fixtures(status="SCHEDULED")
    results_api = get_prem_results()

    # ---- Build Fixtures.csv in the format main.py expects ----
    fixtures = fixtures_api.copy()

    # Convert date to dd/mm/YYYY string
    fixtures["Date"] = (
        pd.to_datetime(fixtures["utc_date"])
        .dt.strftime("%d/%m/%Y")
    )

    # Rename teams + round to match your existing code
    fixtures["HomeTeam"] = fixtures["home_team"]
    fixtures["AwayTeam"] = fixtures["away_team"]
    fixtures["Round"] = fixtures["gameweek"].astype("Int64")

    # Keep only the columns main.py cares about
    fixtures_out = fixtures[["Date", "HomeTeam", "AwayTeam", "Round"]]

    # IMPORTANT: match the path + filename in main.py exactly
    fixtures_out.to_csv("data/Fixtures.csv", index=False)

    # Optional: save these for other parts of your project
    table.to_csv("data/table.csv", index=False)
    results_api.to_csv("data/results_raw.csv", index=False)

    print("Updated data/Fixtures.csv, data/table.csv, data/results_raw.csv")

