import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"


def fetch_epl_odds(regions="uk", markets="h2h", odds_format="decimal"):
    if not API_KEY:
        raise ValueError("ODDS_API_KEY is not set. Put it in your .env file.")

    params = {
        "apiKey": API_KEY,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
    }

    print("Requesting odds from API...")
    resp = requests.get(BASE_URL, params=params)

    print("Status code:", resp.status_code)
    # If something is wrong (bad key, etc.), show the first bit of the response
    if resp.status_code != 200:
        print("Response text (first 500 chars):")
        print(resp.text[:500])
        resp.raise_for_status()

    data = resp.json()
    print(f"API returned {len(data)} events")
    return data


def flatten_odds(data):
    rows = []

    for event in data:
        home = event.get("home_team")
        away = event.get("away_team")
        commence_time = event.get("commence_time")

        if not home or not away or not commence_time:
            continue

        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        dt = dt.astimezone(timezone.utc)

        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            continue

        # for debugging, list bookmakers we see
        # print("Bookmakers for", home, "vs", away, ":", [b["title"] for b in bookmakers])

        bookmaker = bookmakers[0]
        markets = bookmaker.get("markets", [])
        h2h_markets = [m for m in markets if m.get("key") == "h2h"]
        if not h2h_markets:
            continue

        h2h = h2h_markets[0]
        outcomes = h2h.get("outcomes", [])
        odds_map = {o["name"]: o["price"] for o in outcomes}

        home_odds = odds_map.get(home)
        away_odds = odds_map.get(away)
        draw_odds = odds_map.get("Draw") or odds_map.get("Tie")

        rows.append(
            {
                "date_utc": dt.date().isoformat(),
                "kickoff_utc": dt.isoformat(),
                "home_team": home,
                "away_team": away,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "bookmaker": bookmaker.get("title"),
            }
        )

    df = pd.DataFrame(rows)
    print(f"Flattened into {len(df)} rows")
    return df


def main():
    data = fetch_epl_odds()
    odds_df = flatten_odds(data)

    output_path = os.path.join("data", "epl_current_odds.csv")
    odds_df.to_csv(output_path, index=False)
    print(f"Saved odds to {output_path}")
    print(odds_df.head())


if __name__ == "__main__":
    main()
