"""The Odds API: live 1X2 prices from every UK bookmaker for upcoming PL fixtures.

One call returns all upcoming fixtures and costs 1 request from the monthly quota.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lms.sources.http import get
from lms.teams import team_id

URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"


@dataclass
class OddsResponse:
    events: list[dict]
    requests_remaining: str | None


def fetch(api_key: str, regions: str = "uk") -> OddsResponse:
    resp = get(
        URL,
        params={"apiKey": api_key, "regions": regions, "markets": "h2h", "oddsFormat": "decimal"},
    )
    return OddsResponse(resp.json(), resp.headers.get("x-requests-remaining"))


def parse(events: list[dict], fetched_at: str) -> pd.DataFrame:
    """One row per (event, bookmaker) with home/draw/away decimal prices."""
    rows = []
    for e in events:
        home, away = e["home_team"], e["away_team"]
        home_id, away_id = team_id(home, "The Odds API"), team_id(away, "The Odds API")
        for book in e.get("bookmakers", []):
            market = next((m for m in book.get("markets", []) if m["key"] == "h2h"), None)
            if market is None:
                continue
            prices = {o["name"]: o["price"] for o in market["outcomes"]}
            if not {home, away, "Draw"} <= prices.keys():
                continue
            rows.append(
                {
                    "event_id": e["id"],
                    "bookmaker": book["key"],
                    "commence_utc": e["commence_time"],
                    "home_id": home_id,
                    "away_id": away_id,
                    "odds_home": prices[home],
                    "odds_draw": prices["Draw"],
                    "odds_away": prices[away],
                    "last_update": market.get("last_update") or book.get("last_update"),
                    "fetched_at": fetched_at,
                }
            )
    return pd.DataFrame(rows)
