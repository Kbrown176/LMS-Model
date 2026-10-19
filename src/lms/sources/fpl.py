"""Fantasy Premier League API: official gameweeks, fixtures and live results. No key needed."""

from __future__ import annotations

import pandas as pd

from lms.sources.http import get
from lms.teams import team_id

BASE_URL = "https://fantasy.premierleague.com/api"


def fetch_bootstrap() -> dict:
    return get(f"{BASE_URL}/bootstrap-static/").json()


def fetch_fixtures() -> list[dict]:
    return get(f"{BASE_URL}/fixtures/").json()


def parse_gameweeks(bootstrap: dict, season: str) -> pd.DataFrame:
    rows = [
        {
            "season": season,
            "gw": e["id"],
            "deadline_utc": e["deadline_time"],
            "finished": bool(e["finished"]),
            "is_current": bool(e["is_current"]),
            "is_next": bool(e["is_next"]),
        }
        for e in bootstrap["events"]
    ]
    return pd.DataFrame(rows)


def parse_fixtures(bootstrap: dict, fixtures: list[dict], season: str) -> pd.DataFrame:
    """One row per fixture with canonical team ids. Unscheduled fixtures have gw = None."""
    fpl_to_id = {t["id"]: team_id(t["name"], "FPL") for t in bootstrap["teams"]}
    rows = [
        {
            "fixture_id": f["id"],
            "season": season,
            "gw": f["event"],
            "kickoff_utc": f["kickoff_time"],
            "home_id": fpl_to_id[f["team_h"]],
            "away_id": fpl_to_id[f["team_a"]],
            "finished": bool(f["finished"]),
            "home_score": f["team_h_score"],
            "away_score": f["team_a_score"],
        }
        for f in fixtures
    ]
    return pd.DataFrame(rows)
