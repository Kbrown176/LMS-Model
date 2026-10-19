"""Read model inputs out of the SQLite store."""

from __future__ import annotations

import sqlite3

import pandas as pd

from lms import store


def training_matches(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    """All results, plus current-season PL games FPL has finished but football-data.co.uk
    hasn't published yet (it updates a couple of times a week)."""
    results = store.read(
        conn,
        "SELECT season, division, date, home_id, away_id, home_goals, away_goals, "
        "odds_home, odds_draw, odds_away FROM results",
    )
    late = store.read(
        conn,
        """
        SELECT f.season, 'E0' AS division, substr(f.kickoff_utc, 1, 10) AS date,
               f.home_id, f.away_id, f.home_score AS home_goals, f.away_score AS away_goals
        FROM fixtures f
        LEFT JOIN results r ON r.season = f.season AND r.division = 'E0'
             AND r.home_id = f.home_id AND r.away_id = f.away_id
        WHERE f.season = ? AND f.finished AND r.home_id IS NULL
        """,
        (season,),
    )
    if late.empty:
        return results.sort_values("date", ignore_index=True)
    return pd.concat([results, late], ignore_index=True).sort_values("date", ignore_index=True)


def upcoming_fixtures(conn: sqlite3.Connection, season: str) -> pd.DataFrame:
    return store.read(
        conn,
        "SELECT fixture_id, gw, kickoff_utc, home_id, away_id FROM fixtures "
        "WHERE season = ? AND NOT finished ORDER BY kickoff_utc",
        (season,),
    )


def latest_odds(conn: sqlite3.Connection) -> pd.DataFrame:
    """Most recent price per bookmaker for matches that haven't kicked off."""
    return store.read(
        conn,
        "SELECT * FROM odds WHERE commence_utc > strftime('%Y-%m-%dT%H:%M:%SZ', 'now')",
    )
