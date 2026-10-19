"""SQLite store. Every write is an upsert, so refreshing twice never duplicates rows."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from lms.config import DB_PATH
from lms.teams import TEAMS

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    name    TEXT NOT NULL,
    short   TEXT NOT NULL
);

-- Historical results (football-data.co.uk), PL + Championship.
CREATE TABLE IF NOT EXISTS results (
    season     TEXT NOT NULL,
    division   TEXT NOT NULL,
    date       TEXT NOT NULL,
    home_id    TEXT NOT NULL REFERENCES teams(team_id),
    away_id    TEXT NOT NULL REFERENCES teams(team_id),
    home_goals INTEGER NOT NULL,
    away_goals INTEGER NOT NULL,
    odds_home  REAL,
    odds_draw  REAL,
    odds_away  REAL,
    PRIMARY KEY (season, division, home_id, away_id)
);

-- Current-season PL fixtures and gameweeks (FPL).
CREATE TABLE IF NOT EXISTS gameweeks (
    season       TEXT NOT NULL,
    gw           INTEGER NOT NULL,
    deadline_utc TEXT NOT NULL,
    finished     INTEGER NOT NULL,
    is_current   INTEGER NOT NULL,
    is_next      INTEGER NOT NULL,
    PRIMARY KEY (season, gw)
);

CREATE TABLE IF NOT EXISTS fixtures (
    fixture_id  INTEGER NOT NULL,
    season      TEXT NOT NULL,
    gw          INTEGER,
    kickoff_utc TEXT,
    home_id     TEXT NOT NULL REFERENCES teams(team_id),
    away_id     TEXT NOT NULL REFERENCES teams(team_id),
    finished    INTEGER NOT NULL,
    home_score  INTEGER,
    away_score  INTEGER,
    PRIMARY KEY (season, fixture_id)
);

-- Latest price per bookmaker per upcoming match (The Odds API).
CREATE TABLE IF NOT EXISTS odds (
    event_id     TEXT NOT NULL,
    bookmaker    TEXT NOT NULL,
    commence_utc TEXT NOT NULL,
    home_id      TEXT NOT NULL REFERENCES teams(team_id),
    away_id      TEXT NOT NULL REFERENCES teams(team_id),
    odds_home    REAL NOT NULL,
    odds_draw    REAL NOT NULL,
    odds_away    REAL NOT NULL,
    last_update  TEXT,
    fetched_at   TEXT NOT NULL,
    PRIMARY KEY (event_id, bookmaker)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@contextmanager
def connect(path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.executescript(SCHEMA)
        _sync_teams(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _sync_teams(conn: sqlite3.Connection) -> None:
    conn.executemany(
        "INSERT INTO teams VALUES (?, ?, ?) "
        "ON CONFLICT(team_id) DO UPDATE SET name = excluded.name, short = excluded.short",
        [(t.id, t.name, t.short) for t in TEAMS.values()],
    )


def upsert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """Insert rows, replacing any with the same primary key. Returns rows written."""
    if df.empty:
        return 0
    cols = list(df.columns)
    placeholders = ", ".join("?" * len(cols))
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    records = df.astype(object).where(df.notna(), None).itertuples(index=False, name=None)
    conn.executemany(sql, records)
    return len(df)


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def read(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)
