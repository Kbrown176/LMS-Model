"""Shared 'what should I pick now?' pipeline for `lms pick` and `lms report`."""

from __future__ import annotations

import sqlite3
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lms.config import DERBIES_PATH, ENTRY_PATH, Settings
from lms.forecast import team_view, upcoming
from lms.strategy import Recommendation, build_board, load_derbies, recommend
from lms.teams import team_id


@dataclass
class Entry:
    """Your LMS entry: teams already used and lives left (config/entry.toml)."""

    used: set[str]
    lives: int


def load_entry(settings: Settings, path: Path = ENTRY_PATH) -> Entry:
    if not path.exists():
        return Entry(set(), settings.lives)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    used = {team_id(t, path.name) for t in raw.get("used", [])}
    return Entry(used, int(raw.get("lives", settings.lives)))


def parse_used(text: str) -> set[str]:
    return {team_id(t, "--used") for t in text.split(",") if t.strip()}


@dataclass
class PickContext:
    rec: Recommendation
    forecast: pd.DataFrame
    gw: int
    deadline_utc: str | None
    used: set[str]
    lives: int


def plan(
    conn: sqlite3.Connection,
    settings: Settings,
    used: set[str],
    lives: int,
    gw: int | None = None,
    horizon: int | None = None,
) -> PickContext:
    fc, _ = upcoming(conn, settings)
    fc = fc[fc.gw.notna()].astype({"gw": int})
    nxt = conn.execute(
        "SELECT gw, deadline_utc FROM gameweeks WHERE season = ? AND is_next", (settings.season,)
    ).fetchone()
    start = gw or (nxt[0] if nxt else int(fc.gw.min()))
    deadline = nxt[1] if nxt and nxt[0] == start else None
    board = build_board(
        team_view(fc),
        start,
        horizon or settings.horizon,
        used=used,
        derbies=load_derbies(DERBIES_PATH),
        avoid_derbies=settings.avoid_derbies,
        derby_min_intensity=settings.derby_min_intensity,
        derby_penalty=settings.derby_penalty,
        min_win_prob=settings.min_win_prob,
    )
    return PickContext(recommend(board, lives), fc, start, deadline, used, lives)
