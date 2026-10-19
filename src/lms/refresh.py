"""`lms refresh`: pull every source into the SQLite store."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime

from lms import store
from lms.config import CACHE_DIR, Settings, odds_api_key
from lms.sources import footballdata, fpl, odds


@dataclass
class RefreshSummary:
    lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        self.lines.append(line)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def refresh_history(
    conn: sqlite3.Connection,
    settings: Settings,
    summary: RefreshSummary,
    force: bool,
    max_age_hours: float = 12,
) -> None:
    seasons = footballdata.season_codes(settings.season, settings.history_seasons)
    total = 0
    for season in seasons:
        is_current = season == settings.season
        # Past seasons are immutable: cache forever. Current season: refetch when stale.
        max_age = 0 if force else (max_age_hours if is_current else None)
        for division in settings.divisions:
            path = footballdata.download(season, division, CACHE_DIR, max_age)
            total += store.upsert(
                conn, "results", footballdata.parse_results(path, season, division)
            )
    span = f"{seasons[0]}-{seasons[-1]}"
    summary.add(f"Results: {total} matches across {len(seasons)} seasons ({span})")


def refresh_fpl(conn: sqlite3.Connection, settings: Settings, summary: RefreshSummary) -> None:
    bootstrap = fpl.fetch_bootstrap()
    fixtures = fpl.parse_fixtures(bootstrap, fpl.fetch_fixtures(), settings.season)
    gameweeks = fpl.parse_gameweeks(bootstrap, settings.season)
    store.upsert(conn, "gameweeks", gameweeks)
    store.upsert(conn, "fixtures", fixtures)

    nxt = gameweeks[gameweeks["is_next"]]
    next_gw = "none"
    if len(nxt):
        next_gw = f"GW{nxt.iloc[0]['gw']} (deadline {nxt.iloc[0]['deadline_utc']})"
    summary.add(
        f"Fixtures: {len(fixtures)} ({int(fixtures['finished'].sum())} played), next {next_gw}"
    )
    unscheduled = int(fixtures["gw"].isna().sum())
    if unscheduled:
        summary.warnings.append(f"{unscheduled} fixtures have no gameweek yet (postponed/TBC)")


def refresh_odds(conn: sqlite3.Connection, summary: RefreshSummary) -> None:
    key = odds_api_key()
    if not key:
        summary.warnings.append("ODDS_API_KEY not set; skipped odds (model-only probabilities)")
        return
    resp = odds.fetch(key)
    df = odds.parse(resp.events, _now())
    store.upsert(conn, "odds", df)
    if resp.requests_remaining is not None:
        store.set_meta(conn, "odds_requests_remaining", resp.requests_remaining)
    events = df["event_id"].nunique() if len(df) else 0
    summary.add(
        f"Odds: {events} matches, {len(df)} bookmaker prices "
        f"(API requests left this month: {resp.requests_remaining or '?'})"
    )


def run(
    settings: Settings,
    *,
    force: bool = False,
    with_odds: bool = True,
    max_age_hours: float = 12,
) -> RefreshSummary:
    summary = RefreshSummary()
    with store.connect() as conn:
        refresh_history(conn, settings, summary, force, max_age_hours)
        refresh_fpl(conn, settings, summary)
        if with_odds:
            refresh_odds(conn, summary)
        store.set_meta(conn, "last_refresh_utc", _now())
    return summary
