"""football-data.co.uk: historical results and closing odds for PL (E0) and Championship (E1).

Completed seasons never change, so they are downloaded once and cached. The current
season's file is refetched when the cached copy is older than `max_age_hours`.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd

from lms.sources.http import get
from lms.teams import team_id

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Average closing odds first; fall back to average pre-match odds, then Pinnacle closing.
_ODDS_COLUMNS = [("AvgCH", "AvgCD", "AvgCA"), ("AvgH", "AvgD", "AvgA"), ("PSCH", "PSCD", "PSCA")]


def season_codes(current: str, history: int) -> list[str]:
    """'2627', 5 -> ['2122', '2223', '2324', '2425', '2526', '2627']."""
    start = int(current[:2])
    return [f"{y:02d}{y + 1:02d}" for y in range(start - history, start + 1)]


def download(season: str, division: str, cache_dir: Path, max_age_hours: float | None) -> Path:
    """Return a cached CSV path, downloading if missing or stale (None = never stale)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{division}_{season}.csv"
    fresh = path.exists() and (
        max_age_hours is None or time.time() - path.stat().st_mtime < max_age_hours * 3600
    )
    if not fresh:
        path.write_bytes(get(f"{BASE_URL}/{season}/{division}.csv").content)
    return path


def parse_results(raw: bytes | str | Path, season: str, division: str) -> pd.DataFrame:
    if isinstance(raw, bytes):
        raw = io.BytesIO(raw)
    df = pd.read_csv(raw, encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    source = f"football-data.co.uk {division} {season}"
    odds_h = odds_d = odds_a = pd.Series(float("nan"), index=df.index)
    for h, d, a in _ODDS_COLUMNS:
        if h in df.columns:
            odds_h = odds_h.fillna(df[h])
            odds_d = odds_d.fillna(df[d])
            odds_a = odds_a.fillna(df[a])

    dates = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    out = pd.DataFrame(
        {
            "season": season,
            "division": division,
            "date": dates.dt.strftime("%Y-%m-%d"),
            "home_id": df["HomeTeam"].map(lambda n: team_id(n, source)),
            "away_id": df["AwayTeam"].map(lambda n: team_id(n, source)),
            "home_goals": df["FTHG"].astype(int),
            "away_goals": df["FTAG"].astype(int),
            "odds_home": odds_h.astype(float),
            "odds_draw": odds_d.astype(float),
            "odds_away": odds_a.astype(float),
        }
    )
    return out.reset_index(drop=True)
