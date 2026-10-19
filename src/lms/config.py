"""Project paths and settings loaded from config/settings.toml."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "lms.db"
REPORTS_DIR = ROOT / "reports"
SETTINGS_PATH = CONFIG_DIR / "settings.toml"
DERBIES_PATH = CONFIG_DIR / "derbies.csv"

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    season: str
    history_seasons: int
    divisions: tuple[str, ...]
    lives: int
    horizon: int
    min_win_prob: float
    avoid_derbies: bool
    derby_min_intensity: float
    derby_penalty: float
    time_decay_half_life_days: float
    ridge: float
    market_weight: float
    adjust_ridge: float


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return Settings(
        season=raw["season"]["current"],
        history_seasons=raw["season"]["history_seasons"],
        divisions=tuple(raw["season"]["divisions"]),
        lives=raw["rules"]["lives"],
        horizon=raw["strategy"]["horizon"],
        min_win_prob=raw["strategy"]["min_win_prob"],
        avoid_derbies=raw["strategy"]["avoid_derbies"],
        derby_min_intensity=raw["strategy"]["derby_min_intensity"],
        derby_penalty=raw["strategy"]["derby_penalty"],
        time_decay_half_life_days=raw["model"]["time_decay_half_life_days"],
        ridge=raw["model"]["ridge"],
        market_weight=raw["blend"]["market_weight"],
        adjust_ridge=raw["blend"]["adjust_ridge"],
    )


def odds_api_key() -> str | None:
    return os.getenv("ODDS_API_KEY")
