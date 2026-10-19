"""Turning bookmaker decimal odds into probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def devig(odds_home, odds_draw, odds_away) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implied probabilities with the bookmaker margin removed proportionally.

    Accepts scalars or arrays; returns arrays that sum to 1 per match.
    """
    inv = np.vstack([1 / np.asarray(o, dtype=float) for o in (odds_home, odds_draw, odds_away)])
    probs = inv / inv.sum(axis=0)
    return probs[0], probs[1], probs[2]


def consensus(odds: pd.DataFrame) -> pd.DataFrame:
    """Average the de-vigged probabilities of every bookmaker per match.

    `odds` has one row per (event, bookmaker); returns one row per (home_id, away_id).
    """
    ph, pd_, pa = devig(odds.odds_home, odds.odds_draw, odds.odds_away)
    df = odds[["home_id", "away_id", "commence_utc"]].assign(p_home=ph, p_draw=pd_, p_away=pa)
    return (
        df.groupby(["home_id", "away_id"], as_index=False)
        .agg(
            commence_utc=("commence_utc", "first"),
            p_home=("p_home", "mean"),
            p_draw=("p_draw", "mean"),
            p_away=("p_away", "mean"),
            n_books=("p_home", "size"),
        )
        .sort_values("commence_utc", ignore_index=True)
    )
