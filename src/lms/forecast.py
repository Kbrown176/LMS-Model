"""Win probabilities for upcoming fixtures, combining the model with bookmaker odds.

Two ways the market feeds in:

1. Blending. Where a fixture has odds, model and market are combined with a logarithmic pool:
   p ∝ p_model^(1-w) · p_market^w.

2. Market-adjusted ratings. Odds only exist for the next week or two, but LMS planning looks
   ten weeks ahead. Each team's strength is nudged (by δ, split across attack and defence) so
   the model agrees with the odds it can see, with a ridge penalty κ keeping δ small. The
   adjusted model then forecasts the later fixtures that have no odds yet.

Both w and κ are tuned walk-forward (see `evaluate`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from lms.model import MAX_GOALS, DixonColes

PROB_COLS = ["p_home", "p_draw", "p_away"]
MARKET_COLS = ["m_home", "m_draw", "m_away"]


@dataclass
class BlendConfig:
    market_weight: float = 0.5  # w: weight on market odds where they exist
    adjust_ridge: float = 0.03  # κ: penalty on market adjustments (inf = no adjustment)


# ------------------------------------------------------------------ vectorised maths


def outcome_probs(lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles (home, draw, away) probabilities for arrays of expected goals. Shape (n, 3)."""
    g = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(g[None, :], lam[:, None])
    pa = poisson.pmf(g[None, :], mu[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= 1 - lam * mu * rho
    m[:, 0, 1] *= 1 + lam * rho
    m[:, 1, 0] *= 1 + mu * rho
    m[:, 1, 1] *= 1 - rho
    m /= m.sum(axis=(1, 2), keepdims=True)
    home = np.tril(np.ones((MAX_GOALS + 1,) * 2), -1)
    return np.stack(
        [
            (m * home).sum(axis=(1, 2)),
            np.trace(m, axis1=1, axis2=2),
            (m * home.T).sum(axis=(1, 2)),
        ],
        axis=1,
    )


def blend(p_model: np.ndarray, p_market: np.ndarray, w: float) -> np.ndarray:
    """Logarithmic pool of two (n, 3) probability arrays."""
    logp = (1 - w) * np.log(np.clip(p_model, 1e-12, 1)) + w * np.log(np.clip(p_market, 1e-12, 1))
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


# ------------------------------------------------------------ market adjustment


def _strength_arrays(model: DixonColes, home, away, promoted) -> tuple[np.ndarray, ...]:
    hs = np.array([model._strength(t, t in promoted) for t in home]).reshape(-1, 2)
    as_ = np.array([model._strength(t, t in promoted) for t in away]).reshape(-1, 2)
    return hs[:, 0], hs[:, 1], as_[:, 0], as_[:, 1]


def market_adjust(
    model: DixonColes,
    games: pd.DataFrame,
    ridge: float,
    promoted=frozenset(),
) -> DixonColes:
    """Return a copy of `model` with team strengths nudged towards the market.

    `games` has home_id, away_id and market probabilities m_home, m_draw, m_away.
    """
    if not np.isfinite(ridge) or games.empty:
        return model
    d = model.divisions.index("E0")
    c_h, c_a = model.base[d] + model.home_adv[d], model.base[d]
    ha, hd, aa, ad = _strength_arrays(model, games.home_id, games.away_id, promoted)
    teams = sorted(set(games.home_id) | set(games.away_id))
    idx = {t: i for i, t in enumerate(teams)}
    hi = games.home_id.map(idx).to_numpy()
    ai = games.away_id.map(idx).to_numpy()
    target = games[MARKET_COLS].to_numpy()

    def loss(delta):
        lam = np.exp(c_h + ha + delta[hi] / 2 - (ad + delta[ai] / 2))
        mu = np.exp(c_a + aa + delta[ai] / 2 - (hd + delta[hi] / 2))
        p = outcome_probs(lam, mu, model.rho)
        return -np.sum(target * np.log(p)) + ridge * np.sum(delta**2)

    res = minimize(loss, np.zeros(len(teams)), method="L-BFGS-B")
    adjusted = copy.deepcopy(model)
    adjusted.attack = adjusted.attack.copy()
    adjusted.defence = adjusted.defence.copy()
    adjusted.market_delta = dict(zip(teams, res.x, strict=True))
    for t, delta in adjusted.market_delta.items():
        if t in adjusted.teams:
            i = adjusted.teams.index(t)
            adjusted.attack[i] += delta / 2
            adjusted.defence[i] += delta / 2
    return adjusted


def model_probs(model: DixonColes, games: pd.DataFrame, promoted=frozenset()) -> np.ndarray:
    d = model.divisions.index("E0")
    ha, hd, aa, ad = _strength_arrays(model, games.home_id, games.away_id, promoted)
    lam = np.exp(model.base[d] + model.home_adv[d] + ha - ad)
    mu = np.exp(model.base[d] + aa - hd)
    return outcome_probs(lam, mu, model.rho)


# ---------------------------------------------------------------- live forecast


def forecast(
    model: DixonColes,
    fixtures: pd.DataFrame,
    market: pd.DataFrame,
    config: BlendConfig,
    promoted=frozenset(),
) -> pd.DataFrame:
    """Probabilities for `fixtures` (home_id, away_id, ...).

    `market` is the bookmaker consensus (home_id, away_id, p_home, p_draw, p_away, n_books).
    Adds p_home/p_draw/p_away, p_model_* columns and `source` ("blend" or "model").
    """
    mk = market.rename(columns=dict(zip(PROB_COLS, MARKET_COLS, strict=True)))
    df = fixtures.merge(
        mk[["home_id", "away_id", *MARKET_COLS, "n_books"]], on=["home_id", "away_id"], how="left"
    )
    has_odds = df.m_home.notna().to_numpy()
    adjusted = market_adjust(model, df[has_odds], config.adjust_ridge, promoted)

    p_model = model_probs(adjusted, df, promoted)
    p = p_model.copy()
    if has_odds.any():
        p[has_odds] = blend(
            p_model[has_odds], df.loc[has_odds, MARKET_COLS].to_numpy(), config.market_weight
        )
    df[["p_model_home", "p_model_draw", "p_model_away"]] = p_model
    df[PROB_COLS] = p
    df["source"] = np.where(has_odds, "blend", "model")
    return df


def upcoming(conn, settings) -> tuple[pd.DataFrame, DixonColes]:
    """Fit on everything played so far and forecast every remaining fixture this season."""
    from lms import data
    from lms.market import consensus
    from lms.model import FitConfig, promoted_teams

    matches = data.training_matches(conn, settings.season)
    fixtures = data.upcoming_fixtures(conn, settings.season)
    pl_teams = set(fixtures.home_id) | set(fixtures.away_id)
    promoted = promoted_teams(matches, settings.season, pl_teams)
    model = DixonColes(FitConfig(settings.time_decay_half_life_days, settings.ridge)).fit(matches)
    odds = data.latest_odds(conn)
    market = consensus(odds) if len(odds) else pd.DataFrame(columns=["home_id", "away_id"])
    cfg = BlendConfig(settings.market_weight, settings.adjust_ridge)
    return forecast(model, fixtures, market, cfg, promoted), model


def team_view(fc: pd.DataFrame) -> pd.DataFrame:
    """One row per (fixture, team): the LMS question 'does this team win?'."""
    cols = ["fixture_id", "gw", "kickoff_utc", "source"]
    home = fc[cols].assign(
        team_id=fc.home_id, opponent_id=fc.away_id, is_home=True, p_win=fc.p_home
    )
    away = fc[cols].assign(
        team_id=fc.away_id, opponent_id=fc.home_id, is_home=False, p_win=fc.p_away
    )
    return pd.concat([home, away], ignore_index=True).sort_values(
        ["gw", "kickoff_utc", "team_id"], ignore_index=True
    )


# ------------------------------------------------------------------- evaluation


def evaluate(
    matches: pd.DataFrame,
    test_seasons: list[str],
    model_config,
    weights=(0.0, 0.5, 0.7, 0.8, 0.9, 1.0),
    ridges=(np.inf, 3.0, 1.0, 0.3),
    lookahead_weeks: int = 2,
    horizon_weeks: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward tuning of the blend weight (near) and adjustment ridge (far).

    Each week: fit on the past, treat the closing odds of the next `lookahead_weeks` as the
    odds you'd see, adjust ratings for each ridge, then
      - near: score this week's games for every (ridge, weight) blend
      - far: score games 2..horizon weeks out (no odds) for every ridge.
    Returns (near, far) tables of log-loss.
    """
    from lms.calibration import walk_forward_weeks  # shared week bucketing
    from lms.market import devig
    from lms.model import promoted_teams

    m = matches.assign(date=pd.to_datetime(matches["date"]))
    test = m[(m.division == "E0") & m.season.isin(test_seasons)].copy()
    test[MARKET_COLS] = np.column_stack(devig(test.odds_home, test.odds_draw, test.odds_away))
    test["outcome"] = np.select(
        [test.home_goals > test.away_goals, test.home_goals == test.away_goals], [0, 1], 2
    )
    test["week"] = walk_forward_weeks(test.date)
    week_list = sorted(test.week.unique())
    promoted = {s: promoted_teams(m, s) for s in test_seasons}

    near_ll: dict[tuple[float, float], list[float]] = {}
    far_ll: dict[float, list[float]] = {}
    model = None
    for k, week in enumerate(week_list):
        model = DixonColes(model_config).fit(m, as_of=week, warm_start=model)
        season = test.loc[test.week == week, "season"].iloc[0]
        seen = test[test.week.isin(week_list[k : k + lookahead_weeks]) & (test.season == season)]
        now = test[test.week == week]
        later = test[
            test.week.isin(week_list[k + lookahead_weeks : k + horizon_weeks])
            & (test.season == season)
        ]
        for rg in ridges:
            adj = market_adjust(model, seen, rg, promoted[season])
            p_now = model_probs(adj, now, promoted[season])
            for w in weights:
                p = blend(p_now, now[MARKET_COLS].to_numpy(), w)
                near_ll.setdefault((rg, w), []).extend(-np.log(p[np.arange(len(now)), now.outcome]))
            if len(later):
                p = model_probs(adj, later, promoted[season])
                far_ll.setdefault(rg, []).extend(-np.log(p[np.arange(len(later)), later.outcome]))

    near = pd.DataFrame(
        [
            {"adjust_ridge": rg, "market_weight": w, "log_loss": np.mean(v), "n": len(v)}
            for (rg, w), v in near_ll.items()
        ]
    ).sort_values("log_loss", ignore_index=True)
    far = pd.DataFrame(
        [{"adjust_ridge": rg, "log_loss": np.mean(v), "n": len(v)} for rg, v in far_ll.items()]
    ).sort_values("log_loss", ignore_index=True)
    return near, far
