"""Dixon-Coles match model with exponential time decay.

Goals are Poisson with
    log λ_home = base[div] + home_adv[div] + attack[home] - defence[away]
    log λ_away = base[div] + attack[away] - defence[home]
plus the Dixon-Coles correction (rho) for the low-scoring results 0-0, 1-0, 0-1 and 1-1,
which plain Poisson gets wrong and which decide draws.

Each division has its own scoring level and home advantage, but team strengths are shared,
so clubs that move between the Premier League and Championship link the two scales. That is
how promoted clubs get a meaningful Premier League rating before they've played much.

Promoted clubs are systematically weaker in the Premier League than their Championship
results imply. Two extra parameters learn that drop (attack and defence) from earlier promoted
clubs and apply it to a club's first PL season after promotion.

Older matches are down-weighted by exp(-ln2 · age / half_life). A small ridge penalty keeps
ratings of teams with few matches near average and makes the fit unique.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10
RHO_BOUNDS = (-0.2, 0.2)


def _prev_season(season: str) -> str:
    y = int(season[:2])
    return f"{y - 1:02d}{y:02d}"


def promoted_teams(matches: pd.DataFrame, season: str, pl_teams=None) -> set[str]:
    """Clubs in the Premier League in `season` that were in the Championship the season before.

    `pl_teams` overrides who is in the PL (useful before a season's first match is played).
    """
    if pl_teams is None:
        cur = matches[(matches.season == season) & (matches.division == "E0")]
        pl_teams = set(cur.home_id) | set(cur.away_id)
    prev = matches[(matches.season == _prev_season(season)) & (matches.division == "E1")]
    return set(pl_teams) & (set(prev.home_id) | set(prev.away_id))


@dataclass
class FitConfig:
    half_life_days: float = 180.0
    ridge: float = 1.0  # L2 penalty on attack/defence (in units of weighted matches)


class DixonColes:
    def __init__(self, config: FitConfig | None = None):
        self.config = config or FitConfig()
        self.teams: list[str] = []
        self.divisions: list[str] = []
        self.attack: np.ndarray = np.array([])
        self.defence: np.ndarray = np.array([])
        self.base: np.ndarray = np.array([])
        self.home_adv: np.ndarray = np.array([])
        self.rho: float = 0.0
        self.promoted_attack: float = 0.0  # attack lost in first PL season after promotion
        self.promoted_defence: float = 0.0  # defence lost in first PL season after promotion
        self.as_of: pd.Timestamp | None = None
        self.n_matches = 0

    # ------------------------------------------------------------------ fitting

    def fit(
        self,
        matches: pd.DataFrame,
        as_of: str | pd.Timestamp | None = None,
        warm_start: DixonColes | None = None,
    ) -> DixonColes:
        """Fit on matches strictly before `as_of` (default: after the last match).

        `matches` needs columns: date, division, home_id, away_id, home_goals, away_goals.
        """
        dates = pd.to_datetime(matches["date"])
        as_of = pd.Timestamp(as_of) if as_of is not None else dates.max() + pd.Timedelta(days=1)
        df = matches[dates < as_of]
        if df.empty:
            raise ValueError(f"No matches before {as_of.date()}")

        self.as_of = as_of
        self.n_matches = len(df)
        self.teams = sorted(set(df["home_id"]) | set(df["away_id"]))
        self.divisions = sorted(df["division"].unique())
        t_idx = {t: i for i, t in enumerate(self.teams)}
        d_idx = {d: i for i, d in enumerate(self.divisions)}

        h = df["home_id"].map(t_idx).to_numpy()
        a = df["away_id"].map(t_idx).to_numpy()
        div = df["division"].map(d_idx).to_numpy()
        x = df["home_goals"].to_numpy(dtype=float)
        y = df["away_goals"].to_numpy(dtype=float)
        age = (as_of - pd.to_datetime(df["date"])).dt.days.to_numpy(dtype=float)
        w = np.exp(-np.log(2) * age / self.config.half_life_days)

        promoted = {(s, t) for s in df["season"].unique() for t in promoted_teams(matches, s)}
        in_pl = (df["division"] == "E0").to_numpy()
        hp = in_pl & np.array(
            [(s, t) in promoted for s, t in zip(df.season, df.home_id, strict=True)]
        )
        ap = in_pl & np.array(
            [(s, t) in promoted for s, t in zip(df.season, df.away_id, strict=True)]
        )

        n_t, n_d = len(self.teams), len(self.divisions)
        x0 = self._initial_params(n_t, n_d, warm_start, df, d_idx)
        bounds = [(None, None)] * (2 * n_t + 2 * n_d) + [(-1, 1), (-1, 1), RHO_BOUNDS]

        res = minimize(
            _neg_log_lik,
            x0,
            args=(
                h,
                a,
                div,
                hp.astype(float),
                ap.astype(float),
                x,
                y,
                w,
                n_t,
                n_d,
                self.config.ridge,
            ),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if not res.success:
            raise RuntimeError(f"Dixon-Coles fit failed: {res.message}")
        self._unpack(res.x, n_t, n_d)
        return self

    def _initial_params(self, n_t, n_d, warm, df, d_idx) -> np.ndarray:
        params = np.zeros(2 * n_t + 2 * n_d + 3)
        goals = (df["home_goals"] + df["away_goals"]).groupby(df["division"]).mean() / 2
        for d, i in d_idx.items():
            params[2 * n_t + i] = np.log(max(goals.get(d, 1.3), 0.1))
            params[2 * n_t + n_d + i] = 0.2
        if warm is not None:
            for i, t in enumerate(self.teams):
                if t in warm.teams:
                    j = warm.teams.index(t)
                    params[i], params[n_t + i] = warm.attack[j], warm.defence[j]
        return params

    def _unpack(self, p: np.ndarray, n_t: int, n_d: int) -> None:
        self.attack = p[:n_t]
        self.defence = p[n_t : 2 * n_t]
        self.base = p[2 * n_t : 2 * n_t + n_d]
        self.home_adv = p[2 * n_t + n_d : 2 * n_t + 2 * n_d]
        self.promoted_attack = float(p[-3])
        self.promoted_defence = float(p[-2])
        self.rho = float(p[-1])

    # --------------------------------------------------------------- predicting

    def expected_goals(
        self, home: str, away: str, division: str = "E0", promoted=frozenset()
    ) -> tuple[float, float]:
        d = self.divisions.index(division)
        home_att, home_def = self._strength(home, division == "E0" and home in promoted)
        away_att, away_def = self._strength(away, division == "E0" and away in promoted)
        lam = np.exp(self.base[d] + self.home_adv[d] + home_att - away_def)
        mu = np.exp(self.base[d] + away_att - home_def)
        return float(lam), float(mu)

    def _strength(self, team: str, promoted: bool = False) -> tuple[float, float]:
        """(attack, defence) for a team; unseen teams get average (0, 0)."""
        att = dfn = 0.0
        if team in self.teams:
            i = self.teams.index(team)
            att, dfn = float(self.attack[i]), float(self.defence[i])
        if promoted:
            att -= self.promoted_attack
            dfn -= self.promoted_defence
        return att, dfn

    def score_matrix(
        self, home: str, away: str, division: str = "E0", promoted=frozenset()
    ) -> np.ndarray:
        lam, mu = self.expected_goals(home, away, division, promoted)
        goals = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
        rho = self.rho
        m[0, 0] *= 1 - lam * mu * rho
        m[0, 1] *= 1 + lam * rho
        m[1, 0] *= 1 + mu * rho
        m[1, 1] *= 1 - rho
        return m / m.sum()

    def predict(
        self, home: str, away: str, division: str = "E0", promoted=frozenset()
    ) -> tuple[float, float, float]:
        """(P(home win), P(draw), P(away win)). `promoted`: clubs in their first PL season."""
        m = self.score_matrix(home, away, division, promoted)
        return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum())

    def predict_frame(
        self, fixtures: pd.DataFrame, division: str = "E0", promoted=frozenset()
    ) -> pd.DataFrame:
        probs = [
            self.predict(h, a, division, promoted)
            for h, a in zip(fixtures.home_id, fixtures.away_id, strict=True)
        ]
        out = fixtures.copy()
        out[["p_home", "p_draw", "p_away"]] = probs
        return out

    def ratings(self, reference: list[str] | None = None) -> pd.DataFrame:
        """Team ratings on the Premier League scale, best first.

        `rating` is expected goal difference per game on neutral ground against an average
        member of `reference` (default: every team in the fit, PL and Championship).
        """
        d = self.divisions.index("E0") if "E0" in self.divisions else 0
        base = self.base[d] + self.home_adv[d] / 2
        ref = np.isin(self.teams, reference) if reference is not None else slice(None)
        avg_att, avg_def = self.attack[ref].mean(), self.defence[ref].mean()
        gf = np.exp(base + self.attack - avg_def)
        ga = np.exp(base + avg_att - self.defence)
        df = pd.DataFrame(
            {
                "team_id": self.teams,
                "attack": self.attack,
                "defence": self.defence,
                "xg_for": gf,
                "xg_against": ga,
                "rating": gf - ga,
            }
        )
        return df.sort_values("rating", ascending=False, ignore_index=True)


def _neg_log_lik(params, h, a, div, hp, ap, x, y, w, n_t, n_d, ridge):
    att = params[:n_t]
    dfn = params[n_t : 2 * n_t]
    base = params[2 * n_t : 2 * n_t + n_d]
    hadv = params[2 * n_t + n_d : 2 * n_t + 2 * n_d]
    p_att, p_def, rho = params[-3], params[-2], params[-1]

    # Promoted clubs lose p_att attack and p_def defence in their first PL season.
    eta_h = base[div] + hadv[div] + att[h] - dfn[a] - p_att * hp + p_def * ap
    eta_a = base[div] + att[a] - dfn[h] - p_att * ap + p_def * hp
    lam, mu = np.exp(eta_h), np.exp(eta_a)

    # Dixon-Coles tau and its derivatives wrt eta_h, eta_a, rho
    tau = np.ones_like(lam)
    dt_h = np.zeros_like(lam)
    dt_a = np.zeros_like(lam)
    dt_r = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    lm = lam[m00] * mu[m00]
    tau[m00] = 1 - lm * rho
    dt_h[m00] = dt_a[m00] = -lm * rho
    dt_r[m00] = -lm

    tau[m01] = 1 + lam[m01] * rho
    dt_h[m01] = lam[m01] * rho
    dt_r[m01] = lam[m01]

    tau[m10] = 1 + mu[m10] * rho
    dt_a[m10] = mu[m10] * rho
    dt_r[m10] = mu[m10]

    tau[m11] = 1 - rho
    dt_r[m11] = -1.0

    tau = np.maximum(tau, 1e-10)
    ll = w * (np.log(tau) + x * eta_h - lam + y * eta_a - mu)

    g_h = w * (x - lam + dt_h / tau)
    g_a = w * (y - mu + dt_a / tau)

    grad = np.empty_like(params)
    grad[:n_t] = np.bincount(h, g_h, n_t) + np.bincount(a, g_a, n_t)
    grad[n_t : 2 * n_t] = -np.bincount(a, g_h, n_t) - np.bincount(h, g_a, n_t)
    grad[2 * n_t : 2 * n_t + n_d] = np.bincount(div, g_h + g_a, n_d)
    grad[2 * n_t + n_d : 2 * n_t + 2 * n_d] = np.bincount(div, g_h, n_d)
    grad[-3] = -np.sum(g_h * hp + g_a * ap)
    grad[-2] = np.sum(g_h * ap + g_a * hp)
    grad[-1] = np.sum(w * dt_r / tau)

    penalty = 0.5 * ridge * (np.sum(att**2) + np.sum(dfn**2))
    grad[:n_t] -= ridge * att
    grad[n_t : 2 * n_t] -= ridge * dfn

    return -(ll.sum() - penalty), -grad
