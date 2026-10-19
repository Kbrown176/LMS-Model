"""Tests for the Dixon-Coles model and odds helpers, on simulated data."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from lms.market import consensus, devig
from lms.model import DixonColes, FitConfig, _neg_log_lik

TRUE_ATTACK = {"a": 0.5, "b": 0.2, "c": 0.0, "d": -0.2, "e": -0.5}
TRUE_DEFENCE = {"a": 0.4, "b": 0.2, "c": 0.0, "d": -0.2, "e": -0.4}
BASE, HOME = 0.1, 0.25


def simulate(n_rounds=40, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = list(TRUE_ATTACK)
    rows = []
    day = pd.Timestamp("2024-01-01")
    for r in range(n_rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = np.exp(BASE + HOME + TRUE_ATTACK[h] - TRUE_DEFENCE[a])
                mu = np.exp(BASE + TRUE_ATTACK[a] - TRUE_DEFENCE[h])
                rows.append(
                    {
                        "date": (day + pd.Timedelta(days=r)).strftime("%Y-%m-%d"),
                        "season": "2324",
                        "division": "E0",
                        "home_id": h,
                        "away_id": a,
                        "home_goals": rng.poisson(lam),
                        "away_goals": rng.poisson(mu),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def matches():
    return simulate()


@pytest.fixture(scope="module")
def fitted(matches):
    return DixonColes(FitConfig(half_life_days=1e6, ridge=0.01)).fit(matches)


def test_analytic_gradient_matches_numeric(matches):
    rng = np.random.default_rng(1)
    t_idx = {t: i for i, t in enumerate(sorted(TRUE_ATTACK))}
    sample = matches.sample(200, random_state=0)
    args = (
        sample.home_id.map(t_idx).to_numpy(),
        sample.away_id.map(t_idx).to_numpy(),
        np.zeros(len(sample), dtype=int),
        (sample.home_id == "e").to_numpy(float),  # pretend "e" is promoted
        (sample.away_id == "e").to_numpy(float),
        sample.home_goals.to_numpy(float),
        sample.away_goals.to_numpy(float),
        rng.uniform(0.2, 1.0, len(sample)),
        5,
        1,
        0.5,
    )
    x0 = np.concatenate([rng.normal(0, 0.3, 12), [0.1, 0.15, -0.08]])
    err = check_grad(lambda p: _neg_log_lik(p, *args)[0], lambda p: _neg_log_lik(p, *args)[1], x0)
    assert err < 1e-4


def test_recovers_true_strengths(fitted):
    att = dict(zip(fitted.teams, fitted.attack - fitted.attack.mean(), strict=True))
    dfn = dict(zip(fitted.teams, fitted.defence - fitted.defence.mean(), strict=True))
    for t in TRUE_ATTACK:
        assert att[t] == pytest.approx(TRUE_ATTACK[t], abs=0.1)
        assert dfn[t] == pytest.approx(TRUE_DEFENCE[t], abs=0.1)
    assert fitted.home_adv[0] == pytest.approx(HOME, abs=0.06)


def test_probabilities_are_valid_and_ordered(fitted):
    ph, pdraw, pa = fitted.predict("a", "e")
    assert ph + pdraw + pa == pytest.approx(1.0)
    assert ph > 0.6 > pa
    # Home advantage: same pairing is better for the home side.
    assert fitted.predict("c", "d")[0] > fitted.predict("d", "c")[2]


def test_ratings_order(fitted):
    assert list(fitted.ratings().team_id) == ["a", "b", "c", "d", "e"]


def test_fit_excludes_matches_on_or_after_as_of(matches):
    dc = DixonColes().fit(matches, as_of="2024-01-11")
    assert dc.n_matches == 10 * 20


def test_unseen_team_gets_average_rating(fitted):
    ph, _, pa = fitted.predict("new_club", "c")
    assert 0 < pa < ph < 1  # average team at home vs average team


def test_devig_removes_margin():
    ph, pdraw, pa = devig([2.0], [3.4], [4.0])
    assert (ph + pdraw + pa)[0] == pytest.approx(1.0)
    assert ph[0] > pdraw[0] > pa[0]


def test_consensus_averages_books():
    odds = pd.DataFrame(
        {
            "home_id": ["a", "a"],
            "away_id": ["b", "b"],
            "commence_utc": ["2026-10-24T14:00:00Z"] * 2,
            "odds_home": [1.5, 1.6],
            "odds_draw": [4.0, 4.0],
            "odds_away": [6.0, 5.5],
        }
    )
    c = consensus(odds)
    assert len(c) == 1 and c.n_books[0] == 2
    assert c[["p_home", "p_draw", "p_away"]].sum(axis=1)[0] == pytest.approx(1.0)


def test_promoted_teams():
    from lms.model import promoted_teams

    m = pd.DataFrame(
        {
            "season": ["2526", "2526", "2627", "2627"],
            "division": ["E1", "E0", "E0", "E0"],
            "home_id": ["hull", "arsenal", "hull", "arsenal"],
            "away_id": ["leeds", "chelsea", "chelsea", "leeds"],
        }
    )
    assert promoted_teams(m, "2627") == {"hull", "leeds"}
    assert promoted_teams(m, "2627", pl_teams=["hull", "arsenal"]) == {"hull"}
