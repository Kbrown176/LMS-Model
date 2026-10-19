"""Tests for blending, market adjustment and fixture forecasts."""

import numpy as np
import pandas as pd
import pytest

from lms.forecast import BlendConfig, blend, forecast, market_adjust, model_probs, team_view
from lms.model import DixonColes, FitConfig
from tests.test_model import simulate


@pytest.fixture(scope="module")
def model():
    return DixonColes(FitConfig(half_life_days=1e6, ridge=0.01)).fit(simulate(n_rounds=20))


def test_blend_endpoints_and_normalisation():
    pm = np.array([[0.5, 0.3, 0.2]])
    pk = np.array([[0.7, 0.2, 0.1]])
    np.testing.assert_allclose(blend(pm, pk, 0.0), pm)
    np.testing.assert_allclose(blend(pm, pk, 1.0), pk)
    mid = blend(pm, pk, 0.5)
    assert mid.sum() == pytest.approx(1.0)
    assert pm[0, 0] < mid[0, 0] < pk[0, 0]


def test_market_adjust_moves_towards_market(model):
    games = pd.DataFrame({"home_id": ["e"], "away_id": ["a"]})
    before = model_probs(model, games)[0, 0]
    target = games.assign(m_home=0.6, m_draw=0.25, m_away=0.15)  # market rates "e" far higher
    adjusted = market_adjust(model, target, ridge=0.1)
    after = model_probs(adjusted, games)[0, 0]
    assert before < after <= 0.6 + 1e-6
    assert adjusted.market_delta["e"] > 0 > adjusted.market_delta["a"]
    # Adjustment carries over to fixtures without odds.
    other = pd.DataFrame({"home_id": ["e"], "away_id": ["c"]})
    assert model_probs(adjusted, other)[0, 0] > model_probs(model, other)[0, 0]


def test_market_adjust_infinite_ridge_is_noop(model):
    games = pd.DataFrame({"home_id": ["e"], "away_id": ["a"], "m_home": [0.6]})
    assert market_adjust(model, games, ridge=np.inf) is model


def test_forecast_marks_sources(model):
    fixtures = pd.DataFrame(
        {
            "fixture_id": [1, 2],
            "gw": [8, 9],
            "kickoff_utc": ["2026-10-24T14:00:00Z", "2026-10-31T14:00:00Z"],
            "home_id": ["a", "b"],
            "away_id": ["e", "d"],
        }
    )
    market = pd.DataFrame(
        {"home_id": ["a"], "away_id": ["e"], "p_home": [0.7], "p_draw": [0.2], "p_away": [0.1]}
    ).assign(n_books=5)
    fc = forecast(model, fixtures, market, BlendConfig())
    assert list(fc.source) == ["blend", "model"]
    np.testing.assert_allclose(fc[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)

    tv = team_view(fc)
    assert len(tv) == 4
    assert tv.loc[(tv.team_id == "e"), "p_win"].iloc[0] == pytest.approx(fc.p_away.iloc[0])


def test_forecast_without_any_odds(model):
    fixtures = pd.DataFrame(
        {
            "fixture_id": [1],
            "gw": [8],
            "kickoff_utc": ["2026-10-24T14:00:00Z"],
            "home_id": ["a"],
            "away_id": ["e"],
        }
    )
    for market in (None, pd.DataFrame()):
        fc = forecast(model, fixtures, market, BlendConfig())
        assert list(fc.source) == ["model"]
        assert fc[["p_home", "p_draw", "p_away"]].sum(axis=1).iloc[0] == pytest.approx(1.0)
