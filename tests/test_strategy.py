"""Optimiser tests: survival maths and plans checked against brute force."""

import itertools

import numpy as np
import pandas as pd
import pytest

from lms.strategy import build_board, failure_distribution, optimise, recommend, survival


def brute_force(p: np.ndarray, lives: int, first: int | None = None) -> float:
    n_teams, n_weeks = p.shape
    best = 0.0
    for plan in itertools.permutations(range(n_teams), n_weeks):
        if first is not None and plan[0] != first:
            continue
        best = max(best, survival([p[i, j] for j, i in enumerate(plan)], lives))
    return best


def test_failure_distribution():
    dist = failure_distribution([0.5, 0.5])
    np.testing.assert_allclose(dist, [0.25, 0.5, 0.25])
    assert survival([0.8, 0.5], 1) == pytest.approx(0.4)
    assert survival([0.8, 0.5], 2) == pytest.approx(1 - 0.2 * 0.5)


@pytest.mark.parametrize("lives", [1, 2, 3])
@pytest.mark.parametrize("seed", range(40))
def test_optimise_matches_brute_force(lives, seed):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.15, 0.85, size=(7, 5) if seed % 2 else (6, 4))
    p[rng.random(p.shape) < 0.1] = 0.0  # some blank / excluded slots
    plan = optimise(p, lives)
    assert len(set(plan)) == len(plan)  # distinct teams
    got = survival([p[i, j] for j, i in enumerate(plan)], lives)
    assert got == pytest.approx(brute_force(p, lives), abs=1e-9)


def test_optimiser_saves_strong_team_for_its_best_week():
    # Team 0 is good now but brilliant next week; team 1 is fine now and useless later.
    p = np.array([[0.70, 0.95], [0.65, 0.30]])
    plan = optimise(p, lives=1)
    assert plan == [1, 0]


def _team_view():
    rows = []
    fixtures = [
        (8, "arsenal", "everton", 0.70, 0.10),
        (8, "man_city", "man_united", 0.60, 0.20),
        (9, "arsenal", "tottenham", 0.55, 0.25),
        (9, "everton", "man_city", 0.10, 0.75),
    ]
    for gw, h, a, ph, pa in fixtures:
        for team, opp, home, p in ((h, a, True, ph), (a, h, False, pa)):
            rows.append(
                {
                    "fixture_id": len(rows),
                    "gw": gw,
                    "kickoff_utc": f"2026-10-{gw + 16}T14:00:00Z",
                    "source": "blend",
                    "team_id": team,
                    "opponent_id": opp,
                    "is_home": home,
                    "p_win": p,
                }
            )
    return pd.DataFrame(rows)


def test_board_respects_used_and_derbies():
    derbies = {frozenset(("man_city", "man_united")): 1.0}
    board = build_board(_team_view(), 8, 2, used={"arsenal"}, derbies=derbies)
    assert "arsenal" not in board.teams
    city = board.teams.index("man_city")
    assert board.derby[city, 0] == 1.0 and board.p[city, 0] == pytest.approx(0.6)

    avoided = build_board(_team_view(), 8, 2, derbies=derbies, avoid_derbies=True)
    assert avoided.p[avoided.teams.index("man_city"), 0] == 0.0

    penalised = build_board(_team_view(), 8, 2, derbies=derbies, derby_penalty=0.5)
    assert penalised.p[penalised.teams.index("man_city"), 0] == pytest.approx(0.3)


def test_recommend_ranks_options_and_builds_path():
    rec = recommend(build_board(_team_view(), 8, 2), lives=1)
    # City now (0.6) then Arsenal (0.55) = 0.33 beats Arsenal now (0.7) then City (0.75) = 0.525?
    # No: 0.7 * 0.75 = 0.525 > 0.33, so Arsenal first is best.
    assert rec.options.team_id.iloc[0] == "arsenal"
    assert list(rec.path.team_id) == ["arsenal", "man_city"]
    assert rec.path.alive.iloc[-1] == pytest.approx(0.7 * 0.75)
    assert rec.options.vs_best.iloc[0] == 0
    assert (rec.options.vs_best <= 0).all()
