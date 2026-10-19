"""Last Man Standing pick optimiser.

Rules: pick one team per gameweek, never the same team twice, and a pick that fails to win
(draw or loss) costs a life. You are out when lives run out.

The plan picks a distinct team for each of the next H gameweeks to maximise the chance of
still being alive after H weeks:

- With 1 life that is the product of win probabilities, so maximising it is an assignment
  problem (minimise Σ -log p) and the Hungarian algorithm solves it exactly.
- With more lives the survival probability is a Poisson-binomial tail, which isn't separable.
  Several assignment solutions (for -log p and flatter stand-ins like -log(p + a) and -p) seed
  a local search (swap two weeks' teams, or bring in an unused team), and the best result wins.
  Checked against brute force in the tests.

Saving strong teams for later falls out of this: City is only spent this week if no other
arrangement of the next H weeks does better. No hand-picked "elite" lists needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

MIN_P = 1e-9


@dataclass
class Board:
    """Win probabilities for candidate teams over the planning window."""

    teams: list[str]
    gws: list[int]
    p: np.ndarray  # (teams, gws) win probability, 0 = not pickable
    opponent: np.ndarray  # (teams, gws) opponent id or ""
    is_home: np.ndarray  # (teams, gws) bool
    derby: np.ndarray  # (teams, gws) rivalry intensity 0-1
    source: np.ndarray  # (teams, gws) "blend"/"model"/""


def build_board(
    team_view: pd.DataFrame,
    start_gw: int,
    horizon: int,
    used: set[str] = frozenset(),
    derbies: dict[frozenset, float] | None = None,
    avoid_derbies: bool = False,
    derby_min_intensity: float = 0.8,
    derby_penalty: float = 0.0,
    min_win_prob: float = 0.0,
) -> Board:
    """Arrange per-team win probabilities into a teams x gameweeks board.

    In a double gameweek a team's first fixture counts. A blank gameweek leaves p = 0.
    """
    derbies = derbies or {}
    gws = list(range(start_gw, start_gw + horizon))
    tv = team_view[team_view.gw.isin(gws) & ~team_view.team_id.isin(used)]
    tv = tv.sort_values("kickoff_utc").drop_duplicates(["gw", "team_id"])
    gws = [g for g in gws if g in set(tv.gw)]  # drop gameweeks past the season's end
    teams = sorted(tv.team_id.unique())
    shape = (len(teams), len(gws))
    board = Board(
        teams=teams,
        gws=gws,
        p=np.zeros(shape),
        opponent=np.full(shape, "", dtype=object),
        is_home=np.zeros(shape, dtype=bool),
        derby=np.zeros(shape),
        source=np.full(shape, "", dtype=object),
    )
    t_idx = {t: i for i, t in enumerate(teams)}
    g_idx = {g: j for j, g in enumerate(gws)}
    for r in tv.itertuples():
        i, j = t_idx[r.team_id], g_idx[r.gw]
        intensity = derbies.get(frozenset((r.team_id, r.opponent_id)), 0.0)
        p = r.p_win * (1 - derby_penalty * intensity)
        if (avoid_derbies and intensity >= derby_min_intensity) or p < min_win_prob:
            p = 0.0
        board.p[i, j] = p
        board.opponent[i, j] = r.opponent_id
        board.is_home[i, j] = r.is_home
        board.derby[i, j] = intensity
        board.source[i, j] = r.source
    return board


# ------------------------------------------------------------------ survival maths


def failure_distribution(win_probs) -> np.ndarray:
    """P(exactly k failed picks) for k = 0..n (Poisson-binomial)."""
    dist = np.array([1.0])
    for p in win_probs:
        dist = np.append(dist * p, 0.0) + np.insert(dist * (1 - p), 0, 0.0)
    return dist


def survival(win_probs, lives: int) -> float:
    """Probability of still being in after these picks, starting with `lives`.

    Same as failure_distribution(...)[:lives].sum(), but only tracks 0..lives-1 failures,
    which keeps the optimiser's inner loop fast.
    """
    dist = [1.0] + [0.0] * (lives - 1)
    for p in win_probs:
        q = 1.0 - p
        for k in range(lives - 1, 0, -1):
            dist[k] = dist[k] * p + dist[k - 1] * q
        dist[0] *= p
    return sum(dist)


def alive_by_week(win_probs, lives: int) -> list[float]:
    return [survival(win_probs[: k + 1], lives) for k in range(len(win_probs))]


# -------------------------------------------------------------------- optimiser


# Stand-in costs for seeding the multi-life search. With spare lives a weak week hurts less,
# so flatter transforms of p than -log p can be closer to the true objective.
_SEED_COSTS = (
    lambda p: -np.log(np.maximum(p, MIN_P)),
    lambda p: -np.log(p + 0.1),
    lambda p: -np.log(p + 0.3),
    lambda p: -np.log(p + 1.0),
    lambda p: -p,
)


def _assignment(p: np.ndarray, fixed: dict[int, int], cost=None) -> list[int]:
    """Min Σ cost via Hungarian (default -log p = max Π p). `fixed` maps week -> team."""
    cost = -np.log(np.maximum(p, MIN_P)) if cost is None else cost
    free_weeks = [j for j in range(p.shape[1]) if j not in fixed]
    free_teams = [i for i in range(p.shape[0]) if i not in fixed.values()]
    plan = [-1] * p.shape[1]
    for j, i in fixed.items():
        plan[j] = i
    if free_weeks:
        rows, cols = linear_sum_assignment(cost[np.ix_(free_teams, free_weeks)].T)
        for r, c in zip(rows, cols, strict=True):
            plan[free_weeks[r]] = free_teams[c]
    return plan


def _plan_probs(p: np.ndarray, plan: list[int]) -> list[float]:
    return [float(p[i, j]) for j, i in enumerate(plan)]


def _local_search(p: np.ndarray, plan: list[int], lives: int, fixed: dict[int, int]) -> list[int]:
    best = survival(_plan_probs(p, plan), lives)
    free = [j for j in range(len(plan)) if j not in fixed]
    improved = True
    while improved:
        improved = False
        unused = [i for i in range(p.shape[0]) if i not in plan]
        # swap two weeks' teams; replace a week's team with an unused one; move a team to
        # another week and backfill its old week with an unused team.
        moves = [("swap", a, b, None) for a in free for b in free if a < b]
        moves += [("new", a, None, t) for a in free for t in unused]
        moves += [("move", a, b, t) for a in free for b in free if a != b for t in unused]
        for kind, a, b, t in moves:
            cand = plan.copy()
            if kind == "swap":
                cand[a], cand[b] = cand[b], cand[a]
            elif kind == "new":
                cand[a] = t
            else:
                cand[b], cand[a] = cand[a], t
            s = survival(_plan_probs(p, cand), lives)
            if s > best + 1e-12:
                plan, best, improved = cand, s, True
                break
    return plan


def optimise(p: np.ndarray, lives: int, fixed: dict[int, int] | None = None) -> list[int]:
    """Best distinct-team plan for every week of `p` (teams x weeks)."""
    fixed = fixed or {}
    if p.shape[0] < p.shape[1]:
        raise ValueError("More gameweeks than available teams; shorten the horizon.")
    if lives == 1:
        return _assignment(p, fixed)
    best_plan, best = None, -1.0
    for cost_fn in _SEED_COSTS:
        plan = _local_search(p, _assignment(p, fixed, cost_fn(p)), lives, fixed)
        s = survival(_plan_probs(p, plan), lives)
        if s > best + 1e-12:
            best_plan, best = plan, s
    return best_plan


# ------------------------------------------------------------------- outputs


@dataclass
class Recommendation:
    options: pd.DataFrame  # this week's candidates, best first
    path: pd.DataFrame  # recommended plan, week by week
    lives: int
    board: Board


def recommend(board: Board, lives: int) -> Recommendation:
    """Rank every pickable team this week by the best plan that starts with it."""
    n_weeks = min(len(board.gws), len(board.teams))
    p = board.p[:, :n_weeks]
    rows, plans = [], {}
    for i, team in enumerate(board.teams):
        if p[i, 0] <= 0:
            continue
        plan = optimise(p, lives, fixed={0: i})
        probs = _plan_probs(p, plan)
        plans[team] = plan
        rows.append(
            {
                "team_id": team,
                "opponent_id": board.opponent[i, 0],
                "is_home": bool(board.is_home[i, 0]),
                "derby": float(board.derby[i, 0]),
                "p_win": float(p[i, 0]),
                "plan_survival": survival(probs, lives),
            }
        )
    if not rows:
        raise ValueError("No pickable teams this gameweek.")
    options = pd.DataFrame(rows).sort_values("plan_survival", ascending=False, ignore_index=True)
    options["vs_best"] = options.plan_survival - options.plan_survival.iloc[0]

    best_plan = plans[options.team_id.iloc[0]]
    probs = _plan_probs(p, best_plan)
    path = pd.DataFrame(
        {
            "gw": board.gws[:n_weeks],
            "team_id": [board.teams[i] for i in best_plan],
            "opponent_id": [board.opponent[i, j] for j, i in enumerate(best_plan)],
            "is_home": [bool(board.is_home[i, j]) for j, i in enumerate(best_plan)],
            "derby": [float(board.derby[i, j]) for j, i in enumerate(best_plan)],
            "source": [board.source[i, j] for j, i in enumerate(best_plan)],
            "p_win": probs,
            "alive": alive_by_week(probs, lives),
        }
    )
    return Recommendation(options=options, path=path, lives=lives, board=board)


def load_derbies(path) -> dict[frozenset, float]:
    from lms.teams import team_id

    df = pd.read_csv(path)
    return {
        frozenset((team_id(a, "derbies.csv"), team_id(b, "derbies.csv"))): float(i)
        for a, b, i in df.itertuples(index=False)
    }
