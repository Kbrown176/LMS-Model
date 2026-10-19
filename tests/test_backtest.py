"""Backtest mechanics: round rebuilding and competition play."""

import numpy as np
import pandas as pd

from lms import backtest as bt
from lms.backtest import SeasonData, assign_rounds, biggest_favourite, play


def test_assign_rounds_splits_on_repeat_and_drops_fragments():
    teams = [f"t{i}" for i in range(16)]
    rows = []
    for rd in range(2):  # two full rounds of 8 matches
        for k in range(8):
            h, a = teams[2 * k], teams[2 * k + 1]
            rows.append({"date": f"2025-08-{10 + 7 * rd}", "home_id": h, "away_id": a})
    rows.append({"date": "2025-08-30", "home_id": "t0", "away_id": "t3"})  # rearranged game
    rd = assign_rounds(pd.DataFrame(rows))
    assert list(rd[:16]) == [1] * 8 + [2] * 8
    assert np.isnan(rd.iloc[16])


def _season():
    # Round 1: A wins, B loses. Round 2: A loses, B wins. Round 3: both win.
    won = {
        (1, "A"): True,
        (1, "B"): False,
        (2, "A"): False,
        (2, "B"): True,
        (3, "A"): True,
        (3, "B"): True,
        (3, "C"): True,
    }
    market = {(1, "A"): 0.7, (1, "B"): 0.6, (2, "A"): 0.8, (2, "B"): 0.5}
    market |= {(3, "A"): 0.5, (3, "B"): 0.5, (3, "C"): 0.6}
    return SeasonData("2526", [1, 2, 3], {}, won, market)


def test_biggest_favourite_play():
    rng = np.random.default_rng(0)
    # Favourite: A (wins), then B (only unused, wins), then C (wins).
    assert play(_season(), biggest_favourite, 1, 1, rng) == 3


def test_play_stops_when_lives_run_out():
    rng = np.random.default_rng(0)

    def always_b_then_a(data, rd, used, lives, rng):
        return "B" if "B" not in used else "A" if "A" not in used else None

    assert play(_season(), always_b_then_a, 1, 1, rng) == 0  # B loses round 1
    assert play(_season(), always_b_then_a, 1, 2, rng) == 1  # B loses, A loses round 2


def test_paired_vs():
    res = pd.DataFrame(
        {
            "strategy": ["x", "x", "base", "base"],
            "lives": [1, 1, 1, 1],
            "season": ["s", "s", "s", "s"],
            "start": [1, 2, 1, 2],
            "survived": [5, 3, 2, 2],
        }
    )
    out = bt.paired_vs(res, "base").set_index("strategy")
    assert out.loc["x", "vs_base"] == 2.0
    assert out.loc["base", "vs_base"] == 0.0
