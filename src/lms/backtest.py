"""Replay past seasons as Last Man Standing competitions.

Past seasons have no official gameweek numbers in the data, so rounds are rebuilt from match
dates: matches join the current round until a team would appear twice. Rounds with fewer than
`MIN_ROUND_SIZE` matches are rearranged fixtures; they still train the model but aren't
gameweeks you can pick in (like a blank/double gameweek in FPL).

For each round the model is refitted on earlier matches only. The closing odds of that round
and the next stand in for the odds you'd see before the deadline (the same setup the blend was
tuned on). Every strategy then plays a competition starting at each round, picking until it
runs out of lives or reaches `MAX_PICKS` (20 teams, no reuse).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lms.forecast import BlendConfig, forecast, team_view
from lms.market import devig
from lms.model import DixonColes, FitConfig, promoted_teams
from lms.strategy import build_board, optimise

MIN_ROUND_SIZE = 8
MAX_PICKS = 20
MIN_REMAINING = 12  # a competition needs room to run
LOOKAHEAD_ROUNDS = 2  # rounds whose odds are known at decision time


def assign_rounds(season_matches: pd.DataFrame) -> pd.Series:
    """Greedy round numbers (chronological); small fragment rounds get NaN."""
    df = season_matches.sort_values("date")
    rounds, current, k = [], set(), 1
    for h, a in zip(df.home_id, df.away_id, strict=True):
        if h in current or a in current:
            k, current = k + 1, set()
        current |= {h, a}
        rounds.append(k)
    rd = pd.Series(rounds, index=df.index)
    sizes = rd.map(rd.value_counts())
    rd = rd.where(sizes >= MIN_ROUND_SIZE)
    # renumber the real rounds 1..N
    order = {r: i + 1 for i, r in enumerate(sorted(rd.dropna().unique()))}
    return rd.map(order).reindex(season_matches.index)


@dataclass
class SeasonData:
    season: str
    rounds: list[int]
    views: dict[int, pd.DataFrame]  # round -> team view (gw = round) for rounds ahead
    won: dict[tuple[int, str], bool]  # (round, team) -> did they win
    market_p: dict[tuple[int, str], float]  # (round, team) -> closing-odds win prob


def prepare_season(
    matches: pd.DataFrame,
    season: str,
    model_cfg: FitConfig,
    blend_cfg: BlendConfig,
    max_horizon: int,
) -> SeasonData:
    m = matches.assign(date=pd.to_datetime(matches["date"]))
    pl = m[(m.season == season) & (m.division == "E0")].copy()
    pl["round"] = assign_rounds(pl)
    pl = pl.dropna(subset=["round"]).astype({"round": int})
    pl[["m_home", "m_draw", "m_away"]] = np.column_stack(
        devig(pl.odds_home, pl.odds_draw, pl.odds_away)
    )
    pl["fixture_id"] = np.arange(len(pl))
    pl["gw"] = pl["round"]
    pl["kickoff_utc"] = pl.date.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    promoted = promoted_teams(m, season)

    won, market_p = {}, {}
    for r in pl.itertuples():
        won[(r.round, r.home_id)] = r.home_goals > r.away_goals
        won[(r.round, r.away_id)] = r.away_goals > r.home_goals
        market_p[(r.round, r.home_id)] = r.m_home
        market_p[(r.round, r.away_id)] = r.m_away

    rounds = sorted(pl["round"].unique())
    views, model = {}, None
    for rd in rounds:
        as_of = pl.loc[pl["round"] == rd, "date"].min()
        model = DixonColes(model_cfg).fit(m, as_of=as_of, warm_start=model)
        ahead = pl[(pl["round"] >= rd) & (pl["round"] < rd + max_horizon)]
        seen = pl[(pl["round"] >= rd) & (pl["round"] < rd + LOOKAHEAD_ROUNDS)]
        market = seen.rename(columns={"m_home": "p_home", "m_draw": "p_draw", "m_away": "p_away"})[
            ["home_id", "away_id", "p_home", "p_draw", "p_away"]
        ].assign(n_books=1)
        fixtures = ahead[["fixture_id", "gw", "kickoff_utc", "home_id", "away_id"]]
        views[rd] = team_view(forecast(model, fixtures, market, blend_cfg, promoted))
    return SeasonData(season, rounds, views, won, market_p)


# ------------------------------------------------------------------ strategies

# A strategy gets (season data, round, used teams, lives left, rng) and returns a team id.
Strategy = Callable[[SeasonData, int, set, int, np.random.Generator], str | None]


def optimiser(horizon: int) -> Strategy:
    def pick(data, rd, used, lives, rng):
        board = build_board(data.views[rd], rd, horizon, used=used)
        if not board.teams or rd not in board.gws or board.p[:, 0].max() <= 0:
            return None
        p = board.p[:, : min(len(board.gws), len(board.teams))]
        return board.teams[optimise(p, lives)[0]]

    return pick


def biggest_favourite(data, rd, used, lives, rng):
    """Unused team with the highest closing-odds win probability this round."""
    options = {t: p for (r, t), p in data.market_p.items() if r == rd and t not in used}
    return max(options, key=options.get) if options else None


def random_pick(data, rd, used, lives, rng):
    options = sorted(t for (r, t) in data.won if r == rd and t not in used)
    return rng.choice(options) if options else None


def play(data: SeasonData, strategy: Strategy, start: int, lives: int, rng) -> int:
    """Rounds survived (picks that didn't cost the last life), capped at MAX_PICKS."""
    used: set[str] = set()
    survived = 0
    for rd in [r for r in data.rounds if r >= start][:MAX_PICKS]:
        team = strategy(data, rd, used, lives, rng)
        if team is None:
            break
        used.add(team)
        if not data.won[(rd, team)]:
            lives -= 1
            if lives == 0:
                break
        survived += 1
    return survived


def run(
    seasons: list[SeasonData],
    strategies: dict[str, Strategy],
    lives_options=(1, 2),
    random_repeats: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per (strategy, lives, season, start round) with rounds survived."""
    rows = []
    for data in seasons:
        # Start a competition at every round with at least MIN_REMAINING rounds left.
        starts = data.rounds[: max(1, len(data.rounds) - MIN_REMAINING + 1)]
        for name, strat in strategies.items():
            repeats = random_repeats if strat is random_pick else 1
            for lives in lives_options:
                for start in starts:
                    rng = np.random.default_rng(seed + start)
                    for _ in range(repeats):
                        rows.append(
                            {
                                "strategy": name,
                                "lives": lives,
                                "season": data.season,
                                "start": start,
                                "survived": play(data, strat, start, lives, rng),
                            }
                        )
    return pd.DataFrame(rows)


def paired_vs(results: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Mean extra rounds survived vs `baseline` in the same competitions (season, start)."""
    keys = ["lives", "season", "start"]
    base = results[results.strategy == baseline].groupby(keys).survived.mean().rename("base")
    per = results.groupby(["strategy", *keys]).survived.mean().reset_index().join(base, on=keys)
    per["diff"] = per.survived - per.base
    out = per.groupby(["lives", "strategy"])["diff"].agg(["mean", "std", "size"]).reset_index()
    # Competitions starting a round apart share most of their picks, so this standard error
    # is optimistic; treat it as a rough guide.
    out["se"] = out["std"] / np.sqrt(out["size"])
    return out.drop(columns=["std", "size"]).rename(columns={"mean": f"vs_{baseline}"})


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["lives", "strategy"])
        .agg(
            competitions=("survived", "size"),
            mean_rounds=("survived", "mean"),
            median_rounds=("survived", "median"),
            p_5plus=("survived", lambda s: (s >= 5).mean()),
            p_10plus=("survived", lambda s: (s >= 10).mean()),
        )
        .reset_index()
        .sort_values(["lives", "mean_rounds"], ascending=[True, False], ignore_index=True)
    )


def _season_label(s: str) -> str:
    return f"20{s[:2]}/{s[2:]}"


def report_markdown(results: pd.DataFrame, horizon: int) -> str:
    from lms.calibration import _md_table

    summary = summarise(results).merge(
        paired_vs(results, "biggest favourite"), on=["lives", "strategy"]
    )
    summary = summary.rename(
        columns={
            "mean_rounds": "mean",
            "median_rounds": "median",
            "p_5plus": "5+ rounds",
            "p_10plus": "10+ rounds",
            "vs_biggest favourite": "vs favourite",
        }
    )
    by_season = (
        results[results.strategy.isin(["biggest favourite", f"optimiser H={horizon}"])]
        .groupby(["lives", "strategy", "season"])
        .survived.mean()
        .unstack()
        .rename(columns=_season_label)
        .reset_index()
    )
    seasons = ", ".join(_season_label(s) for s in sorted(results.season.unique()))
    parts = [
        "# Backtest",
        "",
        f"Last Man Standing replayed over {seasons}. A competition starts at every gameweek "
        f"with at least {MIN_REMAINING} left; each strategy picks until it runs out of lives "
        f"(a draw or loss costs one) or makes {MAX_PICKS} picks. The model is refitted before "
        "every gameweek on earlier results only, and the closing odds of that gameweek and the "
        "next stand in for the odds you'd see at the deadline.",
        "",
        "Strategies:",
        "",
        "- **optimiser H=n**: this project; best distinct-team plan over the next n gameweeks, "
        "play its first pick, re-plan next week",
        "- **biggest favourite**: the unused team with the shortest odds this week",
        "- **random**: any unused team (averaged over 20 runs)",
        "",
        "`mean`/`median` = gameweeks survived. `vs favourite` = extra gameweeks survived in the "
        "same competitions, with a rough standard error (`se`; overlapping competitions make it "
        "optimistic).",
        "",
        _md_table(summary),
        "",
        f"## By season (mean gameweeks survived, H={horizon})",
        "",
        _md_table(by_season, ".2f"),
        "",
        "## Reading it",
        "",
        "- Planning ahead is where the gain is: H=1 (greedy on the same probabilities) is no "
        "better than taking the biggest favourite.",
        "- Too long a horizon hurts. The optimiser starts saving strong teams for weeks so far "
        "ahead that the forecasts are poor.",
        "- LMS is high-variance: even the best strategy is usually out within a handful of "
        "gameweeks with one life, and season-to-season swings are large.",
    ]
    return "\n".join(parts) + "\n"
