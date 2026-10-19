"""Walk-forward evaluation of the model against bookmaker closing odds.

For every week of the test seasons the model is refitted on matches before that week only,
then predicts that week's Premier League games. Nothing from the future leaks in.
Closing odds are the sharpest public forecast, so they are the benchmark to beat (or match).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lms.market import devig
from lms.model import DixonColes, FitConfig, promoted_teams

OUTCOMES = ("H", "D", "A")


def walk_forward(matches: pd.DataFrame, test_seasons: list[str], config: FitConfig) -> pd.DataFrame:
    """Out-of-sample model predictions for every PL match in `test_seasons`."""
    matches = matches.assign(date=pd.to_datetime(matches["date"]))
    test = matches[(matches.division == "E0") & matches.season.isin(test_seasons)]
    weeks = test.date.dt.to_period("W-MON").dt.start_time  # Tue-Mon blocks span a gameweek

    # Which clubs are newly promoted is known before a season starts, so this isn't leakage.
    promoted = {s: promoted_teams(matches, s) for s in test_seasons}
    out = []
    model = None
    for week_start, games in test.groupby(weeks):
        model = DixonColes(config).fit(matches, as_of=week_start, warm_start=model)
        season = games.season.iloc[0]
        out.append(model.predict_frame(games, promoted=promoted[season]))
    preds = pd.concat(out, ignore_index=True)

    mh, md, ma = devig(preds.odds_home, preds.odds_draw, preds.odds_away)
    preds = preds.assign(m_home=mh, m_draw=md, m_away=ma)
    preds["result"] = np.select(
        [preds.home_goals > preds.away_goals, preds.home_goals == preds.away_goals],
        ["H", "D"],
        "A",
    )
    return preds


def _probs(preds: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = ["p_home", "p_draw", "p_away"] if prefix == "model" else ["m_home", "m_draw", "m_away"]
    return preds[cols].to_numpy()


def _onehot(preds: pd.DataFrame) -> np.ndarray:
    return np.stack([(preds.result == o).to_numpy(float) for o in OUTCOMES], axis=1)


@dataclass
class Scores:
    log_loss: float
    brier: float
    n: int


def score(preds: pd.DataFrame, source: str) -> Scores:
    p = np.clip(_probs(preds, source), 1e-12, 1)
    y = _onehot(preds)
    return Scores(
        log_loss=float(-np.mean(np.log((p * y).sum(axis=1)))),
        brier=float(np.mean(((p - y) ** 2).sum(axis=1))),
        n=len(preds),
    )


def team_view(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, side): the LMS question is 'will this team win?'."""
    home = pd.DataFrame(
        {
            "season": preds.season,
            "team_id": preds.home_id,
            "is_home": True,
            "p_model": preds.p_home,
            "p_market": preds.m_home,
            "won": preds.result == "H",
        }
    )
    away = pd.DataFrame(
        {
            "season": preds.season,
            "team_id": preds.away_id,
            "is_home": False,
            "p_model": preds.p_away,
            "p_market": preds.m_away,
            "won": preds.result == "A",
        }
    )
    return pd.concat([home, away], ignore_index=True)


def reliability(preds: pd.DataFrame, bins=(0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)) -> pd.DataFrame:
    """Predicted vs actual win rate, bucketed by the model's win probability."""
    tv = team_view(preds)
    tv["bucket"] = pd.cut(tv.p_model, bins=list(bins), include_lowest=True)
    return (
        tv.groupby("bucket", observed=True)
        .agg(
            n=("won", "size"),
            model=("p_model", "mean"),
            market=("p_market", "mean"),
            actual=("won", "mean"),
        )
        .reset_index()
    )


def promoted_bias(preds: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Win probability vs actual for clubs in their first PL season after promotion."""
    tv = team_view(preds)
    promoted = {s: promoted_teams(matches, s) for s in tv.season.unique()}
    tv["promoted"] = [t in promoted[s] for s, t in zip(tv.season, tv.team_id, strict=True)]
    return (
        tv.groupby("promoted")
        .agg(
            n=("won", "size"),
            model=("p_model", "mean"),
            market=("p_market", "mean"),
            actual=("won", "mean"),
        )
        .reset_index()
    )


def tune(
    matches: pd.DataFrame,
    test_seasons: list[str],
    half_lives=(90, 120, 180, 270, 365),
    ridges=(0.3, 1.0, 3.0),
) -> pd.DataFrame:
    """Walk-forward log-loss for each (half-life, ridge) pair, best first."""
    rows = []
    for hl in half_lives:
        for rg in ridges:
            s = score(walk_forward(matches, test_seasons, FitConfig(hl, rg)), "model")
            rows.append(
                {"half_life_days": hl, "ridge": rg, "log_loss": s.log_loss, "brier": s.brier}
            )
    return pd.DataFrame(rows).sort_values("log_loss", ignore_index=True)


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report_markdown(
    matches: pd.DataFrame,
    test_seasons: list[str],
    config: FitConfig,
    tuning: pd.DataFrame | None = None,
) -> str:
    preds = walk_forward(matches, test_seasons, config)
    overall = pd.DataFrame(
        [
            {"forecast": name, "log_loss": s.log_loss, "brier": s.brier, "matches": s.n}
            for name, s in (("model", score(preds, "model")), ("market", score(preds, "market")))
        ]
    )
    by_season = pd.DataFrame(
        [
            {
                "season": f"20{s[:2]}/{s[2:]}",
                "matches": len(g),
                "model": score(g, "model").log_loss,
                "market": score(g, "market").log_loss,
            }
            for s, g in preds.groupby("season")
        ]
    )
    rel = reliability(preds)
    rel["bucket"] = rel["bucket"].astype(str)
    final = DixonColes(config).fit(matches)

    parts = [
        "# Model calibration",
        "",
        f"Walk-forward test on {len(preds)} Premier League matches "
        f"({', '.join(f'20{s[:2]}/{s[2:]}' for s in test_seasons)}). The model is refitted every "
        "week on earlier matches only, then compared with de-vigged average closing odds.",
        "",
        f"Settings: half-life {config.half_life_days:g} days, ridge {config.ridge:g}. "
        "Lower log-loss and Brier are better.",
        "",
        "## Overall",
        "",
        _md_table(overall, ".4f"),
        "",
        "## By season (log-loss)",
        "",
        _md_table(by_season, ".4f"),
        "",
        "## Win-probability calibration",
        "",
        "Every team in every match, bucketed by the model's win probability. "
        "`actual` is how often those teams won.",
        "",
        _md_table(rel),
        "",
        "## Promoted clubs (first PL season)",
        "",
        _md_table(promoted_bias(preds, matches)),
        "",
        f"Learned promotion adjustment: attack -{final.promoted_attack:.3f}, "
        f"defence -{final.promoted_defence:.3f} (log goals). Dixon-Coles rho {final.rho:.3f}.",
    ]
    if tuning is not None:
        parts += ["", "## Tuning grid", "", _md_table(tuning.head(10), ".4f")]
    return "\n".join(parts) + "\n"
