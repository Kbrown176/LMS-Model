"""Render PICKS.md: this week's recommendation, the plan ahead and a win-probability grid."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from lms.forecast import team_view
from lms.picks import PickContext
from lms.teams import TEAMS, team_name


def _vs(opp: str, home: bool, derby: float) -> str:
    return f"{team_name(opp)} ({'H' if home else 'A'})" + (" · derby" if derby > 0 else "")


def _when(iso: str) -> str:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%a %d %b %H:%M UTC")


def _table(header: list[str], rows: list[list[str]], right: set[int] = frozenset()) -> str:
    align = ["---:" if i in right else "---" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _grid(fc: pd.DataFrame, gws: list[int], used: set[str]) -> str:
    tv = team_view(fc).drop_duplicates(["gw", "team_id"])
    tv = tv[tv.gw.isin(gws)]
    order = tv.groupby("team_id").p_win.mean().sort_values(ascending=False).index
    rows = []
    for t in order:
        cells = [f"~~{team_name(t)}~~" if t in used else team_name(t)]
        for g in gws:
            r = tv[(tv.team_id == t) & (tv.gw == g)]
            if r.empty:
                cells.append("–")
                continue
            r = r.iloc[0]
            pct = f"{r.p_win * 100:.0f}"
            pct = f"**{pct}**" if r.p_win >= 0.6 else pct
            opp = (
                TEAMS[r.opponent_id].short.lower() if not r.is_home else TEAMS[r.opponent_id].short
            )
            cells.append(f"{pct} {opp}")
        rows.append(cells)
    return _table(["Team", *[f"GW{g}" for g in gws]], rows, right=set(range(1, len(gws) + 1)))


def render(ctx: PickContext, season: str, meta: dict[str, str | None], top: int = 10) -> str:
    rec, path = ctx.rec, ctx.rec.path
    best = rec.options.iloc[0]
    last_gw = path.gw.iloc[-1]
    season_label = f"20{season[:2]}/{season[2:]}"
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    used = ", ".join(sorted(TEAMS[t].name for t in ctx.used)) or "none yet"
    lives = f"{ctx.lives} li{'fe' if ctx.lives == 1 else 'ves'}"

    options = [
        [
            str(i + 1),
            team_name(r.team_id),
            _vs(r.opponent_id, r.is_home, r.derby),
            f"{r.p_win * 100:.0f}%",
            f"{r.plan_survival * 100:.1f}%",
            "" if i == 0 else f"{r.vs_best * 100:+.1f}",
        ]
        for i, r in rec.options.head(top).iterrows()
    ]
    plan_rows = [
        [
            str(r.gw),
            team_name(r.team_id),
            _vs(r.opponent_id, r.is_home, r.derby),
            f"{r.p_win * 100:.0f}%",
            f"{r.alive * 100:.1f}%",
            "odds + model" if r.source == "blend" else "model",
        ]
        for r in path.itertuples()
    ]

    parts = [
        f"# LMS picks · GW{ctx.gw} · {season_label}",
        "",
        f"> Generated {now}"
        + (f" · **deadline {_when(ctx.deadline_utc)}**" if ctx.deadline_utc else "")
        + f" · {lives} · used: {used}",
        "",
        f"## Pick: {team_name(best.team_id)} vs {_vs(best.opponent_id, best.is_home, best.derby)}",
        "",
        f"**{best.p_win * 100:.0f}%** to win this week. Following the plan below, the chance of "
        f"still being in after GW{last_gw} is **{best.plan_survival * 100:.1f}%** "
        f"with {lives}.",
        "",
        f"## This week's options, ranked by survival to GW{last_gw}",
        "",
        _table(
            ["#", "Team", "Opponent", "Win", f"Alive after GW{last_gw}", "vs best"],
            options,
            right={0, 3, 4, 5},
        ),
        "",
        "Ranked by the best plan that starts with each team, not by this week alone: a team "
        "that's much more valuable in a later week can rank below a weaker pick now.",
        "",
        "## Recommended path",
        "",
        _table(
            ["GW", "Team", "Opponent", "Win", "Still in", "Based on"], plan_rows, right={0, 3, 4}
        ),
        "",
        "Only this week's pick is a commitment. The plan is redone every gameweek.",
        "",
        f"## Win probability grid, GW{ctx.gw}–{last_gw}",
        "",
        "Win % and opponent (CAPS = home, lower case = away). **Bold** = 60%+. "
        "Used teams are struck through.",
        "",
        _grid(ctx.forecast, list(path.gw), ctx.used),
        "",
        "## Data",
        "",
        f"- Results up to {meta.get('results_through') or '?'}; "
        f"refreshed {meta.get('last_refresh_utc') or '?'}",
        f"- Bookmaker odds for {meta.get('odds_matches') or 0} upcoming matches "
        f"(Odds API requests left this month: {meta.get('odds_requests_remaining') or '?'})",
        "- Update `config/entry.toml` with each pick and any lost life.",
        "",
    ]
    return "\n".join(parts)
