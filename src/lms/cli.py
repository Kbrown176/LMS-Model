"""Command-line entry point: `lms refresh | pick | backtest | report`."""

from __future__ import annotations

import typer
from rich.console import Console

from lms import __version__
from lms.config import DB_PATH, load_settings

app = typer.Typer(help="Premier League Last Man Standing pick optimiser.", no_args_is_help=True)
console = Console()


@app.command()
def refresh(
    force: bool = typer.Option(False, help="Re-download every results file, ignoring the cache."),
    odds: bool = typer.Option(True, help="Fetch bookmaker odds (uses 1 Odds API request)."),
    max_age_hours: float = typer.Option(
        12, help="Refetch the current season's results file if older than this."
    ),
) -> None:
    """Fetch fixtures, results and odds into the local database."""
    from lms import refresh as refresh_mod

    with console.status("Refreshing data..."):
        summary = refresh_mod.run(
            load_settings(), force=force, with_odds=odds, max_age_hours=max_age_hours
        )
    for line in summary.lines:
        console.print(f"[green]✓[/green] {line}")
    for warning in summary.warnings:
        console.print(f"[yellow]![/yellow] {warning}")


@app.command()
def status() -> None:
    """Show what is in the local database."""
    from lms import store

    if not DB_PATH.exists():
        console.print("[yellow]No database yet. Run `lms refresh` first.[/yellow]")
        raise typer.Exit(code=1)
    with store.connect() as conn:
        for table in ("results", "fixtures", "gameweeks", "odds"):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            console.print(f"{table:<10} {n:>6} rows")
        console.print(f"last refresh: {store.get_meta(conn, 'last_refresh_utc') or 'never'}")
        left = store.get_meta(conn, "odds_requests_remaining")
        console.print(f"odds API requests left: {left or '?'}")


def _load_matches():
    from lms import store
    from lms.data import training_matches

    if not DB_PATH.exists():
        console.print("[yellow]No database yet. Run `lms refresh` first.[/yellow]")
        raise typer.Exit(code=1)
    s = load_settings()
    with store.connect() as conn:
        return (
            s,
            training_matches(conn, s.season),
            store.read(
                conn, "SELECT DISTINCT home_id FROM fixtures WHERE season = ?", (s.season,)
            ).home_id.tolist(),
        )


def _fit_config(s):
    from lms.model import FitConfig

    return FitConfig(half_life_days=s.time_decay_half_life_days, ridge=s.ridge)


@app.command()
def ratings() -> None:
    """Fit the model on all results and show current Premier League team ratings."""
    from rich.table import Table

    from lms.model import DixonColes, promoted_teams
    from lms.teams import team_name

    s, matches, pl_teams = _load_matches()
    model = DixonColes(_fit_config(s)).fit(matches)
    promoted = promoted_teams(matches, s.season, pl_teams)
    r = model.ratings(reference=pl_teams)
    r = r[r.team_id.isin(pl_teams)].reset_index(drop=True)

    table = Table(title=f"PL ratings ({model.n_matches} matches, up to {model.as_of.date()})")
    for col in ("#", "Team", "xG for", "xG against", "Rating"):
        table.add_column(col, justify="left" if col == "Team" else "right")
    for i, row in r.iterrows():
        name = team_name(row.team_id) + (" (P)" if row.team_id in promoted else "")
        table.add_row(
            str(i + 1), name, f"{row.xg_for:.2f}", f"{row.xg_against:.2f}", f"{row.rating:+.2f}"
        )
    console.print(table)
    console.print(
        "Rating = expected goal difference per game vs an average PL side on neutral ground. "
        "(P) = promoted; ratings shown before the promotion adjustment."
    )


@app.command()
def calibrate(
    tune: bool = typer.Option(False, help="Also grid-search half-life and ridge (~2 min)."),
    blend: bool = typer.Option(False, help="Also tune the market blend (~30 s)."),
) -> None:
    """Walk-forward test of the model against closing odds; writes reports/calibration.md."""
    from lms import calibration
    from lms.config import REPORTS_DIR
    from lms.sources.footballdata import season_codes

    s, matches, _ = _load_matches()
    seasons = season_codes(s.season, 3)  # last 3 completed seasons + current
    tuning = None
    if tune:
        with console.status("Tuning (walk-forward over a grid)..."):
            tuning = calibration.tune(matches, seasons[:-1])
        console.print(tuning.head(5).to_string(index=False))
    with console.status("Running walk-forward calibration..."):
        md = calibration.report_markdown(matches, seasons, _fit_config(s), tuning)
    if blend:
        from lms.forecast import evaluate

        with console.status("Tuning market blend (walk-forward)..."):
            near, far = evaluate(
                matches,
                seasons[:-1],
                _fit_config(s),
                weights=(0.0, 0.3, 0.5, 0.7, 1.0),
                ridges=(float("inf"), 0.3, 0.1, 0.03, 0.01),
            )
        md += "\n".join(
            [
                "",
                "## Market blend",
                "",
                "Walk-forward on the tuning seasons. Each week the closing odds of the next two "
                "weeks stand in for the odds you'd see; team ratings are nudged towards them "
                "(`adjust_ridge`, inf = no nudge), then blended with this week's odds "
                "(`market_weight`, 0 = model only, 1 = market only).",
                "",
                "### This week (odds available)",
                "",
                calibration._md_table(near.head(8), ".4f"),
                "",
                "### 2-9 weeks ahead (no odds yet)",
                "",
                calibration._md_table(far, ".4f"),
                "",
            ]
        )
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "calibration.md"
    path.write_text(md)
    console.print(md)
    console.print(f"[green]✓[/green] Wrote {path.relative_to(REPORTS_DIR.parent)}")


@app.command()
def probs(
    gw: int | None = typer.Option(None, help="Gameweek to list (default: next)."),
    grid: bool = typer.Option(False, help="Team x gameweek win-probability grid instead."),
    weeks: int | None = typer.Option(None, help="Gameweeks in the grid (default: horizon)."),
    csv: str | None = typer.Option(None, help="Also write every remaining fixture to this CSV."),
) -> None:
    """Win probabilities for upcoming fixtures (model blended with bookmaker odds)."""
    from rich.table import Table

    from lms import store
    from lms.forecast import team_view, upcoming
    from lms.teams import TEAMS, team_name

    s = load_settings()
    if not DB_PATH.exists():
        console.print("[yellow]No database yet. Run `lms refresh` first.[/yellow]")
        raise typer.Exit(code=1)
    with store.connect() as conn:
        fc, _ = upcoming(conn, s)
        next_gw = conn.execute(
            "SELECT gw FROM gameweeks WHERE season = ? AND is_next", (s.season,)
        ).fetchone()
    fc = fc[fc.gw.notna()].astype({"gw": int})
    start = gw or (next_gw[0] if next_gw else int(fc.gw.min()))

    if csv:
        fc.to_csv(csv, index=False)
        console.print(f"[green]✓[/green] Wrote {len(fc)} fixtures to {csv}")

    if grid:
        tv = team_view(fc).drop_duplicates(["gw", "team_id"])  # first fixture in a double GW
        gws = list(range(start, start + (weeks or s.horizon)))
        tv = tv[tv.gw.isin(gws)]
        table = Table(title=f"Win probability (%) GW{gws[0]}-{gws[-1]}  ·  * = includes odds")
        table.add_column("Team")
        for g in gws:
            table.add_column(f"GW{g}", justify="right")
        order = tv.groupby("team_id").p_win.mean().sort_values(ascending=False).index
        for t in order:
            cells = []
            for g in gws:
                row = tv[(tv.team_id == t) & (tv.gw == g)]
                if row.empty:
                    cells.append("[dim]-[/dim]")
                    continue
                r = row.iloc[0]
                opp = TEAMS[r.opponent_id].short + ("" if r.is_home else "(a)")
                mark = "*" if r.source == "blend" else ""
                colour = "green" if r.p_win >= 0.6 else "yellow" if r.p_win >= 0.45 else "dim"
                cells.append(f"[{colour}]{r.p_win * 100:.0f}{mark}[/{colour}] {opp}")
            table.add_row(team_name(t), *cells)
        console.print(table)
        return

    games = fc[fc.gw == start]
    table = Table(title=f"GW{start} probabilities")
    for col in ("Kick-off (UTC)", "Home", "Away", "Home %", "Draw %", "Away %", "Source"):
        table.add_column(col, justify="right" if "%" in col else "left")
    for r in games.itertuples():
        src = f"blend ({int(r.n_books)} books)" if r.source == "blend" else "model"
        table.add_row(
            r.kickoff_utc[:16].replace("T", " "),
            team_name(r.home_id),
            team_name(r.away_id),
            f"{r.p_home * 100:.0f}",
            f"{r.p_draw * 100:.0f}",
            f"{r.p_away * 100:.0f}",
            src,
        )
    console.print(table)


@app.command()
def pick(
    gw: int | None = typer.Option(None, help="Gameweek to pick for (default: next)."),
    used: str | None = typer.Option(
        None, help="Teams already used, comma separated (default: config/entry.toml)."
    ),
    lives: int | None = typer.Option(None, help="Lives left (default: config/entry.toml)."),
    horizon: int | None = typer.Option(None, help="Gameweeks to plan ahead (default: settings)."),
    top: int = typer.Option(8, help="How many of this week's options to list."),
) -> None:
    """Recommend this gameweek's pick and the best path ahead."""
    from rich.table import Table

    from lms import picks, store
    from lms.teams import TEAMS, UnknownTeamError, team_name

    s = load_settings()
    try:
        entry = picks.load_entry(s)
        used_ids = picks.parse_used(used) if used is not None else entry.used
    except UnknownTeamError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from e
    lives = lives or entry.lives
    if not DB_PATH.exists():
        console.print("[yellow]No database yet. Run `lms refresh` first.[/yellow]")
        raise typer.Exit(code=1)

    with console.status("Fitting model and planning..."), store.connect() as conn:
        ctx = picks.plan(conn, s, used_ids, lives, gw, horizon)
    rec, start = ctx.rec, ctx.gw

    def vs(opp, home, derby):
        tag = f"{team_name(opp)} ({'H' if home else 'A'})"
        return tag + (" [red]derby[/red]" if derby > 0 else "")

    weeks = len(rec.path)
    deadline = (
        f"  ·  deadline {ctx.deadline_utc[:16].replace('T', ' ')} UTC" if ctx.deadline_utc else ""
    )
    console.print(
        f"\n[bold]GW{start} pick[/bold]  ·  {lives} li{'fe' if lives == 1 else 'ves'}  ·  "
        f"planning {weeks} weeks{deadline}"
    )
    if used_ids:
        console.print("Used: " + ", ".join(sorted(TEAMS[t].name for t in used_ids)))

    table = Table(title=f"This week's options, ranked by {weeks}-week survival")
    for col in (
        "#",
        "Team",
        "Opponent",
        "Win %",
        f"Alive after GW{rec.path.gw.iloc[-1]}",
        "vs best",
    ):
        table.add_column(col, justify="left" if col in ("Team", "Opponent") else "right")
    for k, r in rec.options.head(top).iterrows():
        table.add_row(
            str(k + 1),
            team_name(r.team_id),
            vs(r.opponent_id, r.is_home, r.derby),
            f"{r.p_win * 100:.0f}",
            f"{r.plan_survival * 100:.1f}%",
            "" if k == 0 else f"{r.vs_best * 100:+.1f}",
        )
    console.print(table)

    path = Table(title=f"Recommended path (starting with {team_name(rec.path.team_id.iloc[0])})")
    for col in ("GW", "Team", "Opponent", "Win %", "Still alive", "Odds?"):
        path.add_column(col, justify="left" if col in ("Team", "Opponent") else "right")
    for r in rec.path.itertuples():
        path.add_row(
            str(r.gw),
            team_name(r.team_id),
            vs(r.opponent_id, r.is_home, r.derby),
            f"{r.p_win * 100:.0f}",
            f"{r.alive * 100:.1f}%",
            "yes" if r.source == "blend" else "",
        )
    console.print(path)
    console.print(
        "[dim]Win % = probability the team wins (a draw costs a life). Later weeks are "
        "re-planned every gameweek as results and odds come in.[/dim]"
    )


@app.command()
def backtest(
    seasons: str = typer.Option("2223,2324,2425,2526", help="Season codes to replay."),
    horizons: str = typer.Option("1,3,5,8,10,12,15", help="Planning horizons to compare."),
) -> None:
    """Replay past seasons as LMS competitions; writes reports/backtest.md (~1 min)."""
    from lms import backtest as bt
    from lms.config import REPORTS_DIR
    from lms.forecast import BlendConfig

    s, matches, _ = _load_matches()
    hs = [int(h) for h in horizons.split(",")]
    data = []
    for season in seasons.split(","):
        with console.status(f"Rebuilding {season} week by week..."):
            data.append(
                bt.prepare_season(
                    matches,
                    season.strip(),
                    _fit_config(s),
                    BlendConfig(s.market_weight, s.adjust_ridge),
                    max(hs),
                )
            )
    strategies = {"random": bt.random_pick, "biggest favourite": bt.biggest_favourite} | {
        f"optimiser H={h}": bt.optimiser(h) for h in hs
    }
    with console.status("Playing competitions..."):
        results = bt.run(data, strategies)
    md = bt.report_markdown(results, s.horizon if s.horizon in hs else hs[-1])
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "backtest.md"
    path.write_text(md)
    console.print(md)
    console.print(f"[green]✓[/green] Wrote {path.relative_to(REPORTS_DIR.parent)}")


@app.command()
def report(
    top: int = typer.Option(10, help="How many of this week's options to list."),
) -> None:
    """Write PICKS.md for your entry (config/entry.toml) for the upcoming gameweek."""
    from lms import picks, store
    from lms import report as report_mod
    from lms.config import PICKS_PATH

    s = load_settings()
    entry = picks.load_entry(s)
    if not DB_PATH.exists():
        console.print("[yellow]No database yet. Run `lms refresh` first.[/yellow]")
        raise typer.Exit(code=1)
    with console.status("Fitting model and planning..."), store.connect() as conn:
        ctx = picks.plan(conn, s, entry.used, entry.lives)
        meta = {
            "last_refresh_utc": store.get_meta(conn, "last_refresh_utc"),
            "odds_requests_remaining": store.get_meta(conn, "odds_requests_remaining"),
            "results_through": conn.execute(
                "SELECT MAX(d) FROM (SELECT MAX(date) AS d FROM results UNION ALL "
                "SELECT MAX(substr(kickoff_utc, 1, 10)) FROM fixtures WHERE finished)"
            ).fetchone()[0],
            "odds_matches": str((ctx.forecast.source == "blend").sum()),
        }
    PICKS_PATH.write_text(report_mod.render(ctx, s.season, meta, top))
    best = ctx.rec.options.iloc[0]
    console.print(
        f"[green]✓[/green] Wrote {PICKS_PATH.name}: GW{ctx.gw} pick "
        f"{report_mod.team_name(best.team_id)} ({best.p_win * 100:.0f}% to win)"
    )


@app.command()
def info() -> None:
    """Show version and active settings."""
    s = load_settings()
    console.print(f"lms-model {__version__}")
    console.print(s)


if __name__ == "__main__":
    app()
