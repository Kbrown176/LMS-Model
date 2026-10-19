"""Command-line entry point: `lms refresh | pick | backtest | report`."""

from __future__ import annotations

import typer
from rich.console import Console

from lms import __version__
from lms.config import DB_PATH, load_settings

app = typer.Typer(help="Premier League Last Man Standing pick optimiser.", no_args_is_help=True)
console = Console()


def _not_ready(phase: str) -> None:
    console.print(f"[yellow]Not implemented yet ({phase}).[/yellow]")
    raise typer.Exit(code=1)


@app.command()
def refresh(
    force: bool = typer.Option(False, help="Re-download every results file, ignoring the cache."),
    odds: bool = typer.Option(True, help="Fetch bookmaker odds (uses 1 Odds API request)."),
) -> None:
    """Fetch fixtures, results and odds into the local database."""
    from lms import refresh as refresh_mod

    with console.status("Refreshing data..."):
        summary = refresh_mod.run(load_settings(), force=force, with_odds=odds)
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
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / "calibration.md"
    path.write_text(md)
    console.print(md)
    console.print(f"[green]✓[/green] Wrote {path.relative_to(REPORTS_DIR.parent)}")


@app.command()
def pick(
    gw: int | None = typer.Option(None, help="Gameweek to pick for (default: next)."),
    used: str = typer.Option("", help="Comma-separated teams already used."),
    lives: int | None = typer.Option(None, help="Lives remaining (default from settings)."),
) -> None:
    """Recommend this gameweek's pick and the best path ahead."""
    _not_ready("Phase 5")


@app.command()
def backtest() -> None:
    """Replay past seasons to evaluate the strategy."""
    _not_ready("Phase 6")


@app.command()
def report() -> None:
    """Write PICKS.md for the upcoming gameweek."""
    _not_ready("Phase 7")


@app.command()
def info() -> None:
    """Show version and active settings."""
    s = load_settings()
    console.print(f"lms-model {__version__}")
    console.print(s)


if __name__ == "__main__":
    app()
