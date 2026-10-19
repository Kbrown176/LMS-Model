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
