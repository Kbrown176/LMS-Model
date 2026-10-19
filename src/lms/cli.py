"""Command-line entry point: `lms refresh | pick | backtest | report`."""

from __future__ import annotations

import typer
from rich.console import Console

from lms import __version__
from lms.config import load_settings

app = typer.Typer(help="Premier League Last Man Standing pick optimiser.", no_args_is_help=True)
console = Console()


def _not_ready(phase: str) -> None:
    console.print(f"[yellow]Not implemented yet ({phase}).[/yellow]")
    raise typer.Exit(code=1)


@app.command()
def refresh() -> None:
    """Fetch fixtures, results and odds into the local database."""
    _not_ready("Phase 2")


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
