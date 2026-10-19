"""Entry config and PICKS.md rendering."""

import pytest

from lms.config import load_settings
from lms.picks import PickContext, load_entry, parse_used
from lms.report import render
from lms.strategy import build_board, recommend
from lms.teams import UnknownTeamError
from tests.test_strategy import _team_view


def test_load_entry_accepts_any_spelling(tmp_path):
    path = tmp_path / "entry.toml"
    path.write_text('lives = 1\nused = ["Man Utd", "Spurs"]\n')
    entry = load_entry(load_settings(), path)
    assert entry.lives == 1
    assert entry.used == {"man_united", "tottenham"}


def test_missing_entry_falls_back_to_settings(tmp_path):
    entry = load_entry(load_settings(), tmp_path / "nope.toml")
    assert entry.used == set() and entry.lives == load_settings().lives


def test_parse_used_rejects_unknown():
    assert parse_used(" arsenal , Chelsea ") == {"arsenal", "chelsea"}
    with pytest.raises(UnknownTeamError):
        parse_used("Arsenal, Gotham Rovers")


def test_render_picks_markdown():
    tv = _team_view()
    fc = tv[tv.is_home].rename(columns={"team_id": "home_id", "opponent_id": "away_id"})
    fc = fc.assign(p_home=fc.p_win, p_draw=0.2, p_away=0.8 - fc.p_win)
    ctx = PickContext(
        rec=recommend(build_board(tv, 8, 2, used={"everton"}), lives=1),
        forecast=fc,
        gw=8,
        deadline_utc="2026-10-23T17:30:00Z",
        used={"everton"},
        lives=1,
    )
    md = render(ctx, "2627", {"odds_requests_remaining": "490", "odds_matches": "4"})
    assert md.startswith("# LMS picks · GW8 · 2026/27")
    assert "deadline Fri 23 Oct 17:30 UTC" in md
    assert "## Pick: Arsenal" in md
    assert "~~Everton~~" in md  # used teams struck through in the grid
    assert "1 life" in md and "490" in md
