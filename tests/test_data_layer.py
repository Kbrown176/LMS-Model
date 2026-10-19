"""Offline tests for team mapping, source parsing and the SQLite store."""

import json
from pathlib import Path

import pytest

from lms import store
from lms.sources import footballdata, fpl, odds
from lms.teams import TEAMS, UnknownTeamError, team_id

FIXTURES = Path(__file__).parent / "fixtures"


def _json(name):
    return json.loads((FIXTURES / name).read_text())


# --- teams -------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Man United", "man_united"),  # football-data.co.uk
        ("Man Utd", "man_united"),  # FPL
        ("Manchester United FC", "man_united"),  # football-data.org style
        ("Spurs", "tottenham"),
        ("Tottenham Hotspur", "tottenham"),  # The Odds API
        ("Brighton and Hove Albion", "brighton"),
        ("Brighton & Hove Albion FC", "brighton"),
        ("AFC Bournemouth", "bournemouth"),
        ("Nott'm Forest", "nottm_forest"),
        ("  Hull City  ", "hull"),
        ("Peterboro", "peterborough"),
    ],
)
def test_team_aliases(name, expected):
    assert team_id(name) == expected


def test_unknown_team_raises_with_source():
    with pytest.raises(UnknownTeamError, match="Gotham Rovers.*FPL"):
        team_id("Gotham Rovers", "FPL")


def test_short_codes_unique():
    shorts = [t.short for t in TEAMS.values()]
    assert len(shorts) == len(set(shorts))


# --- football-data.co.uk -----------------------------------------------------


def test_season_codes():
    assert footballdata.season_codes("2627", 2) == ["2425", "2526", "2627"]
    assert footballdata.season_codes("0001", 0) == ["0001"]


def test_parse_results_drops_blank_rows_and_maps_teams():
    df = footballdata.parse_results(FIXTURES / "E1_sample.csv", "2526", "E1")
    assert len(df) == 5
    first = df.iloc[0]
    assert (first.home_id, first.away_id) == ("birmingham", "ipswich")
    assert (first.home_goals, first.away_goals) == (1, 1)
    assert first.date == "2025-08-08"
    assert df[["odds_home", "odds_draw", "odds_away"]].notna().all().all()


# --- FPL -----------------------------------------------------------------------


def test_parse_fpl_fixtures():
    df = fpl.parse_fixtures(_json("fpl_bootstrap.json"), _json("fpl_fixtures.json"), "2627")
    assert set(df.home_id) | set(df.away_id) <= TEAMS.keys()
    assert df.loc[df.finished, "home_score"].notna().all()
    assert df["gw"].isna().sum() == 1  # unscheduled fixture kept, gw unknown


def test_parse_fpl_gameweeks():
    df = fpl.parse_gameweeks(_json("fpl_bootstrap.json"), "2627")
    assert df["is_next"].sum() <= 1
    assert list(df.columns) == [
        "season",
        "gw",
        "deadline_utc",
        "finished",
        "is_current",
        "is_next",
    ]


# --- The Odds API --------------------------------------------------------------


def test_parse_odds():
    df = odds.parse(_json("odds.json"), "2026-10-19T00:00:00+00:00")
    assert df["event_id"].nunique() == 2
    assert (df[["odds_home", "odds_draw", "odds_away"]] > 1).all().all()
    assert set(df.home_id) == {"tottenham", "ipswich"}


# --- store ---------------------------------------------------------------------


def test_upsert_is_idempotent(tmp_path):
    results = footballdata.parse_results(FIXTURES / "E1_sample.csv", "2526", "E1")
    fixtures = fpl.parse_fixtures(_json("fpl_bootstrap.json"), _json("fpl_fixtures.json"), "2627")
    db = tmp_path / "test.db"
    for _ in range(2):
        with store.connect(db) as conn:
            store.upsert(conn, "results", results)
            store.upsert(conn, "fixtures", fixtures)
    with store.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM results").fetchone()[0] == len(results)
        assert conn.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0] == len(fixtures)
        assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == len(TEAMS)


def test_upsert_updates_changed_rows(tmp_path):
    fixtures = fpl.parse_fixtures(_json("fpl_bootstrap.json"), _json("fpl_fixtures.json"), "2627")
    db = tmp_path / "test.db"
    with store.connect(db) as conn:
        store.upsert(conn, "fixtures", fixtures)
    played = fixtures.copy()
    played.loc[:, ["finished", "home_score", "away_score"]] = [True, 2, 1]
    with store.connect(db) as conn:
        store.upsert(conn, "fixtures", played)
        unfinished = conn.execute("SELECT COUNT(*) FROM fixtures WHERE finished = 0").fetchone()
    assert unfinished[0] == 0


def test_fpl_provisional_result_counts_as_finished():
    fixtures = _json("fpl_fixtures.json")
    fixtures[2] = {**fixtures[2], "finished": False, "finished_provisional": True}
    df = fpl.parse_fixtures(_json("fpl_bootstrap.json"), fixtures, "2627")
    assert df.loc[2, "finished"]
