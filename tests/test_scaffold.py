from typer.testing import CliRunner

from lms.cli import app
from lms.config import DERBIES_PATH, load_settings


def test_settings_load():
    s = load_settings()
    assert s.lives >= 1
    assert s.horizon >= 1
    assert "E0" in s.divisions


def test_derbies_file_present():
    header = DERBIES_PATH.read_text().splitlines()[0]
    assert header == "team_a,team_b,intensity"


def test_cli_info_runs():
    result = CliRunner().invoke(app, ["info"])
    assert result.exit_code == 0
    assert "lms-model" in result.stdout
