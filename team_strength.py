# team_strength.py

import pandas as pd

# Match what football_api.refresh_csvs() is writing:
# if you used "Table.csv" with a capital T there, change this to "data/Table.csv"
TABLE_PATH = "data/table.csv"

TEAM_CLEAN = {
    # Man City
    "Man City": "Manchester City",
    "Manchester City": "Manchester City",

    # Man United
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Manchester United": "Manchester United",

    # Newcastle
    "Newcastle": "Newcastle United",
    "Newcastle ": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Newcastle United": "Newcastle United",

    # West Ham
    "West Ham": "West Ham",
    "West Ham ": "West Ham",
    "West Ham United": "West Ham",
    "West Ham Utd": "West Ham",

    # Leeds
    "Leeds": "Leeds United",
    "Leeds ": "Leeds United",
    "Leeds Utd": "Leeds United",
    "Leeds United": "Leeds United",

    # Forest
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",

    # Spurs
    "Tottenham": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspurs": "Tottenham",
    "Spurs": "Tottenham",

    # You can add more variants here if needed
}


def clean_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    return TEAM_CLEAN.get(name, name)


# ---- Load and clean the table once ---- #

_table = pd.read_csv(TABLE_PATH)

# New API table has explicit columns: 'position' and 'team'
if "position" not in _table.columns or "team" not in _table.columns:
    raise ValueError(
        f"Expected columns 'position' and 'team' in {TABLE_PATH}, "
        f"found: {_table.columns.tolist()}"
    )

_table["team"] = _table["team"].astype(str).apply(clean_team)

# Map: canonical team name -> league position (1 = best, 20 = worst)
_position_by_team = dict(zip(_table["team"], _table["position"]))


def team_position(team: str) -> int:
    """
    Returns league position (1–20). If unknown, default to 20 (treat as weak).
    """
    if not isinstance(team, str):
        return 20
    team = clean_team(team)
    return int(_position_by_team.get(team, 20))


def is_weak_team(team: str, bottom_n: int = 5) -> bool:
    """
    True if team is in the bottom `bottom_n` (e.g. bottom 5 of the table).
    """
    return team_position(team) >= (21 - bottom_n)
