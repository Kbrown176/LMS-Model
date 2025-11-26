# team_strength.py

import pandas as pd

TABLE_PATH = "data/Table.csv"

# Map short / messy names in the league table to your canonical names
TEAM_CLEAN = {
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Newcastle ": "Newcastle United",
    "West Ham": "West Ham",
    "West Ham ": "West Ham",
    "Leeds": "Leeds United",
    "Leeds ": "Leeds United",
    "Nott'm Forest": "Nottingham Forest",
    # you can add more if your odds/fixtures use other variations
}

def clean_team(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = name.strip()
    return TEAM_CLEAN.get(name, name)


# ---- Load and clean the table once ---- #

_table = pd.read_csv(TABLE_PATH)

# First column = position, second column = team name in your sample
pos_col = _table.columns[0]      # e.g. 'Unnamed: 0'
team_col = _table.columns[1]     # e.g. 'Unnamed: 1'

# Clean team names to match your main model
_table[team_col] = _table[team_col].astype(str).apply(clean_team)

# Build mapping: team -> position (1 = best, 20 = worst)
_position_by_team = dict(zip(_table[team_col], _table[pos_col]))


def team_position(team: str) -> int:
    """
    Returns league position (1-20). If unknown, default to 20 (treat as weak).
    """
    if not isinstance(team, str):
        return 20
    team = clean_team(team)
    return int(_position_by_team.get(team, 20))


def is_weak_team(team: str, bottom_n: int = 5) -> bool:
    """
    True if team is in the bottom `bottom_n` (e.g. bottom 5 of table).
    """
    return team_position(team) >= (21 - bottom_n)
