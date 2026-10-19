"""Canonical team IDs and the aliases each data source uses for them.

Every name coming in from FPL, football-data.co.uk or The Odds API goes through
`team_id()`. An unrecognised name raises instead of silently failing to match,
so a newly promoted club shows up as a clear error rather than missing odds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    id: str
    name: str
    short: str


class UnknownTeamError(ValueError):
    pass


# (id, display name, 3-letter code, extra aliases)
# The display name, id and code are aliases automatically; list only other spellings.
_TEAMS: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("arsenal", "Arsenal", "ARS", ()),
    ("aston_villa", "Aston Villa", "AVL", ()),
    ("barnsley", "Barnsley", "BAR", ()),
    ("birmingham", "Birmingham City", "BIR", ("Birmingham",)),
    ("blackburn", "Blackburn Rovers", "BLA", ("Blackburn",)),
    ("blackpool", "Blackpool", "BPL", ()),
    ("bolton", "Bolton Wanderers", "BOL", ("Bolton",)),
    ("bournemouth", "Bournemouth", "BOU", ("AFC Bournemouth",)),
    ("brentford", "Brentford", "BRE", ()),
    ("brighton", "Brighton", "BHA", ("Brighton and Hove Albion", "Brighton & Hove Albion")),
    ("bristol_city", "Bristol City", "BRC", ()),
    ("burnley", "Burnley", "BUR", ()),
    ("cardiff", "Cardiff City", "CAR", ("Cardiff",)),
    ("charlton", "Charlton Athletic", "CHA", ("Charlton",)),
    ("chelsea", "Chelsea", "CHE", ()),
    ("coventry", "Coventry City", "COV", ("Coventry",)),
    ("crystal_palace", "Crystal Palace", "CRY", ()),
    ("derby", "Derby County", "DER", ("Derby",)),
    ("everton", "Everton", "EVE", ()),
    ("fulham", "Fulham", "FUL", ()),
    ("huddersfield", "Huddersfield Town", "HUD", ("Huddersfield",)),
    ("hull", "Hull City", "HUL", ("Hull",)),
    ("ipswich", "Ipswich Town", "IPS", ("Ipswich",)),
    ("leeds", "Leeds United", "LEE", ("Leeds", "Leeds Utd")),
    ("leicester", "Leicester City", "LEI", ("Leicester",)),
    ("lincoln", "Lincoln City", "LIN", ("Lincoln",)),
    ("liverpool", "Liverpool", "LIV", ()),
    ("luton", "Luton Town", "LUT", ("Luton",)),
    ("man_city", "Manchester City", "MCI", ("Man City",)),
    ("man_united", "Manchester United", "MUN", ("Man United", "Man Utd", "Manchester Utd")),
    ("middlesbrough", "Middlesbrough", "MID", ()),
    ("millwall", "Millwall", "MIL", ()),
    ("newcastle", "Newcastle United", "NEW", ("Newcastle", "Newcastle Utd")),
    ("norwich", "Norwich City", "NOR", ("Norwich",)),
    ("nottm_forest", "Nottingham Forest", "NFO", ("Nott'm Forest", "Nottm Forest")),
    ("oxford", "Oxford United", "OXF", ("Oxford",)),
    ("peterborough", "Peterborough United", "PET", ("Peterboro", "Peterborough")),
    ("plymouth", "Plymouth Argyle", "PLY", ("Plymouth",)),
    ("portsmouth", "Portsmouth", "POR", ()),
    ("preston", "Preston North End", "PNE", ("Preston",)),
    ("qpr", "Queens Park Rangers", "QPR", ()),
    ("reading", "Reading", "REA", ()),
    ("rotherham", "Rotherham United", "ROT", ("Rotherham",)),
    ("sheffield_united", "Sheffield United", "SHU", ("Sheffield Utd",)),
    ("sheffield_weds", "Sheffield Wednesday", "SHW", ("Sheffield Weds",)),
    ("southampton", "Southampton", "SOU", ()),
    ("stoke", "Stoke City", "STK", ("Stoke",)),
    ("sunderland", "Sunderland", "SUN", ()),
    ("swansea", "Swansea City", "SWA", ("Swansea",)),
    ("tottenham", "Tottenham Hotspur", "TOT", ("Tottenham", "Spurs")),
    ("watford", "Watford", "WAT", ()),
    ("west_brom", "West Bromwich Albion", "WBA", ("West Brom",)),
    ("west_ham", "West Ham United", "WHU", ("West Ham",)),
    ("wigan", "Wigan Athletic", "WIG", ("Wigan",)),
    ("wolves", "Wolverhampton Wanderers", "WOL", ("Wolves", "Wolverhampton")),
    ("wrexham", "Wrexham", "WRE", ()),
]

TEAMS: dict[str, Team] = {tid: Team(tid, name, short) for tid, name, short, _ in _TEAMS}


def _norm(name: str) -> str:
    s = name.lower().replace("&", "and")
    s = re.sub(r"\b(a?fc)\b", " ", s)  # "AFC Bournemouth", "Chelsea FC"
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _build_alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for tid, name, short, extra in _TEAMS:
        for alias in (tid, name, short, *extra):
            key = _norm(alias)
            if aliases.get(key, tid) != tid:
                raise RuntimeError(f"Alias {alias!r} maps to both {aliases[key]} and {tid}")
            aliases[key] = tid
    return aliases


_ALIASES = _build_alias_map()


def team_id(name: str, source: str = "unknown") -> str:
    """Return the canonical id for any known spelling of a club name."""
    tid = _ALIASES.get(_norm(name))
    if tid is None:
        raise UnknownTeamError(
            f"Unknown team {name!r} from {source}. Add it (or an alias) to src/lms/teams.py."
        )
    return tid


def team_name(tid: str) -> str:
    return TEAMS[tid].name
