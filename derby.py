import pandas as pd

DERBY_PATH = "data/derbies.csv"

# Load once globally
_DERBY_TABLE = pd.read_csv(DERBY_PATH)

def derby_intensity(team_a: str, team_b: str) -> float:
    """
    Returns the rivalry intensity between two teams (0 to 1.0).
    0 = no rivalry
    1.0 = very intense derby (avoid in LMS)
    """

    # Normalise names for safety
    t1 = team_a.strip()
    t2 = team_b.strip()

    # Check both directions
    row = _DERBY_TABLE[
        ((_DERBY_TABLE["team_a"] == t1) & (_DERBY_TABLE["team_b"] == t2)) |
        ((_DERBY_TABLE["team_a"] == t2) & (_DERBY_TABLE["team_b"] == t1))
    ]

    if row.empty:
        return 0.0

    return float(row["intensity"].iloc[0])
