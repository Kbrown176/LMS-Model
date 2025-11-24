import pandas as pd
from elo import load_elo_table

# Paths
ODDS_PATH = "data/epl_current_odds.csv"
FIXTURES_PATH = "data/Fixtures.csv"

# Team name cleaning map
TEAM_CLEAN = {
    "Tottenham Hotspurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Spurs": "Tottenham",

    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Manchester Utd": "Manchester United",

    "Man City": "Manchester City",
    "Manchester City": "Manchester City",

    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Wolverhampton W.": "Wolverhampton Wanderers",

    "West Ham Utd": "West Ham United",

    "Leeds": "Leeds United",
    "Leeds Utd": "Leeds United",
    "Leeds United": "Leeds United",
}


def clean_team(name: str) -> str:
    if isinstance(name, str):
        name = name.strip()
        return TEAM_CLEAN.get(name, name)
    return name


def add_win_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert decimal odds into implied probabilities (remove overround).
    Adds: p_home_win, p_draw, p_away_win.
    """

    def calc(row):
        H, D, A = row["home_odds"], row["draw_odds"], row["away_odds"]
        if pd.isna(H) or pd.isna(D) or pd.isna(A) or H == 0 or D == 0 or A == 0:
            return pd.Series({"p_home_win": None, "p_draw": None, "p_away_win": None})

        pH_raw = 1.0 / H
        pD_raw = 1.0 / D
        pA_raw = 1.0 / A

        overround = pH_raw + pD_raw + pA_raw

        return pd.Series(
            {
                "p_home_win": pH_raw / overround,
                "p_draw": pD_raw / overround,
                "p_away_win": pA_raw / overround,
            }
        )

    probs = df.apply(calc, axis=1)
    df = pd.concat([df, probs], axis=1)
    return df


def build_candidate_picks(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a given set of fixtures (e.g. one gameweek),
    create candidate LMS picks:
      - Pick the home team (prob = p_home_win)
      - Pick the away team (prob = p_away_win)
    """
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get("p_home_win")) or pd.isna(r.get("p_away_win")):
            continue

        rows.append(
            {
                "team": r["home_clean"],
                "opponent": r["away_clean"],
                "is_home": True,
                "win_prob": r["p_home_win"],
                "kickoff": r["kickoff_utc"],
                "Round": r.get("Round", None),
            }
        )

        rows.append(
            {
                "team": r["away_clean"],
                "opponent": r["home_clean"],
                "is_home": False,
                "win_prob": r["p_away_win"],
                "kickoff": r["kickoff_utc"],
                "Round": r.get("Round", None),
            }
        )

    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates = candidates.sort_values("win_prob", ascending=False)
    return candidates


def clean_and_merge(odds_path: str, fixtures_path: str) -> pd.DataFrame:
    """
    - Load odds and fixtures
    - Clean team names
    - Align dates
    - Merge odds with fixtures to attach gameweek (Round)
    """
    odds = pd.read_csv(odds_path)
    fixtures = pd.read_csv(fixtures_path)

    # Clean team names
    odds["home_clean"] = odds["home_team"].apply(clean_team)
    odds["away_clean"] = odds["away_team"].apply(clean_team)

    fixtures["home_clean"] = fixtures["HomeTeam"].apply(clean_team)
    fixtures["away_clean"] = fixtures["AwayTeam"].apply(clean_team)

    # Standardise date formats to YYYY-MM-DD strings for both
    fixtures["date_clean"] = (
        pd.to_datetime(fixtures["Date"], format="%d/%m/%Y").dt.date.astype(str)
    )
    odds["date_clean"] = pd.to_datetime(odds["date_utc"]).dt.date.astype(str)

    # We only really need the odds columns + kickoff from the odds file
    odds_small = odds[
        ["date_clean", "home_clean", "away_clean",
         "home_odds", "draw_odds", "away_odds", "kickoff_utc"]
    ]

    # IMPORTANT: base = fixtures (full season), left join odds onto it
    merged = fixtures.merge(
        odds_small,
        on=["date_clean", "home_clean", "away_clean"],
        how="left",
    )


    return merged


# ---------- 10-WEEK STRATEGY HELPERS ---------- #

def plan_future_path(
    merged: pd.DataFrame,
    start_gw: int,
    horizon: int,
    used_teams: list,
    first_pick_row: pd.Series,
) -> list[dict]:
    """
    Build a sequence of picks for the next `horizon` gameweeks,
    forcing `first_pick_row` in `start_gw`, then greedy best available thereafter.

    Returns a list of dicts:
      {"gw", "team", "opponent", "is_home", "win_prob", "kickoff"}
    """
    path = []
    used = set(used_teams)

    # First week = the forced starting pick
    team = first_pick_row["team"]
    opponent = first_pick_row["opponent"]
    is_home = bool(first_pick_row["is_home"])
    win_prob = float(first_pick_row["win_prob"])
    kickoff = first_pick_row["kickoff"]

    path.append(
        {
            "gw": start_gw,
            "team": team,
            "opponent": opponent,
            "is_home": is_home,
            "win_prob": win_prob,
            "kickoff": kickoff,
        }
    )
    used.add(team)

    # Future weeks
    for gw in range(start_gw + 1, start_gw + horizon):
        gw_df = merged[merged["Round"] == gw]
        if gw_df.empty:
            break

        candidates = build_candidate_picks(gw_df)
        if candidates.empty:
            break

        # filter out used
        candidates = candidates[~candidates["team"].isin(used)]
        if candidates.empty:
            break

        best = candidates.iloc[0]

        path.append(
            {
                "gw": gw,
                "team": best["team"],
                "opponent": best["opponent"],
                "is_home": bool(best["is_home"]),
                "win_prob": float(best["win_prob"]),
                "kickoff": best["kickoff"],
            }
        )
        used.add(best["team"])

    return path

def survival_prob_from_win_probs(win_probs: list[float], lives: int) -> float:
    """
    Given a sequence of weekly win probabilities and a number of lives,
    compute probability of still having at least 1 life after all weeks.

    Draw counts as a loss, so "loss" prob each week = 1 - win_prob.
    This uses a small DP over the number of losses (Poisson-binomial).
    """
    L = max(1, lives)

    # dp[k] = probability of exactly k losses so far (k = 0..L)
    dp = [0.0] * (L + 1)
    dp[0] = 1.0

    for w in win_probs:
        q = 1.0 - w  # loss prob

        new_dp = [0.0] * (L + 1)
        for k in range(L + 1):
            # No loss this week: stays at k losses
            new_dp[k] += dp[k] * w

            # Loss this week: k -> k+1 (cap at L)
            if k + 1 <= L:
                new_dp[k + 1] += dp[k] * q

        dp = new_dp

    # Survive if total losses <= lives - 1
    # (e.g. with 2 lives, you die on 2nd loss -> survive with 0 or 1 loss)
    cutoff = max(0, lives - 1)
    cutoff = min(cutoff, L)
    return sum(dp[: cutoff + 1])


def run_10_week_strategy(
    merged: pd.DataFrame,
    target_gw: int,
    used_teams: list,
    lives: int,
    horizon: int = 10,
) -> pd.DataFrame:
    """
    For the given gameweek, evaluate each possible pick as the starting choice
    of a 10-week plan, assuming future weeks are picked greedily.

    Returns a DataFrame with columns:
      team, opponent, is_home, this_week_win, survival_next_10_weeks, path
    where 'path' is a list of dicts describing the week-by-week picks.
    """
    gw_df = merged[merged["Round"] == target_gw]
    if gw_df.empty:
        print(f"No fixtures found for Round {target_gw} in 10-week strategy.")
        return pd.DataFrame()

    base_candidates = build_candidate_picks(gw_df)

    used_clean = [clean_team(t) for t in used_teams]
    if used_clean:
        base_candidates = base_candidates[~base_candidates["team"].isin(used_clean)]

    if base_candidates.empty:
        print("No candidates available this week after filtering used teams.")
        return pd.DataFrame()

    rows = []

    for _, row in base_candidates.iterrows():
        team = row["team"]
        opponent = row["opponent"]
        this_w = float(row["win_prob"])

        # Build full path of picks for this starting choice
        path = plan_future_path(
            merged=merged,
            start_gw=target_gw,
            horizon=horizon,
            used_teams=used_clean,
            first_pick_row=row,
        )

        win_probs = [step["win_prob"] for step in path]
        surv = survival_prob_from_win_probs(win_probs, lives)

        rows.append(
            {
                "team": team,
                "opponent": opponent,
                "is_home": row["is_home"],
                "this_week_win": this_w,
                "survival_next_10_weeks": surv,
                "kickoff": row["kickoff"],
                "path": path,  # store the full path as an object
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("survival_next_10_weeks", ascending=False)

    return out

def print_strategy_path(path: list[dict], lives: int):
    """
    Pretty-print the recommended 10-week path.
    """
    print("\nPlanned 10-week path:\n")
    for step in path:
        gw = step["gw"]
        team = step["team"]
        opp = step["opponent"]
        is_home = step["is_home"]
        w = step["win_prob"]

        venue = "H" if is_home else "A"
        print(
            f"GW {gw}: {team} ({venue}) vs {opp}  |  Win prob: {w:.3f}"
        )
    print(f"\nAssumes you start with {lives} lives and draws = loss.\n")


# ===================== MAIN FUNCTION ===================== #

def main():
    # --- USER INPUTS ---

    try:
        gw_input = input("Enter gameweek (Round) number: ")
        TARGET_GW = int(gw_input)
    except Exception:
        print("Invalid gameweek, defaulting to 13")
        TARGET_GW = 13

    used_input = input("Enter teams already used (comma separated), or blank: ").strip()
    if used_input:
        raw_used = [t.strip() for t in used_input.split(",") if t.strip()]
        used_teams = [clean_team(t) for t in raw_used]
    else:
        used_teams = []

    # Optional: lives for strategy
    try:
        lives_input = input("How many lives left? (default 2): ").strip()
        lives = int(lives_input) if lives_input else 2
    except Exception:
        lives = 2

    # --- DATA PREP ---

    # 1. Merge odds + fixtures and attach Round
    merged = clean_and_merge(ODDS_PATH, FIXTURES_PATH)

    # 2. Add win probabilities from bookmaker odds
    merged = add_win_probabilities(merged)

    # 2.5 ELO fallback for missing odds
    elo_table = load_elo_table("data/elo.csv")

    for idx, row in merged.iterrows():
        if pd.isna(row["p_home_win"]):
            home = row["home_clean"]
            away = row["away_clean"]

            if home in elo_table and away in elo_table:
                R_home = elo_table[home] + 100  # home advantage
                R_away = elo_table[away]

                p_home = 1 / (1 + 10 ** ((R_away - R_home) / 400))
                p_away = 1 - p_home

                merged.at[idx, "p_home_win"] = p_home
                merged.at[idx, "p_away_win"] = p_away

    # --- 1-WEEK RECOMMENDATIONS (current GW) ---

    gw_df = merged[merged["Round"] == TARGET_GW]

    if gw_df.empty:
        print(f"No fixtures found for Round {TARGET_GW}.")
        return

    candidates = build_candidate_picks(gw_df)

    if used_teams:
        candidates = candidates[~candidates["team"].isin(used_teams)]

    if candidates.empty:
        print("No valid picks after filtering used teams.")
        return

    print(f"\nTop LMS picks for gameweek {TARGET_GW} (this week only):\n")
    print(
        candidates[["team", "opponent", "is_home", "win_prob", "kickoff"]]
        .head(10)
        .to_string(index=False)
    )

    # --- 10-WEEK STRATEGY ---

    strategy_choice = input(
        "\nRun 10-week strategy based on this gameweek? (y/n): "
    ).strip().lower()

    if strategy_choice == "y":
        strat_df = run_10_week_strategy(
            merged=merged,
            target_gw=TARGET_GW,
            used_teams=used_teams,
            lives=lives,
            horizon=10,
        )

        if strat_df.empty:
            return

        print(f"\n10-week strategy suggestions starting GW {TARGET_GW} (lives = {lives}):\n")
        print(
            strat_df[
                ["team", "opponent", "is_home", "this_week_win", "survival_next_10_weeks"]
            ]
            .head(10)
            .to_string(index=False,
                       float_format=lambda x: f"{x:.3f}")
        )
# NEW: show the full path for the best starting pick
        best = strat_df.iloc[0]
        best_path = best["path"]
        print_strategy_path(best_path, lives)

if __name__ == "__main__":
    main()
