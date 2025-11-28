import pandas as pd

from strategy import (
    build_candidate_picks,
    run_10_week_strategy,
    print_strategy_path,
)
from future_probs import fill_missing_probs_with_elo

# Paths
ODDS_PATH = "data/epl_current_odds.csv"
FIXTURES_PATH = "data/Fixtures.csv"

# Team name cleaning map
TEAM_CLEAN = {
    # Spurs
    "Tottenham Hotspurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham Hotspur FC": "Tottenham",
    "Spurs": "Tottenham",

    # Man United
    "Man Utd": "Manchester United",
    "Man United": "Manchester United",
    "Manchester Utd": "Manchester United",
    "Manchester United": "Manchester United",
    "Manchester United FC": "Manchester United",

    # Man City
    "Man City": "Manchester City",
    "Manchester City": "Manchester City",
    "Manchester City FC": "Manchester City",

    # Wolves
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Wolverhampton W.": "Wolverhampton Wanderers",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",

    # West Ham
    "West Ham": "West Ham",
    "West Ham ": "West Ham",
    "West Ham Utd": "West Ham",
    "West Ham United": "West Ham",
    "West Ham United FC": "West Ham",

    # Leeds
    "Leeds": "Leeds United",
    "Leeds Utd": "Leeds United",
    "Leeds United": "Leeds United",
    "Leeds United FC": "Leeds United",


    # Brighton
    "Brighton": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion FC": "Brighton",

    # Nottingham Forest
    "Nott'm Forest": "Nottingham Forest",
    "Nottingham Forest": "Nottingham Forest",
    "Nottingham Forest FC": "Nottingham Forest",

    # Newcastle
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Newcastle United": "Newcastle United",
    "Newcastle United FC": "Newcastle United",

    # Others from your sample
    "Sunderland AFC": "Sunderland",
    "Brentford FC": "Brentford",
    "Everton FC": "Everton",
    "Crystal Palace FC": "Crystal Palace",
    "Aston Villa FC": "Aston Villa",
    "Chelsea FC": "Chelsea",
    "Arsenal FC": "Arsenal",
    "AFC Bournemouth": "Bournemouth",
    "Fulham FC": "Fulham",
    "Burnley FC": "Burnley",
    "Everton FC": "Everton",
    "Liverpool FC": "Liverpool",
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
        [
            "date_clean",
            "home_clean",
            "away_clean",
            "home_odds",
            "draw_odds",
            "away_odds",
            "kickoff_utc",
        ]
    ]

    # IMPORTANT: base = fixtures (full season), left join odds onto it
    merged = fixtures.merge(
        odds_small,
        on=["date_clean", "home_clean", "away_clean"],
        how="left",
    )

    return merged


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

    # --- Sanity checks before Elo fallback ---
    total_rows = len(merged)
    missing_before = merged["home_odds"].isna().sum()
    print(f"\n[INFO] Fixtures loaded: {total_rows}")
    print(f"[INFO] Missing bookmaker odds before Elo fallback: {missing_before}")

    # 2. Add win probabilities from bookmaker odds
    merged = add_win_probabilities(merged)

    # 2.5 ELO fallback for missing odds
    elo_table = pd.read_csv("data/elo.csv")
    merged = fill_missing_probs_with_elo(merged, elo_table)

    # --- Sanity checks after Elo fallback ---
    missing_after = merged["p_home_win"].isna().sum()
    elo_filled = (merged["prob_source"] == "elo").sum()
    print(f"[INFO] Fixtures using Elo fallback: {elo_filled}")
    if missing_after > 0:
        print(f"[WARNING] {missing_after} fixtures STILL have no probabilities!")


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

        print(
            f"\n10-week strategy suggestions starting GW {TARGET_GW} (lives = {lives}):\n"
        )
        print(
            strat_df[
                ["team", "opponent", "is_home", "this_week_win", "worst_week", "avg_week", "survival_next_10_weeks", "score"]
            ]
            .head(10)
            .to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

        # Show the full path for the best starting pick
        best = strat_df.iloc[0]
        best_path = best["path"]
        print_strategy_path(best_path, lives)


if __name__ == "__main__":
    main()
