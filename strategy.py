import pandas as pd
from derby import derby_intensity
from team_strength import is_weak_team
import importlib, strategy
importlib.reload(strategy)


# =====================
# 0. Config / Tags
# =====================

# Teams you generally want to conserve and treat as "elite"
TOP_SIDES = {
    "Manchester City",
    "Arsenal",
    "Liverpool",
    "Manchester United",
    "Chelsea",
    "Tottenham",
}

# Rough variance tags – tweak as you like
HIGH_VARIANCE_TEAMS = {
    "Brighton",
    "Tottenham",
    "Brentford",
    "Newcastle United",
}

MEDIUM_VARIANCE_TEAMS = {
    "Chelsea",
    "Manchester United",
    "Aston Villa",
    "Crystal Palace",
    "Wolverhampton Wanderers",
    "Bournemouth",
    "West Ham",
    "Leeds United",
}


# =====================
# 1. Helper penalties
# =====================

def cautious_score(this_week, ten_week_survival, lives):
    """
    When lives == 1:
        Be cautious — heavily weight this week's safety.
    When lives > 1:
        Use long-term survival probability.
    """
    if lives > 1:
        return ten_week_survival
    return 0.70 * this_week + 0.30 * ten_week_survival


def variance_penalty(team: str) -> float:
    """
    Penalise high-variance teams a bit so they are less likely
    to be used early in LMS, even if odds like them.
    """
    if team in HIGH_VARIANCE_TEAMS:
        return 0.92   # ~8% haircut
    if team in MEDIUM_VARIANCE_TEAMS:
        return 0.97   # small haircut
    return 1.00      # stable / low-variance teams


def is_midtable_vs_top(team: str, opponent: str) -> bool:
    """
    Always avoid picking a non-elite team when the opponent is a top side.
    E.g. don't pick Brentford vs Liverpool, Brighton vs Spurs, etc.
    """
    return (team not in TOP_SIDES) and (opponent in TOP_SIDES)


def filter_lms_candidates(candidates: pd.DataFrame, min_safe_p: float = 0.60) -> pd.DataFrame:
    """
    Apply LMS-specific safety rules to a set of candidate picks for a single GW.

    Rules:
      1) Never pick mid-table/weak vs TOP_SIDES (is_midtable_vs_top).
      2) Prefer non-derby picks when any non-derby candidate has win_prob >= min_safe_p.
      3) If NO such safe non-derby exists, allow derbies but only if the picked team is in TOP_SIDES
         (e.g. pick Liverpool in a derby if you're desperate, but never Everton vs Liverpool).
    """
    if candidates.empty:
        return candidates

    # 1) Drop midtable vs top-side picks
    candidates = candidates[
        ~candidates.apply(
            lambda r: is_midtable_vs_top(r["team"], r["opponent"]),
            axis=1,
        )
    ]
    if candidates.empty:
        return candidates

    # 2) Helper predicates
    def is_safe_non_derby(row):
        return (
            derby_intensity(row["team"], row["opponent"]) == 0.0
            and row["win_prob"] >= min_safe_p
        )

    def too_volatile(row):
        team = row["team"]
        opp = row["opponent"]
        p = row["win_prob"]

        if team in HIGH_VARIANCE_TEAMS:
            # If opponent is genuinely weak (bottom 5), allow even if p is moderate
            if is_weak_team(opp, bottom_n=5):
                return False

        # Otherwise require a higher probability to trust a volatile side
        if p < 0.72:
            return True

        return False

    # Drop too volatile picks
    candidates = candidates[~candidates.apply(too_volatile, axis=1)]
    if candidates.empty:
        return candidates

    # Safe non-derby mask (one value per row)
    safe_non_derby_mask = candidates.apply(is_safe_non_derby, axis=1)

    # Make absolutely sure this is a plain bool for the if-statement
    safe_non_derby_exists = bool(safe_non_derby_mask.any())

    if safe_non_derby_exists:
        # We have at least one safe non-derby: only keep those
        candidates = candidates[safe_non_derby_mask]
    else:
        # No safe non-derby options:
        # Allow derbies, but only if the picked side is elite;
        # still disallow midtable vs top (already removed).
        def illegal_derby(row):
            i = derby_intensity(row["team"], row["opponent"])
            return i > 0 and (row["team"] not in TOP_SIDES)

        candidates = candidates[
            ~candidates.apply(illegal_derby, axis=1)
        ]

    return candidates



# =====================
# 2. Build Candidates
# =====================

def build_candidate_picks(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a given gameweek's fixtures:
        - candidate = pick home or away side
        - attach win_prob
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


# =====================
# 3. Future Fixture Conservation
# =====================

def compute_future_pick_value(
    merged: pd.DataFrame,
    start_gw: int,
    prob_threshold: float = 0.65,
) -> dict:
    """
    For all gameweeks AFTER start_gw, count how many 'good' future
    opportunities each team has (win_prob >= prob_threshold).

    Returns dict: team -> count_of_good_future_fixtures
    """
    future_df = merged[merged["Round"] > start_gw]
    if future_df.empty:
        return {}

    future_candidates = build_candidate_picks(future_df)
    if future_candidates.empty:
        return {}

    good = future_candidates[future_candidates["win_prob"] >= prob_threshold]
    if good.empty:
        return {}

    counts = good.groupby("team").size().to_dict()
    return counts


def conservation_factor_for_team(
    team: str,
    future_value: dict,
    max_cap: int = 6,
    per_game_penalty: float = 0.03,
    min_factor: float = 0.8,
) -> float:
    """
    Convert 'future_value' (number of good future fixtures) into
    a multiplicative penalty factor in [min_factor, 1].

    - More good future fixtures -> stronger penalty -> lower factor.
    """
    fv = future_value.get(team, 0)
    fv_capped = min(fv, max_cap)

    factor = 1.0 - per_game_penalty * fv_capped
    if factor < min_factor:
        factor = min_factor

    return factor


# =====================
# 4. Future Path Planner
# =====================

def plan_future_path(
    merged: pd.DataFrame,
    start_gw: int,
    horizon: int,
    used_teams: list,
    first_pick_row: pd.Series,
) -> list:
    """
    Given a starting pick in start_gw,
    build a greedy path for next `horizon` gameweeks.
    Applies derby + variance penalties to each weekly probability.

    IMPORTANT:
    - We NEVER skip a gameweek if there is *any* team available.
    - We first try LMS safety rules; if they leave no candidates,
      we fall back to the best raw candidate for that week.
    """
    path = []
    used = set(used_teams)

    # --- Week 1 (forced pick) ---
    team = first_pick_row["team"]
    opponent = first_pick_row["opponent"]
    raw_wp = float(first_pick_row["win_prob"])

    # Derby penalty
    intensity = derby_intensity(team, opponent)
    if intensity > 0:
        wp = raw_wp * (1 - 0.18 * intensity)
    else:
        wp = raw_wp

    # Variance penalty
    wp *= variance_penalty(team)

    path.append(
        {
            "gw": start_gw,
            "team": team,
            "opponent": opponent,
            "is_home": bool(first_pick_row["is_home"]),
            "win_prob": wp,
            "kickoff": first_pick_row["kickoff"],
        }
    )
    used.add(team)

    # --- Future weeks ---
    # horizon = total number of weeks including start_gw,
    # so we need horizon-1 extra weeks.
    for gw in range(start_gw + 1, start_gw + horizon):
        gw_df = merged[merged["Round"] == gw]
        if gw_df.empty:
            # no fixtures at all this round -> we genuinely can't pick anyone
            break

        # Build all candidates for this GW
        candidates_all = build_candidate_picks(gw_df)
        if candidates_all.empty:
            break

        # Remove already used teams
        candidates_avail = candidates_all[~candidates_all["team"].isin(used)]
        if candidates_avail.empty:
            # all teams have been used already; can't extend further
            break

        # Try to apply LMS safety rules
        safe_candidates = filter_lms_candidates(candidates_avail)

        if not safe_candidates.empty:
            # Great: pick the best LMS-safe candidate
            best = safe_candidates.iloc[0]
        else:
            # Fallback: **still** pick someone.
            # Take the best remaining candidate by raw win_prob.
            best = candidates_avail.sort_values("win_prob", ascending=False).iloc[0]

        team = best["team"]
        opponent = best["opponent"]
        raw_wp = float(best["win_prob"])

        # Derby penalty
        intensity = derby_intensity(team, opponent)
        if intensity > 0:
            wp = raw_wp * (1 - 0.18 * intensity)
        else:
            wp = raw_wp

        # Variance penalty
        wp *= variance_penalty(team)

        path.append(
            {
                "gw": gw,
                "team": team,
                "opponent": opponent,
                "is_home": bool(best["is_home"]),
                "win_prob": wp,
                "kickoff": best["kickoff"],
            }
        )
        used.add(team)

    return path



# =====================
# 5. Survival Probability (DP)
# =====================

def survival_prob_from_win_probs(win_probs: list, lives: int) -> float:
    """
    DP model for lives. Draw = loss.
    Returns probability of surviving all weeks.
    """
    L = max(1, lives)

    dp = [0.0] * (L + 1)
    dp[0] = 1.0

    for w in win_probs:
        q = 1 - w  # loss probability

        new_dp = [0.0] * (L + 1)
        for k in range(L + 1):
            new_dp[k] += dp[k] * w
            if k + 1 <= L:
                new_dp[k + 1] += dp[k] * q
        dp = new_dp

    cutoff = max(0, lives - 1)
    cutoff = min(cutoff, L)

    return sum(dp[: cutoff + 1])


# =====================
# 6. Full 10-Week Strategy
# =====================

def run_10_week_strategy(
    merged: pd.DataFrame,
    target_gw: int,
    used_teams: list,
    lives: int,
    horizon: int = 10,
) -> pd.DataFrame:

    gw_df = merged[merged["Round"] == target_gw]
    if gw_df.empty:
        print(f"No fixtures found for Round {target_gw}.")
        return pd.DataFrame()

    base_candidates = build_candidate_picks(gw_df)

    # Filter out already-used teams
    if used_teams:
        base_candidates = base_candidates[~base_candidates["team"].isin(used_teams)]

    if base_candidates.empty:
        print("No valid candidates this week after filtering used teams.")
        return pd.DataFrame()

    # Apply LMS safety rules for the starting GW
    base_candidates = filter_lms_candidates(base_candidates)
    if base_candidates.empty:
        print("No safe LMS candidates this week after applying safety rules.")
        return pd.DataFrame()

    # Future fixture value (conservation)
    future_value = compute_future_pick_value(merged, target_gw)

    rows = []

    for _, row in base_candidates.iterrows():
        team = row["team"]
        opponent = row["opponent"]

        # --- This week's adjusted win prob ---
        raw_this_w = float(row["win_prob"])
        i = derby_intensity(team, opponent)

        if i > 0:
            this_w = raw_this_w * (1 - 0.18 * i)
        else:
            this_w = raw_this_w

        this_w *= variance_penalty(team)

        # --- Build future path starting from this pick ---
        path = plan_future_path(
            merged=merged,
            start_gw=target_gw,
            horizon=horizon,
            used_teams=used_teams,
            first_pick_row=row,
        )

        win_probs = [step["win_prob"] for step in path]

        # Long-term survival
        surv = survival_prob_from_win_probs(win_probs, lives)

        # Conservative scoring
        base_score = cautious_score(this_w, surv, lives)

        # Conservation penalty (don't waste teams with lots of good future fixtures)
        cf = conservation_factor_for_team(team, future_value)
        score = base_score * cf

        # Metrics
        worst_week = min(win_probs) if win_probs else None
        avg_week = sum(win_probs) / len(win_probs) if win_probs else None
        fv = future_value.get(team, 0)

        rows.append(
            {
                "team": team,
                "opponent": opponent,
                "is_home": row["is_home"],
                "this_week_win": this_w,
                "worst_week": worst_week,
                "avg_week": avg_week,
                "future_good_fixtures": fv,
                "survival_next_10_weeks": surv,
                "score": score,
                "kickoff": row["kickoff"],
                "path": path,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values("score", ascending=False)

    return out


# =====================
# 7. Pretty Printer
# =====================

def print_strategy_path(path: list, lives: int):
    print("\nPlanned 10-week path:\n")
    for step in path:
        gw = step["gw"]
        team = step["team"]
        opp = step["opponent"]
        wp = step["win_prob"]
        v = "H" if step["is_home"] else "A"
        print(f"GW {gw}: {team} ({v}) vs {opp}  |  Win prob: {wp:.3f}")
    print(f"\nAssuming {lives} lives (draw = loss).\n")
