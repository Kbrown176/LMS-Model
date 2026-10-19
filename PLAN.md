# LMS Model — 2026/27 Rebuild Plan

## Decisions
- **Repo:** same repo (`Kbrown176/LMS-Model`). Old code tagged `v2025-26`, rebuild on branch `rebuild`, merged to `main` when done.
- **Rules:** draw = out, multiple lives (configurable; default 2), no team reuse.
- **Output:** CLI + scheduled GitHub Action that commits `PICKS.md` each gameweek.
- **History:** 5 seasons (2021/22 → 2025/26) of Premier League + Championship, plus the current season.

## Guiding rule
Nothing about specific teams is hardcoded. Ratings, tiers and form come from data on every run.
The only hand-written team data is `config/derbies.csv`, which is reviewed for 2026/27 and tracked in git.

## Data sources
| Source | Key? | Used for |
|---|---|---|
| Fantasy Premier League API | No | Official gameweeks, fixtures, kickoff times, live results, current clubs |
| football-data.co.uk CSVs | No | Past PL + Championship results and closing odds (model training + backtest) |
| The Odds API | Yes (`ODDS_API_KEY`) | Live bookmaker odds for upcoming fixtures |

football-data.org is dropped, along with its key.

## Layout
```
src/lms/
  teams.py       # canonical team IDs + alias map for all sources; unknown names raise
  sources/       # fpl.py, footballdata.py, odds.py — fetch + normalise
  store.py       # SQLite (data/lms.db), idempotent upserts
  model.py       # Dixon-Coles with time decay → P(home/draw/away)
  blend.py       # combine model + market odds (de-vigged, multi-bookmaker average)
  strategy.py    # optimal non-repeating pick path over N gameweeks, with lives
  backtest.py    # replay past seasons week by week
  cli.py         # lms refresh | lms pick | lms backtest | lms report
config/          # settings.toml, derbies.csv
tests/           # offline sample data, no API calls
.github/workflows/  # ci.yml (lint+test), weekly.yml (refresh → PICKS.md)
.vscode/         # settings, debug configs, recommended extensions
```

## Phases
1. **Archive + scaffold.** Commit the pending `strategy.py` change, then tag `v2025-26`. Create the `rebuild` branch with `pyproject.toml`, venv, ruff, pytest, `.env.example`, `.gitignore`, `.vscode/`, and remove the tracked `.DS_Store` files.
2. **Data layer.** Write the three source clients and one team-ID map. `lms refresh` fills SQLite incrementally and caches the downloaded CSVs.
3. **Model.** Dixon-Coles fitted on PL + Championship with exponential time decay (promoted clubs rated from Championship form). Check calibration against closing odds (log-loss / Brier).
4. **Blend.** Use market probabilities where odds exist, and the model elsewhere, with weights tuned on past seasons.
5. **Strategy.** Choose distinct teams for the next N gameweeks to maximise survival probability with k lives (draw = out). Derby and minimum-probability rules are configurable. Output shows this week's ranked picks plus the full recommended path.
6. **Backtest.** 2023/24 → 2025/26: weeks survived vs simple approaches (biggest favourite each week, random), plus tuning of the horizon and blend weights.
7. **CLI + automation.** `lms pick --gw 8 --used Arsenal,Chelsea --lives 2`. A weekly GitHub Action (odds key as a repo secret) refreshes the data and commits `PICKS.md`. A CI workflow runs lint and tests.
8. **README + release.** Covers how it works, setup, usage, sample output, backtest results and limitations. MIT licence. Merge to `main` and push with `gh`.

## Target
Phases 1–5 + CLI done before the GW8 deadline (Fri 23 Oct 2026, 17:30 UTC), then backtest, automation and README.
