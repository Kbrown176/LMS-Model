# LMS Model

Pick optimiser for Premier League **Last Man Standing**: each gameweek you pick one team to win, you can't pick the same team twice, and a draw or a loss costs a life.

It forecasts every remaining fixture with a Dixon-Coles goals model blended with bookmaker odds, then picks the team that gives the best chance of surviving the next 8 gameweeks, rather than simply the biggest favourite this week. A scheduled GitHub Action refreshes the data twice a week and commits the latest recommendation to [`PICKS.md`](PICKS.md).

```
$ lms pick
GW8 pick  ·  2 lives  ·  planning 8 weeks  ·  deadline 2026-10-23 17:30 UTC
                 This week's options, ranked by 8-week survival
┏━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ # ┃ Team              ┃ Opponent        ┃ Win % ┃ Alive after GW15 ┃ vs best ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ 1 │ Brentford         │ Hull City (A)   │    54 │            14.7% │         │
│ 2 │ Manchester United │ Bournemouth (H) │    50 │            13.6% │    -1.2 │
│ 3 │ Arsenal           │ Everton (H)     │    68 │            13.5% │    -1.2 │
```

Arsenal are the most likely winners this week, but the optimiser saves them for Hull at home in GW10 (81%) and takes Brentford at Hull now.

## How it works

```
FPL API ─────────────┐  fixtures, gameweeks, live results
football-data.co.uk ─┼─▶ SQLite ─▶ Dixon-Coles ─▶ blend with odds ─▶ optimiser ─▶ pick + plan
The Odds API ────────┘  5 seasons PL + Championship; UK bookmaker prices
```

**Data** (`lms refresh`). Official gameweeks and fixtures come from the Fantasy Premier League API. Five seasons of Premier League and Championship results, with closing odds, come from football-data.co.uk. Live 1X2 prices from about 20 UK bookmakers come from The Odds API. Everything goes into one SQLite file through idempotent upserts. Every team name from every source maps to one canonical ID, and an unknown name stops the run instead of silently failing to match.

**Model** ([`model.py`](src/lms/model.py)). A Dixon-Coles model: Poisson goals with a correction for 0-0, 1-0, 0-1 and 1-1, the scores plain Poisson gets wrong and the ones that decide draws. Older matches count for less (270-day half-life). Team strengths are shared across the Premier League and Championship, and each division has its own scoring level and home advantage. That's how Coventry, Hull and Ipswich get Premier League ratings from their Championship form. A learned adjustment covers the known drop-off in a club's first season after promotion.

**Blending** ([`forecast.py`](src/lms/forecast.py)). Bookmaker odds only exist for the next week or two, but the plan looks 8 weeks ahead. So each team's rating is nudged until the model agrees with the odds it can see, and the nudge carries over to later fixtures. Where odds exist, the model is also blended directly with them.

**Optimiser** ([`strategy.py`](src/lms/strategy.py)). It picks a different team for each of the next 8 gameweeks to maximise the chance of still being in at the end.
- **With one life:** the target is the product of the win probabilities, so this is an assignment problem and the Hungarian algorithm solves it exactly.
- **With more lives:** survival is a Poisson-binomial tail. A multi-start local search handles it, and matches brute force on 120 random test cases.

Saving strong teams for later falls out of the maths, with no hand-picked "top sides" lists. Only this week's pick is a commitment; the plan is redone every gameweek.

## Results

Everything below is walk-forward: before each week the model is refitted on earlier matches only.

**Forecast accuracy** ([`reports/calibration.md`](reports/calibration.md)), Premier League 2023/24 to 2025/26, log-loss (lower is better):

| Forecast | Log-loss |
|---|---|
| Model alone | 0.984 |
| Bookmaker closing odds | 0.960 |
| Model + odds, this week | 0.958 |
| Model alone, 2–9 weeks ahead | 0.999 |
| Market-adjusted model, 2–9 weeks ahead | 0.984 |

The model alone doesn't beat closing odds. Few models do. Its job is to carry the market's information into weeks the bookmakers haven't priced yet, and adjusting the ratings to the odds does that measurably. The model's win probabilities are well calibrated: teams it gives 60–70% won 65% of the time.

**Last Man Standing backtest** ([`reports/backtest.md`](reports/backtest.md)). 2022/23 to 2025/26 replayed, with a competition started at every gameweek that has at least 12 left (96 competitions per setting). Table shows mean gameweeks survived:

| Strategy | 1 life | 2 lives |
|---|---|---|
| Random unused team | 0.6 | 2.2 |
| Biggest favourite each week | 2.3 | 4.8 |
| Optimiser, 1 week ahead (greedy) | 2.2 | 4.7 |
| **Optimiser, 8 weeks ahead** | **2.6** | **6.1** |
| Optimiser, 15 weeks ahead | 2.2 | 4.9 |

The gain comes from planning ahead: picking greedily from the same probabilities is no better than taking the favourite. Planning too far ahead throws the gain away, because the optimiser starts saving teams for weeks it can't forecast well. With two lives the 8-week optimiser lasts about 1.3 gameweeks longer than the favourite. With one life the edge is smaller and noisier.

## Quick start

Needs Python 3.11+ and a free [Odds API](https://the-odds-api.com) key (500 requests a month; this uses about 10). Without a key it still runs, using the model only.

```bash
git clone https://github.com/Kbrown176/LMS-Model.git
cd LMS-Model
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add your ODDS_API_KEY
lms refresh            # about a minute the first time, seconds after that
lms pick
```

## Weekly use

Record your entry in [`config/entry.toml`](config/entry.toml). Any spelling of a club name works:

```toml
lives = 2
used = ["Brentford", "Spurs"]
```

| Command | What it does |
|---|---|
| `lms refresh` | Pull fixtures, results and odds |
| `lms pick` | This week's ranked options and the recommended path (`--used`, `--lives`, `--gw`, `--horizon` override) |
| `lms probs` | Win probabilities for a gameweek; `--grid` for team × gameweek, `--csv` to export |
| `lms ratings` | Current team ratings |
| `lms report` | Write `PICKS.md` |
| `lms calibrate` | Walk-forward forecast check (`--tune`, `--blend` to re-tune) |
| `lms backtest` | Replay past seasons (about a minute) |
| `lms status` | What's in the database |

In VS Code the debug panel has launch configurations for refresh, pick and backtest.

## Automation

[`.github/workflows/weekly.yml`](.github/workflows/weekly.yml) runs every Tuesday and Friday at 06:00 UTC, and on demand. It refreshes the data, writes `PICKS.md` for your entry and commits it. Add your key as a repository secret:

```bash
gh secret set ODDS_API_KEY
```

and run it by hand with `gh workflow run weekly.yml`. [`ci.yml`](.github/workflows/ci.yml) runs ruff and the tests on every push.

## Configuration

[`config/settings.toml`](config/settings.toml) holds rules and tuning only, never team-specific opinions:

| Setting | Default | |
|---|---|---|
| `strategy.horizon` | 8 | Gameweeks to plan ahead (best in backtest) |
| `rules.lives` | 2 | Default lives if `entry.toml` doesn't say |
| `model.time_decay_half_life_days` | 270 | Tuned walk-forward |
| `blend.market_weight` / `adjust_ridge` | 0.5 / 0.03 | Tuned walk-forward |
| `strategy.avoid_derbies` | false | Skip derbies listed in [`config/derbies.csv`](config/derbies.csv) |

Derbies are always flagged. They aren't avoided by default because five seasons of Premier League data show no derby upset effect beyond what the odds already price in.

## Project layout

```
src/lms/
  teams.py        canonical team IDs and every source's spelling
  sources/        FPL, football-data.co.uk, The Odds API clients
  store.py        SQLite schema and upserts
  model.py        Dixon-Coles with time decay and promotion adjustment
  market.py       de-vig and bookmaker consensus
  forecast.py     blending and market-adjusted ratings
  strategy.py     pick optimiser
  backtest.py     past-season LMS replay
  calibration.py  walk-forward forecast evaluation
  picks.py, report.py, cli.py
tests/            offline tests on trimmed real samples and simulated data
reports/          calibration.md, backtest.md (regenerated by the CLI)
```

```bash
pytest          # about 170 tests, no network needed
ruff check .
```

## Limitations

- **Old odds stand in for live ones.** The backtest and the blend tuning use closing odds to represent the odds you'd see before the deadline. Real pre-deadline prices are a little less sharp.
- **Past gameweeks are approximate.** They're rebuilt from match dates, because past seasons have no official gameweek numbers in the data. Rearranged fixtures count as blank gameweeks.
- **Promoted clubs are still overrated.** Promoted sides won 17% of their games against 21% predicted, and the bookmakers overrate them by the same amount.
- **Rules vary between competitions.** This assumes a draw costs a life, no team reuse, and a team's first fixture counts in a double gameweek. If yours differs (for example, team reuse after all 20 have been picked), the optimiser needs adjusting.
- **Nothing beyond the horizon counts.** The plan gives no value to teams after week 8, so late in a long run it may spend a strong team that a longer view would keep.

## Data sources

[Fantasy Premier League](https://fantasy.premierleague.com) API · [football-data.co.uk](https://www.football-data.co.uk) · [The Odds API](https://the-odds-api.com). Not affiliated with any of them. This is a model for a game with friends, not betting advice.

The 2025/26 version of this project is preserved at tag [`v2025-26`](https://github.com/Kbrown176/LMS-Model/tree/v2025-26).

## License

[MIT](LICENSE)
