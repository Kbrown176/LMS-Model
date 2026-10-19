# Backtest

Last Man Standing replayed over 2022/23, 2023/24, 2024/25, 2025/26. A competition starts at every gameweek with at least 12 left; each strategy picks until it runs out of lives (a draw or loss costs one) or makes 20 picks. The model is refitted before every gameweek on earlier results only, and the closing odds of that gameweek and the next stand in for the odds you'd see at the deadline.

Strategies:

- **optimiser H=n**: this project; best distinct-team plan over the next n gameweeks, play its first pick, re-plan next week
- **biggest favourite**: the unused team with the shortest odds this week
- **random**: any unused team (averaged over 20 runs)

`mean`/`median` = gameweeks survived. `vs favourite` = extra gameweeks survived in the same competitions, with a rough standard error (`se`; overlapping competitions make it optimistic).

| lives | strategy | competitions | mean | median | 5+ rounds | 10+ rounds | vs favourite | se |
|---|---|---|---|---|---|---|---|---|
| 1 | optimiser H=10 | 96 | 2.906 | 2.000 | 0.240 | 0.021 | 0.562 | 0.311 |
| 1 | optimiser H=3 | 96 | 2.740 | 2.000 | 0.250 | 0.010 | 0.396 | 0.241 |
| 1 | optimiser H=12 | 96 | 2.625 | 2.000 | 0.198 | 0.010 | 0.281 | 0.318 |
| 1 | optimiser H=8 | 96 | 2.604 | 2.000 | 0.229 | 0.000 | 0.260 | 0.258 |
| 1 | optimiser H=5 | 96 | 2.594 | 2.000 | 0.250 | 0.000 | 0.250 | 0.222 |
| 1 | biggest favourite | 96 | 2.344 | 2.000 | 0.146 | 0.010 | 0.000 | 0.000 |
| 1 | optimiser H=1 | 96 | 2.177 | 2.000 | 0.115 | 0.000 | -0.167 | 0.102 |
| 1 | optimiser H=15 | 96 | 2.177 | 2.000 | 0.146 | 0.000 | -0.167 | 0.268 |
| 1 | random | 1920 | 0.610 | 0.000 | 0.009 | 0.000 | -1.733 | 0.209 |
| 2 | optimiser H=8 | 96 | 6.135 | 6.000 | 0.625 | 0.188 | 1.302 | 0.384 |
| 2 | optimiser H=5 | 96 | 5.802 | 5.000 | 0.562 | 0.125 | 0.969 | 0.376 |
| 2 | optimiser H=10 | 96 | 5.677 | 5.000 | 0.583 | 0.125 | 0.844 | 0.406 |
| 2 | optimiser H=12 | 96 | 5.562 | 5.000 | 0.573 | 0.156 | 0.729 | 0.421 |
| 2 | optimiser H=3 | 96 | 5.219 | 5.000 | 0.542 | 0.083 | 0.385 | 0.286 |
| 2 | optimiser H=15 | 96 | 4.854 | 5.000 | 0.510 | 0.042 | 0.021 | 0.383 |
| 2 | biggest favourite | 96 | 4.833 | 4.000 | 0.396 | 0.094 | 0.000 | 0.000 |
| 2 | optimiser H=1 | 96 | 4.729 | 4.000 | 0.438 | 0.073 | -0.104 | 0.099 |
| 2 | random | 1920 | 2.184 | 2.000 | 0.064 | 0.000 | -2.649 | 0.280 |

## By season (mean gameweeks survived, H=8)

| lives | strategy | 2022/23 | 2023/24 | 2024/25 | 2025/26 |
|---|---|---|---|---|---|
| 1 | biggest favourite | 2.13 | 2.50 | 2.58 | 2.16 |
| 1 | optimiser H=8 | 3.22 | 2.59 | 2.08 | 2.60 |
| 2 | biggest favourite | 4.65 | 4.73 | 5.35 | 4.56 |
| 2 | optimiser H=8 | 7.78 | 5.95 | 4.85 | 6.12 |

## Reading it

- Planning ahead is where the gain is: H=1 (greedy on the same probabilities) is no better than taking the biggest favourite.
- Too long a horizon hurts. The optimiser starts saving strong teams for weeks so far ahead that the forecasts are poor.
- LMS is high-variance: even the best strategy is usually out within a handful of gameweeks with one life, and season-to-season swings are large.
