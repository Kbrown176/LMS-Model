# Model calibration

Walk-forward test on 1209 Premier League matches (2023/24, 2024/25, 2025/26, 2026/27). The model is refitted every week on earlier matches only, then compared with de-vigged average closing odds.

Settings: half-life 270 days, ridge 1. Lower log-loss and Brier are better.

## Overall

| forecast | log_loss | brier | matches |
|---|---|---|---|
| model | 0.9855 | 0.5877 | 1209 |
| market | nan | nan | 1209 |

## By season (log-loss)

| season | matches | model | market |
|---|---|---|---|
| 2023/24 | 380 | 0.9330 | 0.9007 |
| 2024/25 | 380 | 0.9759 | 0.9667 |
| 2025/26 | 380 | 1.0431 | 1.0118 |
| 2026/27 | 69 | 1.0096 | nan |

## Win-probability calibration

Every team in every match, bucketed by the model's win probability. `actual` is how often those teams won.

| bucket | n | model | market | actual |
|---|---|---|---|---|
| (-0.001, 0.3] | 838 | 0.195 | 0.194 | 0.184 |
| (0.3, 0.4] | 509 | 0.348 | 0.344 | 0.346 |
| (0.4, 0.5] | 426 | 0.446 | 0.441 | 0.430 |
| (0.5, 0.6] | 305 | 0.548 | 0.540 | 0.528 |
| (0.6, 0.7] | 213 | 0.647 | 0.640 | 0.648 |
| (0.7, 0.8] | 89 | 0.747 | 0.737 | 0.708 |
| (0.8, 1.0] | 38 | 0.849 | 0.822 | 0.921 |

## Promoted clubs (first PL season)

| promoted | n | model | market | actual |
|---|---|---|---|---|
| False | 2056 | 0.417 | 0.412 | 0.412 |
| True | 362 | 0.213 | 0.213 | 0.171 |

Learned promotion adjustment (log goals, applied to promoted clubs): attack +0.010, defence -0.170. Dixon-Coles rho -0.054.

## Market blend

Walk-forward on the tuning seasons. Each week the closing odds of the next two weeks stand in for the odds you'd see; team ratings are nudged towards them (`adjust_ridge`, inf = no nudge), then blended with this week's odds (`market_weight`, 0 = model only, 1 = market only).

### This week (odds available)

| adjust_ridge | market_weight | log_loss | n |
|---|---|---|---|
| 0.0300 | 0.0000 | 0.9572 | 1140 |
| 0.0300 | 0.3000 | 0.9575 | 1140 |
| 0.0100 | 0.0000 | 0.9575 | 1140 |
| 0.1000 | 0.0000 | 0.9576 | 1140 |
| 0.1000 | 0.3000 | 0.9578 | 1140 |
| 0.0100 | 0.3000 | 0.9578 | 1140 |
| 0.0300 | 0.5000 | 0.9580 | 1140 |
| 0.1000 | 0.5000 | 0.9581 | 1140 |

### 2-9 weeks ahead (no odds yet)

| adjust_ridge | log_loss | n |
|---|---|---|
| 0.0300 | 0.9841 | 7806 |
| 0.1000 | 0.9845 | 7806 |
| 0.0100 | 0.9849 | 7806 |
| 0.3000 | 0.9867 | 7806 |
| inf | 0.9989 | 7806 |
