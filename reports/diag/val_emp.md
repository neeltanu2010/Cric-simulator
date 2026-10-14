# Baseline on validation window IPL 2022-24

Matches tested: 219

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 729.7 | 65.1% |
| Form baseline (recent average) | 710.1 | 63.3% |
| Random valid team | 619.7 | 55.3% |
| Best possible (hindsight) | 1121.3 | 100% |

- Model minus form: **+19.6 pts per match** (95% range +2.6 to +35.3); model won 57% of matches
- Model minus random: **+110.1 pts per match** (95% range +90.4 to +131.0); model won 79% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 10% | 722.7 | +12.5 (+2.4 to +23.2) |
| 25% | 734.9 | +24.8 (+11.0 to +39.1) |
| 50% | 734.2 | +24.0 (+9.9 to +38.4) |
| 75% | 730.0 | +19.9 (+3.6 to +36.0) |
| 90% | 727.5 | +17.4 (-0.0 to +32.9) |

## Player predictions

- Rank correlation with real points: model 0.175, form baseline 0.165, 50/50 mix 0.184
- Average error per player: model 32.5, form baseline 33.0
- Predicted top-5 players that really were top-5: model 1.61, form 1.53 (of 5)
- Real points inside the model's 10-90% range: 84% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 26.5 runs, ground average 27.9 runs
- Win prediction: accuracy 49.1%; Brier score model 0.2536 vs base rate 0.2508 (lower is better)