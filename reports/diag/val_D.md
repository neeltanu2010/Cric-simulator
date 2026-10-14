# Validation IPL 2022-24, variant D

Matches tested: 219

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 734.5 | 65.5% |
| Form baseline (recent average) | 710.1 | 63.3% |
| Random valid team | 618.5 | 55.2% |
| Best possible (hindsight) | 1121.4 | 100% |

- Model minus form: **+24.4 pts per match** (95% range +8.0 to +40.5); model won 53% of matches
- Model minus random: **+116.0 pts per match** (95% range +96.6 to +135.1); model won 79% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 10% | 724.5 | +14.4 (+0.2 to +27.9) |
| 25% | 730.8 | +20.6 (+6.2 to +35.1) |
| 50% | 734.7 | +24.5 (+9.3 to +39.2) |
| 75% | 732.1 | +21.9 (+5.3 to +38.1) |
| 90% | 733.8 | +23.7 (+7.2 to +40.1) |

## Player predictions

- Rank correlation with real points: model 0.184, form baseline 0.165, 50/50 mix 0.191
- Average error per player: model 32.5, form baseline 33.0
- Predicted top-5 players that really were top-5: model 1.58, form 1.53 (of 5)
- Real points inside the model's 10-90% range: 85% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 26.7 runs, ground average 27.9 runs
- Win prediction: accuracy 51.4%; Brier score model 0.2540 vs base rate 0.2508 (lower is better)