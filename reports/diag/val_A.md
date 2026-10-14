# Validation IPL 2022-24, variant A

Matches tested: 219

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 728.0 | 64.9% |
| Form baseline (recent average) | 710.1 | 63.3% |
| Random valid team | 618.5 | 55.2% |
| Best possible (hindsight) | 1121.4 | 100% |

- Model minus form: **+17.9 pts per match** (95% range +1.1 to +34.4); model won 53% of matches
- Model minus random: **+109.5 pts per match** (95% range +89.6 to +130.2); model won 79% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 10% | 725.6 | +15.5 (+1.6 to +28.9) |
| 25% | 732.0 | +21.9 (+8.2 to +36.1) |
| 50% | 736.6 | +26.5 (+11.3 to +41.3) |
| 75% | 729.3 | +19.1 (+2.3 to +34.9) |
| 90% | 724.3 | +14.2 (-2.5 to +30.6) |

## Player predictions

- Rank correlation with real points: model 0.186, form baseline 0.165, 50/50 mix 0.190
- Average error per player: model 32.4, form baseline 33.0
- Predicted top-5 players that really were top-5: model 1.62, form 1.53 (of 5)
- Real points inside the model's 10-90% range: 84% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 26.6 runs, ground average 27.9 runs
- Win prediction: accuracy 51.8%; Brier score model 0.2533 vs base rate 0.2508 (lower is better)