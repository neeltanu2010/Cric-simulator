# Validation IPL 2022-24, variant C

Matches tested: 219

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 733.2 | 65.4% |
| Form baseline (recent average) | 710.1 | 63.3% |
| Random valid team | 618.5 | 55.2% |
| Best possible (hindsight) | 1121.4 | 100% |

- Model minus form: **+23.1 pts per match** (95% range +6.0 to +39.5); model won 55% of matches
- Model minus random: **+114.7 pts per match** (95% range +95.3 to +135.0); model won 79% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 10% | 725.7 | +15.5 (+1.5 to +28.6) |
| 25% | 729.8 | +19.6 (+5.9 to +34.1) |
| 50% | 736.3 | +26.1 (+11.2 to +40.9) |
| 75% | 731.2 | +21.1 (+5.2 to +36.8) |
| 90% | 730.5 | +20.4 (+3.9 to +36.7) |

## Player predictions

- Rank correlation with real points: model 0.186, form baseline 0.165, 50/50 mix 0.190
- Average error per player: model 32.5, form baseline 33.0
- Predicted top-5 players that really were top-5: model 1.59, form 1.53 (of 5)
- Real points inside the model's 10-90% range: 85% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 26.7 runs, ground average 27.9 runs
- Win prediction: accuracy 50.9%; Brier score model 0.2530 vs base rate 0.2508 (lower is better)