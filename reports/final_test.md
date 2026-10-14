# FINAL held-out test: IPL 2025-26 (pooled leagues)

Matches tested: 148

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 769.0 | 67.3% |
| Form baseline (recent average) | 717.1 | 62.7% |
| Random valid team | 610.0 | 53.4% |
| Best possible (hindsight) | 1143.1 | 100% |

- Model minus form: **+52.0 pts per match** (95% range +28.1 to +75.3); model won 66% of matches
- Model minus random: **+159.0 pts per match** (95% range +130.1 to +188.9); model won 85% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 25% | 756.4 | +39.4 (+21.9 to +59.1) |
| 50% | 763.0 | +46.0 (+25.7 to +67.8) |
| 75% | 776.2 | +59.2 (+37.4 to +82.5) |

## Player predictions

- Rank correlation with real points: model 0.234, form baseline 0.203, 50/50 mix 0.235
- Average error per player: model 33.4, form baseline 34.4
- Predicted top-5 players that really were top-5: model 1.82, form 1.59 (of 5)
- Real points inside the model's 10-90% range: 85% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 31.6 runs, ground average 35.4 runs
- Win prediction: accuracy 54.2%; Brier score model 0.2533 vs base rate 0.2475 (lower is better)