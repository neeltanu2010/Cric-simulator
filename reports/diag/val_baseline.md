# Baseline on validation window IPL 2022-24

Matches tested: 219

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 722.4 | 64.4% |
| Form baseline (recent average) | 710.1 | 63.3% |
| Random valid team | 619.7 | 55.3% |
| Best possible (hindsight) | 1121.3 | 100% |

- Model minus form: **+12.3 pts per match** (95% range -5.6 to +30.1); model won 52% of matches
- Model minus random: **+102.7 pts per match** (95% range +83.6 to +122.1); model won 78% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 10% | 719.1 | +9.0 (-2.1 to +20.2) |
| 25% | 725.2 | +15.1 (+0.8 to +29.6) |
| 50% | 730.2 | +20.0 (+4.8 to +35.4) |
| 75% | 727.6 | +17.4 (+0.7 to +33.3) |
| 90% | 722.6 | +12.4 (-4.8 to +28.6) |

## Player predictions

- Rank correlation with real points: model 0.154, form baseline 0.165, 50/50 mix 0.176
- Average error per player: model 33.1, form baseline 33.0
- Predicted top-5 players that really were top-5: model 1.55, form 1.53 (of 5)
- Real points inside the model's 10-90% range: 84% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 26.7 runs, ground average 27.9 runs
- Win prediction: accuracy 48.6%; Brier score model 0.2531 vs base rate 0.2508 (lower is better)