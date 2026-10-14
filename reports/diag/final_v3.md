# Held-out test IPL 2025-26: confirmed XI and subs + realistic lineups + slower form, 50% mix

Matches tested: 148

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 755.0 | 66.0% |
| Form baseline (recent average) | 716.6 | 62.7% |
| Random valid team | 610.7 | 53.4% |
| Best possible (hindsight) | 1143.1 | 100% |

- Model minus form: **+38.4 pts per match** (95% range +16.2 to +60.9); model won 61% of matches
- Model minus random: **+144.3 pts per match** (95% range +116.4 to +171.5); model won 81% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 10% | 742.1 | +25.4 (+8.5 to +42.6) |
| 25% | 756.8 | +40.2 (+21.3 to +59.1) |
| 50% | 755.9 | +39.3 (+18.9 to +60.1) |
| 75% | 757.6 | +40.9 (+19.2 to +63.7) |
| 90% | 756.7 | +40.1 (+18.2 to +61.5) |

## Player predictions

- Rank correlation with real points: model 0.223, form baseline 0.203, 50/50 mix 0.232
- Average error per player: model 33.2, form baseline 34.4
- Predicted top-5 players that really were top-5: model 1.82, form 1.59 (of 5)
- Real points inside the model's 10-90% range: 85% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 31.1 runs, ground average 35.4 runs
- Win prediction: accuracy 44.4%; Brier score model 0.2559 vs base rate 0.2475 (lower is better)