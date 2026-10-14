# Backtest

Matches tested: 148

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 741.8 | 64.9% |
| Form baseline (recent average) | 739.2 | 64.7% |
| Random valid team | 615.1 | 53.8% |
| Best possible (hindsight) | 1143.3 | 100% |

- Model minus form: **+2.6 pts per match** (95% range -17.8 to +25.3); model won 43% of matches
- Model minus random: **+126.7 pts per match** (95% range +101.8 to +153.3); model won 79% of matches

## Player predictions

- Rank correlation with real points: model 0.194, form baseline 0.211
- Average error per player: model 34.4, form baseline 33.1
- Predicted top-5 players that really were top-5: model 1.57, form 1.78 (of 5)
- Real points inside the model's 10-90% range: 72% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 32.5 runs, ground average 36.1 runs
- Win prediction: accuracy 47.2%; Brier score model 0.2625 vs base rate 0.2456 (lower is better)