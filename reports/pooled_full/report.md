# Backtest

Matches tested: 148

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 732.5 | 64.1% |
| Form baseline (recent average) | 717.1 | 62.7% |
| Random valid team | 619.3 | 54.2% |
| Best possible (hindsight) | 1143.5 | 100% |

- Model minus form: **+15.4 pts per match** (95% range -7.2 to +38.8); model won 54% of matches
- Model minus random: **+113.3 pts per match** (95% range +88.8 to +138.7); model won 75% of matches

## Player predictions

- Rank correlation with real points: model 0.209, form baseline 0.202
- Average error per player: model 34.2, form baseline 34.4
- Predicted top-5 players that really were top-5: model 1.67, form 1.59 (of 5)
- Real points inside the model's 10-90% range: 72% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 32.4 runs, ground average 35.4 runs
- Win prediction: accuracy 49.3%; Brier score model 0.2686 vs base rate 0.2475 (lower is better)