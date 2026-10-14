# Which feature helps?

| Step | Avg points of XI | vs form baseline | Rank corr. | Score error | Brier |
|---|---:|---:|---:|---:|---:|
| players only | 729.7 | -11.9 | 0.195 | 34.1 | 0.2631 |
| + ground | 723.7 | -17.9 | 0.189 | 33.8 | 0.2651 |
| + pace/spin | 733.8 | -7.8 | 0.195 | 33.5 | 0.2666 |
| + innings/toss | 732.1 | -9.5 | 0.194 | 33.9 | 0.2635 |
| + game situation | 733.1 | -8.6 | 0.205 | 33.9 | 0.2683 |
| + extras | 731.9 | -9.8 | 0.203 | 32.3 | 0.2653 |
| + impact (full) | 732.4 | -6.8 | 0.197 | 32.4 | 0.2637 |

# Full model

Matches tested: 148

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 732.4 | 64.1% |
| Form baseline (recent average) | 739.2 | 64.7% |
| Random valid team | 615.1 | 53.8% |
| Best possible (hindsight) | 1143.3 | 100% |

- Model minus form: **-6.8 pts per match** (95% range -27.4 to +15.5); model won 40% of matches
- Model minus random: **+117.3 pts per match** (95% range +92.1 to +143.4); model won 78% of matches

## Player predictions

- Rank correlation with real points: model 0.197, form baseline 0.211
- Average error per player: model 34.4, form baseline 33.1
- Predicted top-5 players that really were top-5: model 1.61, form 1.78 (of 5)
- Real points inside the model's 10-90% range: 71% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 32.4 runs, ground average 36.1 runs
- Win prediction: accuracy 49.3%; Brier score model 0.2637 vs base rate 0.2456 (lower is better)