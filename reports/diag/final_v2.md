# Held-out test IPL 2025-26: improved setting (realistic lineups + slower form, 50% mix)

Matches tested: 148

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 760.9 | 66.6% |
| Form baseline (recent average) | 717.1 | 62.7% |
| Random valid team | 610.0 | 53.4% |
| Best possible (hindsight) | 1143.1 | 100% |

- Model minus form: **+43.8 pts per match** (95% range +21.8 to +66.8); model won 65% of matches
- Model minus random: **+150.8 pts per match** (95% range +123.3 to +178.7); model won 80% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 10% | 744.3 | +27.2 (+11.5 to +44.0) |
| 25% | 756.5 | +39.4 (+19.5 to +58.9) |
| 50% | 757.3 | +40.2 (+18.9 to +62.0) |
| 75% | 762.4 | +45.3 (+23.3 to +68.3) |
| 90% | 762.4 | +45.4 (+23.0 to +68.2) |

## Player predictions

- Rank correlation with real points: model 0.235, form baseline 0.203, 50/50 mix 0.238
- Average error per player: model 32.9, form baseline 34.4
- Predicted top-5 players that really were top-5: model 1.87, form 1.59 (of 5)
- Real points inside the model's 10-90% range: 85% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 31.0 runs, ground average 35.4 runs
- Win prediction: accuracy 47.2%; Brier score model 0.2540 vs base rate 0.2475 (lower is better)