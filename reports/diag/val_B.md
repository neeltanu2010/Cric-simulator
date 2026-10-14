# Validation IPL 2022-24, variant B

Matches tested: 219

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 734.5 | 65.5% |
| Form baseline (recent average) | 710.1 | 63.3% |
| Random valid team | 618.5 | 55.2% |
| Best possible (hindsight) | 1121.4 | 100% |

- Model minus form: **+24.3 pts per match** (95% range +7.3 to +41.1); model won 55% of matches
- Model minus random: **+116.0 pts per match** (95% range +96.3 to +136.8); model won 78% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 10% | 724.5 | +14.4 (+0.0 to +27.9) |
| 25% | 732.1 | +21.9 (+7.3 to +36.5) |
| 50% | 737.4 | +27.2 (+11.8 to +42.7) |
| 75% | 737.2 | +27.0 (+10.6 to +43.3) |
| 90% | 732.1 | +22.0 (+5.0 to +38.6) |

## Player predictions

- Rank correlation with real points: model 0.188, form baseline 0.165, 50/50 mix 0.191
- Average error per player: model 32.4, form baseline 33.0
- Predicted top-5 players that really were top-5: model 1.63, form 1.53 (of 5)
- Real points inside the model's 10-90% range: 84% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 26.6 runs, ground average 27.9 runs
- Win prediction: accuracy 49.1%; Brier score model 0.2542 vs base rate 0.2508 (lower is better)