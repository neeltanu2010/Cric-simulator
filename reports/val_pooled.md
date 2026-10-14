# Validation 2023-24 (pooled)

Matches tested: 100

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 705.8 | 62.3% |
| Form baseline (recent average) | 713.8 | 63.0% |
| Random valid team | 612.8 | 54.1% |
| Best possible (hindsight) | 1133.8 | 100% |

- Model minus form: **-8.0 pts per match** (95% range -31.9 to +17.8); model won 41% of matches
- Model minus random: **+93.0 pts per match** (95% range +67.0 to +119.9); model won 70% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 25% | 739.5 | +25.7 (+5.7 to +46.3) |
| 50% | 731.6 | +17.8 (-3.5 to +40.0) |
| 75% | 721.6 | +7.8 (-16.4 to +34.4) |

## Player predictions

- Rank correlation with real points: model 0.179, form baseline 0.200, 50/50 mix 0.202
- Average error per player: model 33.5, form baseline 33.2
- Predicted top-5 players that really were top-5: model 1.45, form 1.52 (of 5)
- Real points inside the model's 10-90% range: 83% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 28.0 runs, ground average 31.8 runs
- Win prediction: accuracy 48.0%; Brier score model 0.2582 vs base rate 0.2500 (lower is better)