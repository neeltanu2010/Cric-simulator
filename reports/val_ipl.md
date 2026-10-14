# Validation 2023-24 (ipl)

Matches tested: 100

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 688.6 | 60.9% |
| Form baseline (recent average) | 709.8 | 62.8% |
| Random valid team | 611.6 | 54.1% |
| Best possible (hindsight) | 1130.2 | 100% |

- Model minus form: **-21.2 pts per match** (95% range -49.1 to +7.1); model won 43% of matches
- Model minus random: **+77.0 pts per match** (95% range +50.3 to +105.6); model won 68% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 25% | 713.3 | +3.5 (-17.2 to +24.7) |
| 50% | 704.4 | -5.5 (-29.0 to +18.7) |
| 75% | 699.6 | -10.2 (-36.2 to +17.0) |

## Player predictions

- Rank correlation with real points: model 0.168, form baseline 0.208, 50/50 mix 0.195
- Average error per player: model 33.7, form baseline 33.1
- Predicted top-5 players that really were top-5: model 1.48, form 1.70 (of 5)
- Real points inside the model's 10-90% range: 81% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 28.1 runs, ground average 32.7 runs
- Win prediction: accuracy 52.0%; Brier score model 0.2572 vs base rate 0.2509 (lower is better)