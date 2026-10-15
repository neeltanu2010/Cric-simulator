# Held-out test IPL 2025-26, re-run after fixing double-counted World Cup matches (same settings as final_v3)

Matches tested: 148

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 752.3 | 65.8% |
| Form baseline (recent average) | 698.4 | 61.1% |
| Random valid team | 609.5 | 53.3% |
| Best possible (hindsight) | 1143.0 | 100% |

- Model minus form: **+53.9 pts per match** (95% range +28.6 to +79.0); model won 64% of matches
- Model minus random: **+142.8 pts per match** (95% range +116.1 to +170.7); model won 80% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 10% | 729.8 | +31.4 (+12.9 to +49.7) |
| 25% | 736.0 | +37.6 (+17.0 to +58.2) |
| 50% | 744.7 | +46.3 (+24.7 to +67.8) |
| 75% | 747.6 | +49.2 (+26.2 to +72.6) |
| 90% | 748.2 | +49.8 (+24.8 to +74.3) |

## Player predictions

- Rank correlation with real points: model 0.225, form baseline 0.203, 50/50 mix 0.228
- Average error per player: model 33.3, form baseline 33.5
- Predicted top-5 players that really were top-5: model 1.85, form 1.74 (of 5)
- Real points inside the model's 10-90% range: 85% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 31.3 runs, ground average 36.2 runs
- Win prediction: accuracy 46.5%; Brier score model 0.2561 vs base rate 0.2475 (lower is better)