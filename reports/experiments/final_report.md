# Final: best setting from 68 tried, tested on IPL 2025-26

```json
{
 "features": [
  "lineup",
  "form",
  "venue",
  "type",
  "innings",
  "state",
  "extras",
  "impact",
  "league"
 ],
 "sigmas": [
  0.7,
  0.35,
  0.2
 ],
 "half_life_days": 3000.0,
 "prior_scale": 4.0,
 "pool": "leagues",
 "simulator_share": 0.5,
 "chosen_on": "IPL 2023-24 validation",
 "validation_xi_points": 740.1103448275862,
 "test_xi_points": 760.6621621621622,
 "test_form_points": 717.0641891891892
}
```

# Held-out test, best setting

Matches tested: 148

## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)

| Method | Avg points | % of best-possible |
|---|---:|---:|
| Model | 765.7 | 67.0% |
| Form baseline (recent average) | 717.1 | 62.7% |
| Random valid team | 610.0 | 53.4% |
| Best possible (hindsight) | 1143.1 | 100% |

- Model minus form: **+48.6 pts per match** (95% range +26.2 to +71.4); model won 65% of matches
- Model minus random: **+155.7 pts per match** (95% range +127.5 to +183.8); model won 82% of matches

## Mixing the simulator with the form baseline

| Share of simulator | Avg points | vs form baseline (95% range) |
|---|---:|---|
| 10% | 738.4 | +21.3 (+7.5 to +37.0) |
| 25% | 757.1 | +40.1 (+22.0 to +59.4) |
| 50% | 760.7 | +43.6 (+22.7 to +65.5) |
| 75% | 768.5 | +51.4 (+29.9 to +74.7) |
| 90% | 770.9 | +53.8 (+32.4 to +78.7) |

## Player predictions

- Rank correlation with real points: model 0.231, form baseline 0.203, 50/50 mix 0.235
- Average error per player: model 33.3, form baseline 34.4
- Predicted top-5 players that really were top-5: model 1.80, form 1.59 (of 5)
- Real points inside the model's 10-90% range: 84% (80% would be perfectly calibrated)

## Match predictions

- First-innings score error: model 31.0 runs, ground average 35.4 runs
- Win prediction: accuracy 52.1%; Brier score model 0.2534 vs base rate 0.2475 (lower is better)