# Cric-simulator

Ball-by-ball T20 simulator that predicts **Dream11 fantasy points for every player**, picks the best XI, captain and vice-captain, and **backtests itself** against what really happened.

## What the model includes
| Area | How it is modelled |
|---|---|
| Player vs player | Each ball's chances (dot, 1, 2, 3, 4, 6, wicket) = league average x batter effect x bowler effect, by phase (powerplay / middle / death). Few balls of data = pulled toward average. Recent matches count more. |
| Pace vs spin | Bowler type changes the league average, each batter has a vs-pace and vs-spin effect, and each ground has its own pace/spin effect. Wickets happen differently (more stumpings vs spin). |
| Ground / pitch | Per-ground effect on scoring and wickets, learned from that ground's history. |
| Toss / chase | Per-ground, per-innings effect (this is how "chasing is easier here" is learned). Who bats first is an input after the toss. |
| Match situation | New batter effect, batting position, and required-rate / wickets-lost pressure when chasing. |
| Extras | Wides, no-balls (and the free hit next ball), byes / leg byes, by bowler and phase. |
| Run outs | Own chance per ball, striker or non-striker. |
| Impact player | A bowling sub can replace a batter who has batted; a batting sub can replace a bowler in the chase. Whether it happens is random per simulation (`p_used`). |
| Dream11 points | Standard T20 scoring, including dot balls, maidens, economy / strike-rate bands, catches, stumpings, run outs. |
| Team pick | Best XI under 100 credits, role limits, max 7 per team, captain (2x) and vice (1.5x). |

## Use
```
pip install -r requirements.txt
# 1. put Cricsheet "IPL (JSON)" files in data/ipl_json/
# 2. (recommended) write a bowling-type file, fix our pace/spin guesses by hand:
python -m cric_sim.cli types --data data/ipl_json --out data/players.csv
# 3. see what the model learned about grounds:
python -m cric_sim.cli venues --data data/ipl_json
# 4. backtest. Add --ablate to see what each feature adds, one at a time:
python -m cric_sim.cli backtest --data data/ipl_json --types-csv data/players.csv --n-test 150 --ablate
# 5. predict a match after the toss (uses the setting in configs/best.json; fill a squad file first):
python -m cric_sim.cli predict squads/example_ipl.json --types-csv data/players.csv
# 6. rerun the whole experiment (about 2 hours on 4 cores):
for g in features hyperA hyperB pools; do python -m cric_sim.experiments worker $g & done; wait
python -m cric_sim.experiments final
python -m pytest
```

## Results on real data (read this first)
All 6 data files were used for learning (IPL, BBL, PSL, SA20, T20 World Cup, men's T20 internationals; internationals between associate nations are dropped). Every test match is predicted using only matches played before it.

**Settings tried:** 70 settings x 6 "how much to trust the simulator" mixes = 420 recorded results (`reports/experiments/results.csv`, top 30 in `leaderboard.md`). The full grid on paper is 516,096 combinations (512 effect on/off sets x 4 swing sizes x 4 memory lengths x 3 pull-to-average strengths x 3 data pools x 7 pick mixes), so this is a structured sample, not all of them.

**Chosen on IPL 2023-24, tested once on IPL 2025-26 (148 matches)** (`configs/best.json`, `reports/experiments/final_report.md`):

| Method | Avg Dream11 points of the picked XI |
|---|---:|
| Best setting (50% simulator + 50% recent form) | 760.7 |
| Simulator alone | 765.7 |
| Recent-form baseline | 717.1 |
| Random valid team | 610.0 |
| Best possible in hindsight | 1143.1 |

The best setting beat the form baseline by +43.6 points per match (95% range +22.7 to +65.5). The same setting on the validation seasons was +25 to +30, and some settings with the simulator alone were no better than form there. So the edge over a simple form rule is **real but modest and not stable**. Many settings are within noise of each other: the top 20 are only 7 points apart.

What the experiments showed:
- Day-to-day variation in player form is the one effect that clearly matters (it fixed the over-narrow ranges). Ground, pace/spin, toss/chase, game situation, extras and impact players each add little that can be measured with 145-148 test matches. A model with only the player and day-to-day form effects did about as well as the full model.
- Mixing the simulator with recent form beats either alone on the validation seasons.
- Predicted first-innings scores are off by a season-dependent amount: about +0 runs in 2023-24 but +11 runs in 2025-26 (second innings about +1). I tried cancelling the day-to-day-swing drift (`mean_preserving=True`) and it made team picks **worse** in both periods (held-out 751 vs 761 points), so it stays off. The likely cause is that the league scoring level moves from year to year; re-centring the model on recent actual scoring is the next thing to try.
- Win prediction has **no skill** (52% accuracy, no better than the historical rate). Do not use it for betting.
- Fantasy points are noisy, and a 7% edge over a simple rule does not mean you will win contests against smart players. Entry fees take a cut.

## Push for "good" (68%+ of the best possible): what was tried
New validation window: IPL 2022-24 (219 matches). 2025-26 stayed held out and was tested once at the end.

| Idea | Result on 2022-24 validation | Kept? |
|---|---|---|
| Learned correction layer (ridge / boosted trees on simulator outputs + form + history) | Ridge +0.8 pts over the 50/50 mix, boosted trees -21. No real gain. | No |
| Realistic lineups (batting positions and phase-by-phase bowling drawn from real history) | Rank corr 0.194 -> 0.223, XI +7 (sim alone), +4 (mix). Consistent but small. | Yes |
| Slower-fading form inside the mix (decay 0.95 instead of 0.9) | Mix XI 730 -> 736 | Yes |
| Cancel day-to-day-swing score drift | Worse on both periods | No |

**Held-out IPL 2025-26 (148 matches), improved setting:** 50/50 mix 757.3 points (66.3% of best possible), simulator alone 760.9, form rule 717.1. The previous version scored 760.7 on the same matches: the difference is -3.4 (95% range -16 to +9), i.e. **no measurable change**. Rank correlation rose slightly (0.231 -> 0.235).

**Conclusion: the model sits at about 66-67% of the best possible, a steady +40 points over a simple form rule on 2025-26 and about +20-25 on 2022-24. It did not reach "good" (68%+).** Every method tried, including machine-learned corrections, tops out at a rank correlation of about 0.22-0.24 with real points. That looks like the limit of what ball-by-ball history can predict about one match's fantasy points. Bigger gains would need information this data does not have: confirmed batting orders and bowling plans, pitch reports, weather and dew, injuries, and real Dream11 credits.

## How the backtest stays honest
- For each test match, the model is trained **only on matches before that date** (a test checks this).
- Batting order, bowlers, roles and credits are worked out from past matches only.
- It is compared with a **form baseline** (recent average points), a **random valid team**, and the **best possible team in hindsight**.
- It reports ranges (95% bootstrap), calibration of the 10-90% range, and win-probability error, not just averages.
- `--ablate` adds features one step at a time so you can see which ones really help and which ones do nothing.

## Known weaknesses (read these)
- **Cricsheet does not say who the impact sub was**, only that 12 names are listed. The backtest guesses the sub and the replaced player from past matches, so impact results are rough.
- **Credits are estimated** from form, not the real Dream11 credits.
- **Pace/spin is guessed** from when a bowler bowls, because Cricsheet has no bowling style. Fix it with the CSV. Left/right-hand batting is not modelled (also not in Cricsheet).
- Credits in `squads/example_ipl.json` are placeholders (8.5); put the real Dream11 credits in. That example is the last match in the data, so it is not a real prediction.
- Not modelled: dew / time of day, weather, injuries, a bowler's death-over specialism beyond phase effects, exact bowling plans (a simple rotation is used unless you give `bowl_plan`), overthrows, Dream11 rule changes (edit `T20_RULES`).
- `tests/synth.py` makes a **fake league with planted effects**. Tests on it prove the code finds real effects. They say **nothing** about how well it predicts real cricket. Only a backtest on real Cricsheet data does.
- Fantasy points are very noisy. Do not stake money you cannot lose, and check that paid fantasy contests are legal where you live.
