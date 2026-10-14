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
# 5. predict a match after the toss (fill squads/example.json first):
python -m cric_sim.cli predict squads/example.json --data data/ipl_json --types-csv data/players.csv
python -m pytest
```

## How the backtest stays honest
- For each test match, the model is trained **only on matches before that date** (a test checks this).
- Batting order, bowlers, roles and credits are worked out from past matches only.
- It is compared with a **form baseline** (recent average points), a **random valid team**, and the **best possible team in hindsight**.
- It reports ranges (95% bootstrap), calibration of the 10-90% range, and win-probability error, not just averages.
- `--ablate` adds features one step at a time so you can see which ones really help and which ones do nothing.

## Known weaknesses (read these)
- **Impact player identity is taken from the real match** in the backtest, so that feature looks better there than it will in real use.
- **Credits are estimated** from form, not the real Dream11 credits.
- **Pace/spin is guessed** from when a bowler bowls, because Cricsheet has no bowling style. Fix it with the CSV. Left/right-hand batting is not modelled (also not in Cricsheet).
- Not modelled: dew / time of day, weather, injuries, a bowler's death-over specialism beyond phase effects, exact bowling plans (a simple rotation is used unless you give `bowl_plan`), overthrows, Dream11 rule changes (edit `T20_RULES`).
- `tests/synth.py` makes a **fake league with planted effects**. Tests on it prove the code finds real effects. They say **nothing** about how well it predicts real cricket. Only a backtest on real Cricsheet data does.
- Fantasy points are very noisy. Do not stake money you cannot lose, and check that paid fantasy contests are legal where you live.
