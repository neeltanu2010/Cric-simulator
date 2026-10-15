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
| Team pick | Best XI under Dream11 rules: 100 credits, WK 1-4 / BAT 3-6 / AR 1-4 / BOWL 3-6, max 10 from one team, captain (2x) and vice (1.5x). |

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

## Confirmed lineups and pitch reports
**What could be obtained.** Searches found no public archive of past pitch reports (they exist only as text on ESPNcricinfo / Cricbuzz pages, which are blocked from this environment and whose terms forbid bulk scraping) and no IPL weather/dew dataset. Historical weather is available free from Open-Meteo, but `archive-api.open-meteo.com` is blocked by this environment's network policy; it can be added under Allowed domains in the environment settings.

**What was found in our own data:** Cricsheet records every impact-player substitution on the delivery where it happened (557 subs, all 291 IPL matches with 12 listed players, 2023-26). The loader now reads them, so the backtest uses the exact starting XI (announced at the toss) and the real sub. Who the sub replaced is still guessed, because that is decided during the match.

**Data stand-ins for conditions:** afternoon vs evening game (IPL double-headers, from the fixture list; dew is an evening effect) and a "recent pitch" adjustment (how the ground played in the last 60 days vs the model's expectation).

| Change (IPL 2022-24 validation, 50/50 mix) | XI points vs real XI/subs | Kept? |
|---|---|---|
| Real XI and subs instead of guessing | +0.3 (no change); rank corr 0.223 -> 0.230 | Yes (correct data) |
| + afternoon/evening effect | +1.5 (-2.5 to +5.5) | Off by default (`daynight` effect) |
| + recent pitch | +0.4 (-5.1 to +5.7), first-innings bias worse | Off by default (`--recent-pitch`) |
| + both | -2.3 (-8.3 to +4.0) | No |

A ground's recent scoring surprise barely predicts its next match (correlation +0.04 to +0.11), so data-made "pitch reports" carry little information. **Held-out IPL 2025-26 with confirmed lineups:** 755.9 points (66.1% of best possible), +39 over the form rule; vs the guessed-sub version: -1.4 (95% range -11 to +8), i.e. no measurable change.

**Using them for a real match (after the toss):**
```
# paste the announced XIs as written in the app; team A bats first; add the likely impact subs and real credits
python -m cric_sim.cli squad --team-a GT --xi-a "Shubman Gill, Sai Sudharsan, ..." --team-b RCB --xi-b "Virat Kohli, ..." \
    --venue "Narendra Modi Stadium" --sub-b "Rasikh Salam" --credits-a "10,9,..." --credits-b "..." \
    --pitch-scoring high --pitch-spin some --out squads/today.json
python -m cric_sim.cli predict squads/today.json            # add --recent-pitch to also use the last 60 days at the ground
```
The pitch report you type (`--pitch-scoring very high/high/normal/low/very low`, `--pitch-spin` / `--pitch-pace big/some/normal/little/none`, or numbers) moves the simulation, e.g. "high" scoring raised a first-innings estimate from 202 to 211. **This input is your judgement and could not be tested on past matches**, because no archive of pitch reports exists.

## Adding recent matches that only exist as scorecards
Cricsheet adds ball-by-ball files a day or more after a match (and is blocked from the cloud environment). Recent matches can be added from their published scorecards as a small JSON file (see `data/manual/ind_wi_2026_t20i_2to4.json`: batting rows in batting order, bowling rows, catches, bowled/lbw counts; unknown fours/sixes and dot balls are estimated). Pass it to any command with `--extra-scorecards`:
```
python -m cric_sim.cli predict squads/match.json --league t20s --extra-scorecards data/manual/ind_wi_2026_t20i_2to4.json
```
Scorecard matches update recent form, batting positions, roles and estimated credits. They do **not** update the ball-by-ball ratings; when Cricsheet publishes the matches, add the real files instead.

## Captain choice and other weak points (reverse brainstorm, then tested)
"How could we pick the worst captain?" -> steady players with no upside, players who barely bat or bowl, ignoring who
is likely to win, C and VC from the same team, stale or one-off form, judging by averages. Each was turned into a rule
and tested on IPL 2022-24 (training on 2018 onward), then checked once on IPL 2025-26.

| Change | 2022-24 | 2025-26 | Kept? |
|---|---|---|---|
| **Captain/VC chosen by the simulator's average** (team still picked by the 50/50 value) | team +1.6 pts/match; captain in real top 3 24% -> 30% | team +5.0 (95% -1.5 to +11.5); captain = top scorer 11.5% -> 14.2% | **Yes** |
| Captain by chance of being top scorer / top 3 / 100+ points (from the simulations) | slightly worse | - | No |
| Learned captain models (logistic, boosted trees) | no gain / worse | - | No |
| Captain and vice-captain from different teams | no gain | - | No |
| Win prediction calibrated with recent and ground chase records | 51% -> 49% correct | 49% -> 55% (chasers won 58%) | No: not stable |
| League scoring-level adjustment (last 20 matches) | bias +6.7 -> +10.8, team points lower | - | No |
| Season-to-season correction of reported scores | no change in error | no change in error | No |

Even the best possible captain/vice-captain *inside our XI* would add ~100 points a match: captaincy matters a lot,
but the favourite is the top scorer only ~10-15% of the time, so most of it is luck. `predict` now lists the captain
candidates with their simulated chance of finishing as top scorer and in the top 3. Match winners stay close to a
coin flip with this data; do not use the win chances for betting.

## Dream11 credit rules
Dream11 gives 100 credits for 11 players (average 9.09), allows up to 10 players from one real team, and sets credits per
match in bands: premium 9.5-11.0 (marquee players, top-order batters, elite all-rounders), mid-tier 8.5-9.0, budget 7.0-8.0.
The optimizer uses these rules. Past credits are not public, so the backtest estimates them from each player's recent-form
rank inside the match using those bands (about 6 premium, 7 mid, 9 budget; average 8.6). Type the real credits in when you have them.

Re-scored with these rules (same predictions, teams re-picked):

| | 2022-24 | 2025-26 |
|---|---:|---:|
| Our team (avg points, C 2x, VC 1.5x) | 732.1 | 753.2 |
| Recent-form rule | 686.6 | 691.2 |
| Random valid team | 626.1 | 613.2 |
| Best possible team | 1121.1 | 1142.5 |
| Our team beats the form rule | 63% of matches | 68% |
| Our team beats a random team | 79% | 82% |

With realistic credit bands the budget bites harder; our team loses a few points but the form rule loses much more, so the
gap grows (+45 and +62 points a match, from +30 and +39). The 10-per-team limit changes almost nothing.

## Getting closer to the best possible team
Ideas tested (chosen on IPL 2022-24 with 2018-21 for training, checked on 2025-26, Dream11 rules):

| Idea | Result | Kept? |
|---|---|---|
| Keeper takes his real share of catches (15.7%, not 9.1%); bowler caught-and-bowled 4.2% | keeper bias -3.8 -> -2.1 pts, team -4.5 (within noise) | Option only |
| Batting order / bowling usage from recent matches for the same team | positions better predicted (error 0.80 -> 0.72), team -4.4 (within noise) | Option only |
| Both together | team -14.4 (95% -24 to -5) | No |
| Different simulator/form mix per role | best on 2018-21, -7.4 on 2022-24 | No |
| Extra points per role (e.g. bowlers, keepers) | no consistent gain | No |
| **Several different teams** (`predict --teams N`) | best of N gets much closer (below) | **Yes** |

**Several teams.** Team 1 is the normal pick; each extra team is built for a different plausible version of the match
(each player's value redrawn from his predicted spread, spread 0.3). IPL 2025-26 (148 matches):

| Teams entered | Best of them | % of best possible | Average team |
|---:|---:|---:|---:|
| 1 | 753 | 65.9% | 753 |
| 3 | 821 | 71.8% | 728 |
| 6 | 854 | 74.7% | 717 |
| 11 | 879 | 76.9% | 709 |
| 20 | 902 | 78.9% | 705 |

The best team in the set gets much closer to the best possible team, but the average team is lower and every team
costs an entry fee. A single team stays at ~66%: the rest of the gap is the randomness of one match.

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
