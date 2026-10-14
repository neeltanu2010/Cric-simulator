# Cric-simulator

Ball-by-ball T20 simulator that predicts **Dream11 fantasy points for every player** and picks the best XI, captain and vice-captain.

## How it works
1. `loader.py` reads Cricsheet ball-by-ball JSON (IPL / T20).
2. `model.py` learns each batter's and bowler's outcome chances per ball (dot, 1, 2, 3, 4, 6, wicket) in 3 phases (powerplay, middle, death). Players with little data are pulled toward the league average. Recent matches count more.
3. `simulator.py` plays the match thousands of times, ball by ball, for both innings.
4. `dream11.py` turns each simulated match into Dream11 points.
5. `optimizer.py` picks the best XI under 100 credits, role limits and the 7-per-team limit.

## Use
```
pip install -r requirements.txt
# put Cricsheet "IPL (JSON)" files in data/ipl_json/
python -m cric_sim.cli squads/example.json --data data/ipl_json --sims 5000
python -m pytest
```
Fill a squad file (see `squads/example.json`) after the toss, with names spelled as in Cricsheet (for example `V Kohli`) and the credits shown in the Dream11 app. `--risk 1` favours high-ceiling players for big grand leagues.

## Limits (v1)
- Check `T20_RULES` in `dream11.py` against the current Dream11 rules.
- Not modelled yet: pitch and venue effects, bowler type (pace vs spin), toss, dew, extras, impact players, batting-order changes, exact bowling plans.
- Fantasy points are noisy. **Backtest before trusting it, and do not stake money you cannot lose.**
- Check that paid fantasy contests are legal where you live.
