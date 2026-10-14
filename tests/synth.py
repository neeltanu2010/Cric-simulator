"""Make fake Cricsheet-format matches so the code can be tested without real data.
Results from fake data mean nothing; this only checks that the code works."""
import json
import random
from pathlib import Path

import numpy as np

# dot, 1, 2, 3, 4, 6, wicket  (roughly IPL average)
BASE = np.array([0.36, 0.34, 0.06, 0.005, 0.11, 0.05, 0.045])
BASE = BASE / BASE.sum()
RUNS = [0, 1, 2, 3, 4, 6, 0]


def team_names(prefix):
    return [f"{prefix}{i}" for i in range(1, 12)]


def make_match(path: Path, rng: random.Random, teams=("A", "B"), date="2024-04-01", power=None):
    power = power or {}
    squads = {t: team_names(t) for t in teams}
    innings = []
    for bat, bowl in ((teams[0], teams[1]), (teams[1], teams[0])):
        overs, order, striker, ns, nxt, wk = [], squads[bat], 0, 1, 2, 0
        for ov in range(20):
            if wk >= 10:
                break
            bowler = squads[bowl][6 + ov % 5]
            dels = []
            for _ in range(6):
                if wk >= 10:
                    break
                p = BASE.copy()
                p[1:6] *= power.get(order[striker], 1.0)
                p /= p.sum()
                o = rng.choices(range(7), weights=p)[0]
                d = {"batter": order[striker], "bowler": bowler, "non_striker": order[ns],
                     "runs": {"batter": RUNS[o], "extras": 0, "total": RUNS[o]}}
                if o == 6:
                    d["wickets"] = [{"player_out": order[striker], "kind": "caught",
                                     "fielders": [{"name": squads[bowl][0]}]}]
                    wk += 1
                    if nxt < 11:
                        striker, nxt = nxt, nxt + 1
                elif RUNS[o] % 2:
                    striker, ns = ns, striker
                dels.append(d)
            striker, ns = ns, striker
            overs.append({"over": ov, "deliveries": dels})
        innings.append({"team": bat, "overs": overs})
    info = {"dates": [date], "venue": "Test Ground", "players": squads}
    path.write_text(json.dumps({"info": info, "innings": innings}))


def make_dataset(folder: Path, n=60, seed=1, power=None):
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for i in range(n):
        make_match(folder / f"{1000 + i}.json", rng, date=f"2024-0{1 + i % 9}-15", power=power)
