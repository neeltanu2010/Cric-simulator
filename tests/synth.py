"""Make a fake T20 league in Cricsheet format, with effects planted on purpose, so the code can be
tested without real data. Results from fake data say NOTHING about real cricket; they only show
that the code can find effects that are really there.

Planted effects
- Grounds: "High Ground" scores more; "Spin Ground" helps spinners (more wickets, fewer boundaries);
  "Chase Ground" makes the second innings easier.
- Batters: power hitters, and one batter (T0_P3) who is very weak against spin.
- Bowlers: skill levels, pace or spin.
- Chasing: batters take more risk when the required rate is high.
- New batters are a bit more likely to get out early.
- Wides, no-balls (free hit), leg byes and run outs.
- Impact players (2023+): the batting-first team swaps a batter for a bowler, the chasing team
  swaps a bowler for a batter.
"""
import json
import random
from pathlib import Path

import numpy as np

BASE = np.array([0.36, 0.34, 0.06, 0.005, 0.11, 0.05, 0.035])
RUNS = [0, 1, 2, 3, 4, 6, 0]
VENUES = {
    "High Ground": dict(city="Hiville", score=1.30, spin_help=1.0, chase=1.0),
    "Spin Ground": dict(city="Spinton", score=0.95, spin_help=1.6, chase=1.0),
    "Chase Ground": dict(city="Chaseburg", score=1.0, spin_help=1.0, chase=1.25),
    "Plain Ground": dict(city="Plainfield", score=1.0, spin_help=1.0, chase=1.0),
}
N_TEAMS = 8
POOL = 14


def build_league(seed=0):
    rng = random.Random(seed)
    teams = {}
    for t in range(N_TEAMS):
        players = []
        for k in range(POOL):
            pos = k if k < 11 else rng.randint(4, 9)
            players.append({
                "name": f"T{t}_P{k}", "pos": pos + rng.random() * 0.5,
                "power": 0.7 + 0.9 * (1 - min(pos, 10) / 10) * rng.uniform(0.6, 1.4),
                "skill": rng.uniform(0.75, 1.3), "spin": (k % 2 == 0),
                "bowler": k in (5, 6, 7, 8, 9, 10, 11),
                "keeper": k == 1,
            })
        teams[f"Team{t}"] = players
    teams["Team0"][3]["weak_spin"] = True
    teams["Team0"][3]["power"] = 1.5  # a power hitter, but he struggles against spin
    return teams


def _probs(bat, bowler, venue, inn, new_batter, rrr, wk, free_hit):
    p = BASE.copy()
    mult_bound = bat["power"] * VENUES[venue]["score"]
    if inn == 2:
        mult_bound *= VENUES[venue]["chase"]
    w = bowler["skill"]
    if bowler["spin"]:
        w *= VENUES[venue]["spin_help"]
        mult_bound /= VENUES[venue]["spin_help"] ** 0.5
        if bat.get("weak_spin"):
            mult_bound *= 0.45
            w *= 1.8
    if inn == 2 and rrr is not None:
        risk = float(np.clip((rrr - 8) / 6, -0.3, 0.8))
        mult_bound *= 1 + 0.7 * risk
        w *= 1 + 0.6 * risk
    if new_batter:
        w *= 1.35
    p[4] *= mult_bound
    p[5] *= mult_bound
    p[6] *= w * 1.0 / bowler["skill"] ** 0.0
    if free_hit:
        p[6] = 0.0
    return p / p.sum()


def _innings(rng, bat_team, bowl_team, bat_list, bowl_list, venue, inn, target, bat_sub=None, bowl_swap=None):
    """Returns the innings dict plus the batters / bowlers used."""
    order = list(bat_list)
    striker, ns, nxt, wk = 0, 1, 2, 0
    faced = {p["name"]: 0 for p in order}
    overs, legal_total, score = [], 0, 0
    free_hit = False
    for ov in range(20):
        if wk >= 10 or (target is not None and score >= target):
            break
        bowler = bowl_list[ov % len(bowl_list)]
        dels, lb = [], 0
        while lb < 6 and wk < 10:
            if target is not None and score >= target:
                break
            bat = order[striker]
            rrr = None if target is None else (target - score) / max(120 - legal_total, 1) * 6
            d = {"batter": bat["name"], "bowler": bowler["name"], "non_striker": order[ns]["name"]}
            u = rng.random()
            pw, pn, pb = 0.035, 0.006, 0.02
            if u < pw:
                d["runs"] = {"batter": 0, "extras": 1, "total": 1}
                d["extras"] = {"wides": 1}
                score += 1
                dels.append(d)
                continue
            probs = _probs(bat, bowler, venue, inn, faced[bat["name"]] < 6, rrr, wk, free_hit)
            o = rng.choices(range(7), weights=probs)[0]
            if u < pw + pn:
                if o == 6:
                    o = 0
                d["runs"] = {"batter": RUNS[o], "extras": 1, "total": RUNS[o] + 1}
                d["extras"] = {"noballs": 1}
                faced[bat["name"]] += 1
                score += RUNS[o] + 1
                free_hit = True
                if RUNS[o] % 2:
                    striker, ns = ns, striker
                dels.append(d)
                continue
            free_hit = False
            faced[bat["name"]] += 1
            lb += 1
            legal_total += 1
            if u < pw + pn + pb:
                d["runs"] = {"batter": 0, "extras": 1, "total": 1}
                d["extras"] = {"legbyes": 1}
                score += 1
                striker, ns = ns, striker
                dels.append(d)
                continue
            d["runs"] = {"batter": RUNS[o], "extras": 0, "total": RUNS[o]}
            score += RUNS[o]
            run_out = rng.random() < 0.005 and o != 6
            if o == 6:
                kind = rng.choices(["bowled", "lbw", "caught", "stumped"],
                                   weights=[0.2, 0.1, 0.6, 0.1 if bowler["spin"] else 0.01])[0]
                f = rng.choice(bowl_team)["name"]
                d["wickets"] = [{"player_out": bat["name"], "kind": kind, "fielders": [{"name": f}] if kind in ("caught", "stumped") else []}]
                wk += 1
                if nxt < len(order):
                    striker, nxt = nxt, nxt + 1
            elif run_out:
                out_striker = rng.random() < 0.6
                who = order[striker] if out_striker else order[ns]
                d["wickets"] = [{"player_out": who["name"], "kind": "run out", "fielders": [{"name": rng.choice(bowl_team)["name"]}]}]
                wk += 1
                if nxt < len(order):
                    if out_striker:
                        striker, nxt = nxt, nxt + 1
                    else:
                        ns, nxt = nxt, nxt + 1
            elif RUNS[o] % 2:
                striker, ns = ns, striker
            dels.append(d)
        striker, ns = ns, striker
        overs.append({"over": ov, "deliveries": dels})
    return {"overs": overs}, score


def make_dataset(folder: Path, n_matches=240, seed=1, with_impact=True):
    folder.mkdir(parents=True, exist_ok=True)
    league = build_league(0)
    rng = random.Random(seed)
    names = list(league)
    venues = list(VENUES)
    for m in range(n_matches):
        season = 2022 + m * 3 // n_matches
        date = f"{season}-0{1 + (m % 9)}-{10 + (m % 15)}"
        ta, tb = rng.sample(names, 2)
        venue = venues[m % len(venues)]
        xi = {}
        for t in (ta, tb):
            core = league[t][:11]
            xi[t] = list(core)
            if rng.random() < 0.3:  # a rotation: swap one starter for a reserve
                xi[t][rng.randint(2, 10)] = league[t][rng.randint(11, POOL - 1)]
        toss_winner = rng.choice([ta, tb])
        bat_first_pref = 0.55 if venue != "Chase Ground" else 0.2
        decision = "bat" if rng.random() < bat_first_pref else "field"
        first = toss_winner if decision == "bat" else (tb if toss_winner == ta else ta)
        second = tb if first == ta else ta
        bat1 = sorted(xi[first], key=lambda p: p["pos"])
        bat2 = sorted(xi[second], key=lambda p: p["pos"])
        bowlers = {t: [p for p in sorted(xi[t], key=lambda p: -p["bowler"] * 2 + p["pos"] * 0) if p["bowler"]][:6] for t in (ta, tb)}
        for t in (ta, tb):
            while len(bowlers[t]) < 5:
                bowlers[t].append(next(p for p in xi[t] if p not in bowlers[t]))
        repl = []
        use_imp = with_impact and season >= 2023 and rng.random() < 0.8
        sub_first = sub_second = None
        if use_imp:
            # batting-first team: a bowler comes on for a batter who has batted
            cands = [p for p in league[first] if p not in xi[first] and p["bowler"]]
            if cands:
                sub_first = cands[0]
                out = bat1[-1] if False else bat1[2]
                repl.append({"in": sub_first["name"], "out": out["name"], "team": first, "reason": "impact_player"})
                bowlers[first] = [p for p in bowlers[first] if p["name"] != out["name"]] + [sub_first]
                bowlers[first] = bowlers[first][:6]
            cands = [p for p in league[second] if p not in xi[second] and not p["bowler"]]
            if cands:
                sub_second = cands[0]
                out = next(p for p in reversed(bat2) if p["bowler"])
                repl.append({"in": sub_second["name"], "out": out["name"], "team": second, "reason": "impact_player"})
                bat2 = [sub_second if p is out else p for p in bat2]
        inn1, s1 = _innings(rng, xi[first], xi[second], bat1, bowlers[second], venue, 1, None)
        inn2, s2 = _innings(rng, xi[second], xi[first], bat2, bowlers[first], venue, 2, s1 + 1)
        # fielding sides for the players list
        pl = {}
        for t, sub in ((first, sub_first), (second, sub_second)):
            pl[t] = [p["name"] for p in xi[t]] + ([sub["name"]] if sub else [])
        winner = first if s1 > s2 else (second if s2 > s1 else None)
        info = {"dates": [date], "venue": f"{venue}, {VENUES[venue]['city']}", "city": VENUES[venue]["city"],
                "season": str(season), "teams": [first, second], "players": pl,
                "toss": {"winner": toss_winner, "decision": decision},
                "outcome": {"winner": winner} if winner else {"result": "tie"}}
        if repl:
            info["replacements"] = {"match": repl}
        inn1.update(team=first)
        inn2.update(team=second)
        (folder / f"{1000 + m}.json").write_text(json.dumps({"info": info, "innings": [inn1, inn2]}))
