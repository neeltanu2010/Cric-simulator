"""How close does the BEST of N entered teams get to the best possible team? Each extra team is built for a
different plausible version of the match: every player's value is redrawn from his predicted spread.
Dream11 rules (credit bands, max 10 per team). Team 1 is always our normal pick."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from cric_sim.lineup import credits_from_rank
from cric_sim.optimizer import pick_team
from cric_sim.simulator import Player
f95 = pd.read_csv('reports/diag/form_d095_k2.csv', dtype={'match_id': str})
label, spread = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
p = pd.read_csv(f'reports/diag/cap_players_{label}.csv', dtype={'match_id': str}).merge(f95, on=['match_id', 'name'], how='left')
p['f95'] = p['f95'].fillna(p['form']); p['value'] = 0.5 * p['pred'] + 0.5 * p['f95']
rng = np.random.default_rng(0)
NS = (1, 3, 6, 11, 20)
rows = []
for mid, g in p[p['played']].groupby('match_id'):
    g = g.reset_index(drop=True)
    if len(g) < 18: continue
    pct = (g['form'].rank(method='first').to_numpy() - 1) / (len(g) - 1)
    pl = [Player(n, r, float(c), t) for n, r, c, t in zip(g['name'], g['role'], credits_from_rank(pct), g['team'])]
    act = g['actual'].to_numpy()
    sc = lambda t: sum(act[pl.index(q)] for q in t['team']) + act[pl.index(t['captain'])] + 0.5 * act[pl.index(t['vice_captain'])]
    best = sc(pick_team(pl, act[None, :]))
    teams, seen = [], set()
    t0 = pick_team(pl, g['value'].to_numpy()[None, :], captain_value=g['pred'].to_numpy())
    teams.append(sc(t0)); seen.add((frozenset(q.name for q in t0['team']), t0['captain'].name, t0['vice_captain'].name))
    tries = 0
    while len(teams) < max(NS) and tries < 80:
        tries += 1
        z = rng.standard_normal(len(g))
        draw = np.maximum(g['value'].to_numpy() + spread * z * g['pred_std'].to_numpy(), 0)  # one plausible version of the match
        cap = np.maximum(g['pred'].to_numpy() + spread * z * g['pred_std'].to_numpy(), 0)
        t = pick_team(pl, draw[None, :], captain_value=cap)
        key = (frozenset(q.name for q in t['team']), t['captain'].name, t['vice_captain'].name)
        if key in seen: continue
        seen.add(key); teams.append(sc(t))
    rows.append([best] + [max(teams[:n]) for n in NS] + [np.mean(teams[:n]) for n in NS])
a = np.array(rows)
print(f"{label}: {len(a)} matches, spread x{spread}; best possible team averages {a[:,0].mean():.0f} points")
print(f"{'teams entered':>14s}{'best of them':>14s}{'% of best possible':>20s}{'average team':>14s}")
for i, n in enumerate(NS):
    print(f"{n:>14d}{a[:,1+i].mean():>14.0f}{a[:,1+i].mean()/a[:,0].mean():>20.1%}{a[:,1+len(NS)+i].mean():>14.0f}")
