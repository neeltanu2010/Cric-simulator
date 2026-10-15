"""Offline test: role offsets (points added per role) and a different simulator/form mix per role group.
Chosen on IPL 2018-21, checked on IPL 2022-24, all under Dream11 rules (credit bands, max 10 per team)."""
import sys, itertools; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from cric_sim.lineup import credits_from_rank
from cric_sim.optimizer import pick_team
from cric_sim.simulator import Player
f95 = pd.read_csv('reports/diag/form_d095_k2.csv', dtype={'match_id': str})
def prep(label):
    p = pd.read_csv(f'reports/diag/cap_players_{label}.csv', dtype={'match_id': str}).merge(f95, on=['match_id', 'name'], how='left')
    p['f95'] = p['f95'].fillna(p['form'])
    games = []
    for mid, g in p[p['played']].groupby('match_id'):
        g = g.reset_index(drop=True)
        if len(g) < 18: continue
        pct = (g['form'].rank(method='first').to_numpy() - 1) / (len(g) - 1)
        pl = [Player(n, r, float(c), t) for n, r, c, t in zip(g['name'], g['role'], credits_from_rank(pct), g['team'])]
        games.append((pl, g['pred'].to_numpy(), g['f95'].to_numpy(), g['role'].to_numpy(), g['actual'].to_numpy()))
    return games
def evaluate(games, off, w_bat, w_bowl):
    tot = []
    for pl, pred, form, role, act in games:
        w = np.where(np.isin(role, ['BAT', 'WK']), w_bat, w_bowl)
        val = w * pred + (1 - w) * form + np.array([off.get(r, 0.0) for r in role])
        t = pick_team(pl, val[None, :], captain_value=pred)
        tot.append(sum(act[pl.index(q)] for q in t['team']) + act[pl.index(t['captain'])] + 0.5 * act[pl.index(t['vice_captain'])])
    return np.array(tot)
train, val = prep('train1821'), prep('val2224')
grid = []
for ob, oa, ow, wb, wbo in itertools.product([-4, 0, 4, 8], [-4, 0, 4], [0, 4], [0.5], [0.5]):
    grid.append(({'BOWL': ob, 'AR': oa, 'WK': ow}, wb, wbo))
for wb, wbo in itertools.product([0.3, 0.5, 0.7], [0.3, 0.5, 0.7]):
    grid.append(({}, wb, wbo))
res = []
for off, wb, wbo in grid:
    res.append((evaluate(train, off, wb, wbo).mean(), off, wb, wbo))
res.sort(key=lambda r: -r[0])
base_train = evaluate(train, {}, 0.5, 0.5).mean()
print(f"2018-21 baseline (no offsets, 50/50 for all): {base_train:.1f}")
print("best 5 settings on 2018-21:")
for r in res[:5]:
    print(f"  {r[0]:.1f}  offsets {r[1]}  sim share bat/wk {r[2]}  bowl/ar {r[3]}")
best = res[0]
b0, b1 = evaluate(val, {}, 0.5, 0.5), evaluate(val, best[1], best[2], best[3])
d = b1 - b0; rng = np.random.default_rng(0); bs = [rng.choice(d, len(d)).mean() for _ in range(3000)]
print(f"\nCheck on 2022-24: baseline {b0.mean():.1f}, chosen setting {b1.mean():.1f}, difference {d.mean():+.1f} (95% {np.percentile(bs,2.5):+.1f} to {np.percentile(bs,97.5):+.1f})")
