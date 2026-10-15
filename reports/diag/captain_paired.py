"""Paired comparison of two captain rules (same XI): per-match differences with a 95% bootstrap range."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from cric_sim.optimizer import pick_team
from cric_sim.simulator import Player
f95 = pd.read_csv('reports/diag/form_d095_k2.csv', dtype={'match_id': str})
p = pd.read_csv(f'reports/diag/cap_players_{sys.argv[1]}.csv', dtype={'match_id': str}).merge(f95, on=['match_id', 'name'], how='left')
p['f95'] = p['f95'].fillna(p['form']); p['value'] = 0.5 * p['pred'] + 0.5 * p['f95']
res = []
for mid, g in p.groupby('match_id'):
    g = g[g['played']].reset_index(drop=True)
    if len(g) < 18:
        continue
    pl = [Player(n, r, c, t) for n, r, c, t in zip(g['name'], g['role'], g['credits'], g['team'])]
    act = g['actual'].to_numpy()
    cur = pick_team(pl, g['value'].to_numpy()[None, :])
    new = pick_team(pl, g['value'].to_numpy()[None, :], captain_value=g['pred'].to_numpy())  # captain by simulator average
    sc = lambda r: sum(act[pl.index(q)] for q in r['team']) + act[pl.index(r['captain'])] + 0.5 * act[pl.index(r['vice_captain'])]
    top = act.argmax(); top3 = set(np.argsort(-act)[:3])
    res.append((sc(cur), sc(new), pl.index(cur['captain']) == top, pl.index(new['captain']) == top,
                pl.index(cur['captain']) in top3, pl.index(new['captain']) in top3, act[pl.index(cur['captain'])], act[pl.index(new['captain'])]))
a = np.array(res, dtype=float); rng = np.random.default_rng(0)
def ci(x):
    b = [rng.choice(x, len(x)).mean() for _ in range(3000)]; return np.percentile(b, [2.5, 97.5])
d = a[:, 1] - a[:, 0]; lo, hi = ci(d)
print(f"{sys.argv[1]}: {len(a)} matches (captain chosen inside the MILP: captain value = simulator average)")
print(f"  team points: current {a[:,0].mean():.1f}  new {a[:,1].mean():.1f}  difference {d.mean():+.1f} (95% {lo:+.1f} to {hi:+.1f}); new better in {np.mean(d>0):.0%}, worse in {np.mean(d<0):.0%}, same in {np.mean(d==0):.0%}")
print(f"  captain = real top scorer: {a[:,2].mean():.1%} -> {a[:,3].mean():.1%} | captain in real top 3: {a[:,4].mean():.1%} -> {a[:,5].mean():.1%} | captain's points: {a[:,6].mean():.1f} -> {a[:,7].mean():.1f}")
