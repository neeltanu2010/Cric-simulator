"""Re-score the saved backtests under Dream11's actual rules: credit bands (premium 9.5-11, mid 8.5-9,
budget 7-8) and up to 10 players from one team, versus the old estimate (7 + 4 x rank^2) and max 7.
Teams are re-picked from each player's saved predicted points; nothing is re-simulated."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from cric_sim.lineup import credits_from_rank
from cric_sim.optimizer import pick_team
from cric_sim.simulator import Player
f95 = pd.read_csv('reports/diag/form_d095_k2.csv', dtype={'match_id': str})
old_credits = lambda pct: np.round((7 + 4 * pct ** 2) * 2) / 2
CONFIGS = [("old rules: old credit estimate, max 7 per team", old_credits, 7),
           ("max 10 per team only", old_credits, 10),
           ("new Dream11 credit bands only", credits_from_rank, 7),
           ("NEW RULES: credit bands + max 10 per team", credits_from_rank, 10)]
rng = np.random.default_rng(0)
for label in sys.argv[1:]:
    p = pd.read_csv(f'reports/diag/cap_players_{label}.csv', dtype={'match_id': str}).merge(f95, on=['match_id', 'name'], how='left')
    p['f95'] = p['f95'].fillna(p['form']); p['value'] = 0.5 * p['pred'] + 0.5 * p['f95']
    print(f"\n=== {label} ===")
    print(f"{'rules':46s}{'ours':>7s}{'form':>7s}{'random':>8s}{'best':>7s}{'% best':>8s}{'beat form':>10s}{'beat rand':>10s}{'top1 in XI':>11s}{'C=top':>7s}{'cr used':>8s}")
    for name, credit_fn, cap in CONFIGS:
        res = []
        for mid, g in p.groupby('match_id'):
            g = g[g['played']].reset_index(drop=True)
            if len(g) < 18:
                continue
            pct = g['form'].rank(method='first').to_numpy() - 1
            pct = pct / max(len(g) - 1, 1)
            cr = credit_fn(pct)
            pl = [Player(n, r, float(c), t) for n, r, c, t in zip(g['name'], g['role'], cr, g['team'])]
            act = g['actual'].to_numpy()
            score = lambda t: sum(act[pl.index(q)] for q in t['team']) + act[pl.index(t['captain'])] + 0.5 * act[pl.index(t['vice_captain'])]
            ours = pick_team(pl, g['value'].to_numpy()[None, :], max_per_team=cap, captain_value=g['pred'].to_numpy())
            form = pick_team(pl, g['form'].to_numpy()[None, :], max_per_team=cap)
            best = pick_team(pl, act[None, :], max_per_team=cap)
            rand = np.mean([score(pick_team(pl, rng.random((1, len(pl))), max_per_team=cap)) for _ in range(8)])
            top1 = pl[act.argmax()]
            res.append((score(ours), score(form), rand, score(best), top1 in ours['team'], ours['captain'] is top1, ours['credits']))
        a = np.array(res, dtype=float)
        print(f"{name:46s}{a[:,0].mean():7.1f}{a[:,1].mean():7.1f}{a[:,2].mean():8.1f}{a[:,3].mean():7.1f}{a[:,0].mean()/a[:,3].mean():8.1%}"
              f"{np.mean(a[:,0] > a[:,1]):10.0%}{np.mean(a[:,0] > a[:,2]):10.0%}{a[:,4].mean():11.0%}{a[:,5].mean():7.1%}{a[:,6].mean():8.1f}")
