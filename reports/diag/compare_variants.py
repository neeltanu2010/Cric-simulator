"""Compare backtest variants match by match under Dream11 rules (credit bands, max 10 per team,
50/50 value for the XI, captain by simulator average)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from cric_sim.lineup import credits_from_rank
from cric_sim.optimizer import pick_team
from cric_sim.simulator import Player
f95 = pd.read_csv('reports/diag/form_d095_k2.csv', dtype={'match_id': str})
def score_label(label):
    p = pd.read_csv(f'reports/diag/cap_players_{label}.csv', dtype={'match_id': str}).merge(f95, on=['match_id', 'name'], how='left')
    p['f95'] = p['f95'].fillna(p['form']); p['value'] = 0.5 * p['pred'] + 0.5 * p['f95']
    out, bias = {}, p[p.played].groupby('role').apply(lambda g: g.pred.mean() - g.actual.mean(), include_groups=False)
    corr = p[p.played][['pred', 'actual']].corr().iloc[0, 1]
    for mid, g in p[p['played']].groupby('match_id'):
        g = g.reset_index(drop=True)
        if len(g) < 18: continue
        pct = (g['form'].rank(method='first').to_numpy() - 1) / (len(g) - 1)
        pl = [Player(n, r, float(c), t) for n, r, c, t in zip(g['name'], g['role'], credits_from_rank(pct), g['team'])]
        act = g['actual'].to_numpy()
        t = pick_team(pl, g['value'].to_numpy()[None, :], captain_value=g['pred'].to_numpy())
        top = pl[act.argmax()]
        out[mid] = (sum(act[pl.index(q)] for q in t['team']) + act[pl.index(t['captain'])] + 0.5 * act[pl.index(t['vice_captain'])],
                    top in t['team'], t['captain'] is top)
    return pd.DataFrame(out, index=['pts', 'top_in_xi', 'c_top']).T, bias, corr
base, bias0, corr0 = score_label(sys.argv[1])
print(f"{'variant':24s}{'team pts':>9s}{'vs base (95%)':>22s}{'better':>8s}{'worse':>7s}{'top1 in XI':>11s}{'C=top':>7s}{'sim corr':>9s}{'WK bias':>8s}{'BOWL bias':>10s}")
rng = np.random.default_rng(0)
for label in sys.argv[1:]:
    r, bias, corr = (base, bias0, corr0) if label == sys.argv[1] else score_label(label)
    d = (r['pts'] - base['pts']).dropna().to_numpy().astype(float)
    bs = [rng.choice(d, len(d)).mean() for _ in range(3000)]
    ci = "" if label == sys.argv[1] else f"{d.mean():+.1f} ({np.percentile(bs,2.5):+.1f} to {np.percentile(bs,97.5):+.1f})"
    print(f"{label:24s}{r['pts'].mean():9.1f}{ci:>22s}{np.mean(d>0):8.0%}{np.mean(d<0):7.0%}{r['top_in_xi'].mean():11.0%}{r['c_top'].mean():7.1%}{corr:9.3f}{bias.get('WK', np.nan):+8.1f}{bias.get('BOWL', np.nan):+10.1f}")
