"""Walk-forward test of the learned correction layer on IPL 2022-24 (trained only on earlier seasons)."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from cric_sim.calibrate import walk_forward, xi_points
tr = pd.read_csv('reports/diag/train_players_2018_21.csv'); va = pd.read_csv('reports/diag/val_players_baseline.csv')
rows = pd.concat([tr, va], ignore_index=True)
rows['date'] = pd.to_datetime(rows['date'])
for kind in ('ridge', 'gbm'):
    rows[f'stack_{kind}'] = walk_forward(rows, kind)
v = rows[(rows.date >= '2022-01-01') & rows.played].copy()
v['blend50'] = 0.5 * v.pred + 0.5 * v.form
for k in ('ridge', 'gbm'):
    v[f'stack_{k}_form'] = 0.5 * v[f'stack_{k}'] + 0.5 * v.form
cols = ['pred', 'form', 'blend50', 'stack_ridge', 'stack_gbm', 'stack_ridge_form', 'stack_gbm_form']
print('rank/linear corr with real points (2022-24):')
for c in cols: print(f'  {c:18s} corr {v[c].corr(v.actual):.3f}  MAE {(v[c]-v.actual).abs().mean():.2f}')
res = pd.DataFrame({c: xi_points(v, c) for c in cols})
print('\navg XI points per match (n=%d):' % len(res))
rng = np.random.default_rng(0)
for c in cols:
    d = (res[c] - res['form']).to_numpy()
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    print(f'  {c:18s} {res[c].mean():6.1f}   vs form {d.mean():+6.1f} (95% {np.percentile(bs,2.5):+.1f} to {np.percentile(bs,97.5):+.1f})')
res.to_csv('reports/diag/stack_eval_xi.csv')
