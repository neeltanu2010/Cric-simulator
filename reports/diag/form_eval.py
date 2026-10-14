"""Try variants of the recent-form estimate (past matches only) and their 50/50 mix with the simulator."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from cric_sim.loader import load_many
from cric_sim.experiments import POOL_FOLDERS
from cric_sim.backtest import prepare
from cric_sim.actual import points_of
from cric_sim.lineup import history_table, infer_roles
from cric_sim.calibrate import xi_points
folders, only = POOL_FOLDERS['leagues']
balls, matches = load_many([f'data/{f}_json' for f in folders], full_members_only=only)
_, st = prepare(balls, matches)
st = st.merge(matches[['match_id', 'league']], on='match_id')
roles = infer_roles(history_table(st[st.date < '2022-01-01']))
st['pts'] = points_of(st, roles)
st = st.sort_values(['name', 'date', 'match_id']).reset_index(drop=True)
role_mean = st[st.date < '2022-01-01'].assign(r=st['name'].map(roles).fillna('BAT')).groupby('r')['pts'].mean().to_dict()
st['prior'] = st['name'].map(roles).fillna('BAT').map(role_mean)

def ewm_form(df, d, k, ipl_only=False):
    """value before each match = (sum d^age * pts + k * prior) / (sum d^age + k), over earlier matches."""
    out = np.full(len(df), np.nan)
    for name, g in df.groupby('name', sort=False):
        idx = g.index.to_numpy(); pts = g['pts'].to_numpy(); use = (g['league'] == 'ipl').to_numpy() if ipl_only else np.ones(len(g), bool)
        S = W = 0.0
        for j, i in enumerate(idx):
            out[i] = (S + k * g['prior'].iloc[j]) / (W + k)
            if use[j]:
                S = d * S + pts[j]; W = d * W + 1.0
    return out

va = pd.read_csv('reports/diag/val_players_baseline.csv', dtype={'match_id': str}); va = va[va.played].copy()
variants = {}
for d in (0.8, 0.9, 0.95, 0.97):
    for k in (1, 2, 5):
        variants[f'd{d}_k{k}'] = ewm_form(st, d, k)
variants['d0.9_k2_iplonly'] = ewm_form(st, 0.9, 2, ipl_only=True)
key = st[['match_id', 'name']].copy(); key['match_id'] = key['match_id'].astype(str)
res = {}
for name, vals in variants.items():
    key[name] = vals
m = va.merge(key, on=['match_id', 'name'], how='left')
print('check: current form vs d0.9_k2 corr', round(m['form'].corr(m['d0.9_k2']), 3))
rows = []
for name in variants:
    m[f'mix_{name}'] = 0.5 * m['pred'] + 0.5 * m[name]
    xa = xi_points(m, name).mean(); xb = xi_points(m, f'mix_{name}').mean()
    rows.append((name, m[name].corr(m.actual), xa, xb))
base_form = xi_points(m, 'form').mean(); m['mix_cur'] = 0.5 * m.pred + 0.5 * m.form; base_mix = xi_points(m, 'mix_cur').mean()
print(f'current form: XI {base_form:.1f}, 50/50 mix {base_mix:.1f}')
for r in sorted(rows, key=lambda x: -x[3]):
    print(f'{r[0]:18s} corr {r[1]:.3f}  form-only XI {r[2]:6.1f}  50/50 mix XI {r[3]:6.1f}')
