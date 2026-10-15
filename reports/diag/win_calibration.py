"""Fix the win prediction: combine the simulator's win chance with pre-match facts (recent chase success in
the league and at the ground). Logistic model trained only on seasons before the one predicted."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from cric_sim.loader import load_matches

_, ipl = load_matches('data/ipl_json')
ipl = ipl[ipl['winner'].isin(ipl['team_a']) | ipl['winner'].isin(ipl['team_b'])].copy()
ipl = ipl[ipl['winner'].notna()].sort_values('date').reset_index(drop=True)
ipl['chase_won'] = (ipl['winner'] == ipl['team_b']).astype(int)

def pre_match_facts(row):
    past = ipl[ipl['date'] < row['date']]
    recent = past.tail(40)['chase_won'].mean() if len(past) else 0.5
    v = past[past['venue_key'] == row['venue']]
    venue = (v['chase_won'].sum() + 0.5 * 10) / (len(v) + 10)  # shrunk toward 50%
    return recent, venue

d = pd.concat([pd.read_csv(f'reports/diag/cap_matches_{l}.csv', dtype={'match_id': str}) for l in sys.argv[1].split(',')], ignore_index=True)
d = d.dropna(subset=['a_won']).copy()
d['date'] = pd.to_datetime(d['date'])
d['season'] = d['date'].dt.year
d['chase_won'] = 1 - d['a_won'].astype(int)
d['p_chase_sim'] = 1 - d['p_a_win']
facts = d.apply(pre_match_facts, axis=1, result_type='expand')
d['recent_chase'], d['venue_chase'] = facts[0], facts[1]
lg = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))
F = lambda x: np.column_stack([lg(x['p_chase_sim']), lg(x['recent_chase']), lg(x['venue_chase'])])
d['p_chase_cal'] = np.nan
for s in sorted(d['season'].unique()):
    tr, te = d[d['season'] < s], d[d['season'] == s]
    if len(tr) < 150:
        continue
    m = LogisticRegression(C=1.0).fit(F(tr), tr['chase_won'])
    d.loc[te.index, 'p_chase_cal'] = m.predict_proba(F(te))[:, 1]
for label, seasons in (("IPL 2022-24 (validation)", [2022, 2023, 2024]), ("IPL 2025-26 (test)", [2025, 2026])):
    t = d[d['season'].isin(seasons) & d['p_chase_cal'].notna()]
    y = t['chase_won'].to_numpy()
    print(f"\n{label}: {len(t)} matches with a result; chasing side won {y.mean():.0%}")
    for name, p in (("simulator as it is", t['p_chase_sim']), ("always pick the chasing team", np.full(len(t), 0.51)),
                    ("league's recent chase record", t['recent_chase']), ("calibrated (sim + recent + ground)", t['p_chase_cal'])):
        p = np.asarray(p, dtype=float)
        acc = np.mean((p > 0.5) == (y == 1)); brier = np.mean((p - y) ** 2)
        print(f"  {name:36s} correct {acc:6.1%}   Brier {brier:.4f}")
d.to_csv('reports/diag/win_calibration_rows.csv', index=False)
