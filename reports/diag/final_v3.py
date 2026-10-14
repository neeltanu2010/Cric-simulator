"""ONE-SHOT held-out test of the improved setting on IPL 2025-26. Settings fixed beforehand from 2022-24."""
import json, sys, time
sys.path.insert(0, '.')
import pandas as pd
from cric_sim.experiments import POOL_FOLDERS
from cric_sim.loader import load_many
from cric_sim.backtest import BTConfig, prepare, run_backtest, summarize, format_report
b = json.load(open('configs/best.json'))
folders, only = POOL_FOLDERS[b['pool']]
balls, matches = load_many([f'data/{f}_json' for f in folders], full_members_only=only)
bf, st = prepare(balls, matches)
cfg = BTConfig(features=tuple(b['features']), sigmas=tuple(b['sigmas']), half_life_days=b['half_life_days'],
               prior_scale=b['prior_scale'], empirical_lineup=True, mix_form_decay=0.95, real_subs=True,
               n_test=148, test_from='2025-01-01', min_history=100, test_league='ipl', sims=1000, refit_every=25,
               random_draws=10, seed=0, label='final_v3')
t = time.time(); df = run_backtest(bf, matches, st, cfg, progress=True)
df.to_csv('reports/diag/final_v3_matches.csv', index=False)
s = summarize(df)
open('reports/diag/final_v3.md', 'w').write(format_report(s, 'Held-out test IPL 2025-26: confirmed XI and subs + realistic lineups + slower form, 50% mix'))
json.dump(s, open('reports/diag/final_v3.json', 'w'), indent=1, default=float)
print('done', round(time.time() - t), len(df))
