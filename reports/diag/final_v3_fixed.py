"""Re-run of final_v3 (same settings) after the loader fix: World Cup matches were loaded twice (wc + t20s folders)."""
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
               random_draws=10, seed=0, label='final_v3_fixed')
t = time.time(); df = run_backtest(bf, matches, st, cfg, progress=True)
df.to_csv('reports/diag/final_v3_fixed_matches.csv', index=False)
s = summarize(df)
open('reports/diag/final_v3_fixed.md', 'w').write(format_report(s, 'Held-out test IPL 2025-26, re-run after fixing double-counted World Cup matches (same settings as final_v3)'))
json.dump(s, open('reports/diag/final_v3_fixed.json', 'w'), indent=1, default=float)
print('done', round(time.time() - t), len(df))
