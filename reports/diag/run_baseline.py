"""Baseline (current best setting) on the wider VALIDATION window IPL 2022-2024. Writes per-player rows.
The 2025-26 test seasons are deliberately NOT touched here."""
import json, sys, time
import pandas as pd
sys.path.insert(0, '.')
from cric_sim.experiments import POOL_FOLDERS
from cric_sim.loader import load_many
from cric_sim.backtest import BTConfig, prepare, run_backtest, summarize, format_report
best = json.load(open('configs/best.json'))
folders, only = POOL_FOLDERS[best['pool']]
balls, matches = load_many([f'data/{f}_json' for f in folders], full_members_only=only)
bf, st = prepare(balls, matches)
cfg = BTConfig(features=tuple(best['features']), sigmas=tuple(best['sigmas']), half_life_days=best['half_life_days'],
               prior_scale=best['prior_scale'], n_test=400, test_from='2022-01-01', test_before='2025-01-01',
               min_history=100, test_league='ipl', sims=500, refit_every=40, random_draws=6, seed=0, label='baseline')
col = []
t = time.time(); df = run_backtest(bf, matches, st, cfg, collect=col, progress=True)
pd.DataFrame(col).to_csv('reports/diag/val_players_baseline.csv', index=False)
df.to_csv('reports/diag/val_matches_baseline.csv', index=False)
s = summarize(df); open('reports/diag/val_baseline.md', 'w').write(format_report(s, 'Baseline on validation window IPL 2022-24'))
json.dump(s, open('reports/diag/val_baseline.json', 'w'), indent=1, default=float)
print('done', round(time.time() - t), 'matches', len(df))
