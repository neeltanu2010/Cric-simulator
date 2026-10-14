"""Validation IPL 2022-24: real XI/subs, plus day/night and/or recent-pitch adjustment. Writes per-player rows."""
import json, sys, time
sys.path.insert(0, '.')
import pandas as pd
from cric_sim.experiments import POOL_FOLDERS
from cric_sim.loader import load_many
from cric_sim.backtest import BTConfig, prepare, run_backtest, summarize, format_report
tag = sys.argv[1]
b = json.load(open('configs/best.json'))
feats = tuple(b['features']) + (('daynight',) if tag in ('B', 'D') else ())
folders, only = POOL_FOLDERS[b['pool']]
balls, matches = load_many([f'data/{f}_json' for f in folders], full_members_only=only)
bf, st = prepare(balls, matches)
cfg = BTConfig(features=feats, sigmas=tuple(b['sigmas']), half_life_days=b['half_life_days'], prior_scale=b['prior_scale'],
               empirical_lineup=True, mix_form_decay=0.95, real_subs=True, recent_pitch=tag in ('C', 'D'),
               n_test=400, test_from='2022-01-01', test_before='2025-01-01', min_history=100, test_league='ipl',
               sims=500, refit_every=40, random_draws=6, seed=0, label=tag)
col = []
t = time.time(); df = run_backtest(bf, matches, st, cfg, collect=col, progress=True)
pd.DataFrame(col).to_csv(f'reports/diag/val_players_{tag}.csv', index=False)
df.to_csv(f'reports/diag/val_matches_{tag}.csv', index=False)
s = summarize(df)
open(f'reports/diag/val_{tag}.md', 'w').write(format_report(s, f'Validation IPL 2022-24, variant {tag}'))
json.dump(s, open(f'reports/diag/val_{tag}.json', 'w'), indent=1, default=float)
print('done', round(time.time() - t), len(df))
