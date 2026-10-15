"""IPL 2025-26 backtest with the best setting (configs/best.json), keeping every player's predicted and
real Dream11 points, to measure how well it finds the best Dream11 players in each match."""
import json, sys, time
sys.path.insert(0, '.')
import pandas as pd
from cric_sim.experiments import POOL_FOLDERS
from cric_sim.loader import load_many
from cric_sim.backtest import BTConfig, prepare, run_backtest
b = json.load(open('configs/best.json'))
folders, only = POOL_FOLDERS[b['pool']]
balls, matches = load_many([f'data/{f}_json' for f in folders], full_members_only=only)
bf, st = prepare(balls, matches)
cfg = BTConfig(features=tuple(b['features']), sigmas=tuple(b['sigmas']), half_life_days=b['half_life_days'],
               prior_scale=b['prior_scale'], empirical_lineup=True, mix_form_decay=b['mix_form_decay'], real_subs=True,
               n_test=400, test_from='2025-01-01', min_history=100, test_league='ipl', sims=1000, refit_every=25,
               random_draws=4, seed=0, label='ipl2526')
col = []
t = time.time(); df = run_backtest(bf, matches, st, cfg, collect=col, progress=True)
pd.DataFrame(col).to_csv('reports/diag/ipl2526_players.csv', index=False)
df.to_csv('reports/diag/ipl2526_matches.csv', index=False)
print('done', round(time.time() - t), len(df))
