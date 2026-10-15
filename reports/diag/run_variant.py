"""Best setting + optional simulator improvements. usage: run_variant.py LABEL FROM BEFORE SIMS FLAGS
FLAGS: comma list of keeper (real catch shares + gloveman) and team (same-team lineup history), or none."""
import json, sys, time
sys.path.insert(0, '.')
import pandas as pd
from cric_sim.experiments import POOL_FOLDERS
from cric_sim.loader import load_many
from cric_sim.backtest import BTConfig, prepare, run_backtest
label, start, before, sims, flags = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5].split(',')
b = json.load(open('configs/best.json'))
folders, only = POOL_FOLDERS[b['pool']]
balls, matches = load_many([f'data/{f}_json' for f in folders], full_members_only=only)
bf, st = prepare(balls, matches)
cfg = BTConfig(features=tuple(b['features']), sigmas=tuple(b['sigmas']), half_life_days=b['half_life_days'],
               prior_scale=b['prior_scale'], empirical_lineup=True, mix_form_decay=b['mix_form_decay'], real_subs=True,
               keeper_catches='keeper' in flags, team_lineups='team' in flags, captain_by='simulator',
               n_test=400, test_from=start, test_before=None if before == '-' else before, min_history=100,
               test_league='ipl', sims=sims, refit_every=25, random_draws=2, seed=0, label=label)
col = []
t = time.time(); df = run_backtest(bf, matches, st, cfg, collect=col, progress=True)
pd.DataFrame(col).to_csv(f'reports/diag/cap_players_{label}.csv', index=False)
df.to_csv(f'reports/diag/cap_matches_{label}.csv', index=False)
print('done', round(time.time() - t), len(df), flush=True)
