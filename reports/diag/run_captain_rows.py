"""Best setting backtest that also records captain signals (chance of being the match's top scorer, etc.).
usage: run_captain_rows.py LABEL FROM BEFORE SIMS [RECENT_LEAGUE_N]"""
import json, sys, time
sys.path.insert(0, '.')
import pandas as pd
from cric_sim.experiments import POOL_FOLDERS
from cric_sim.loader import load_many
from cric_sim.backtest import BTConfig, prepare, run_backtest, summarize
label, start, before, sims = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rl = int(sys.argv[5]) if len(sys.argv) > 5 else 0
b = json.load(open('configs/best.json'))
folders, only = POOL_FOLDERS[b['pool']]
balls, matches = load_many([f'data/{f}_json' for f in folders], full_members_only=only)
bf, st = prepare(balls, matches)
cfg = BTConfig(features=tuple(b['features']), sigmas=tuple(b['sigmas']), half_life_days=b['half_life_days'],
               prior_scale=b['prior_scale'], empirical_lineup=True, mix_form_decay=b['mix_form_decay'], real_subs=True,
               recent_league_n=rl, n_test=400, test_from=start, test_before=None if before == '-' else before,
               min_history=100, test_league='ipl', sims=sims, refit_every=25, random_draws=4, seed=0, label=label)
col = []
t = time.time(); df = run_backtest(bf, matches, st, cfg, collect=col, progress=True)
pd.DataFrame(col).to_csv(f'reports/diag/cap_players_{label}.csv', index=False)
df.to_csv(f'reports/diag/cap_matches_{label}.csv', index=False)
json.dump(summarize(df), open(f'reports/diag/cap_summary_{label}.json', 'w'), indent=1, default=float)
print('done', round(time.time() - t), len(df), flush=True)
