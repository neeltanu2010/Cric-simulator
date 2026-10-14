"""Does cancelling the swing-induced score drift help on real data? Best recorded setting, fix off vs on."""
import json, sys, time
import pandas as pd
sys.path.insert(0, '.')
from cric_sim.experiments import POOL_FOLDERS, VAL
from cric_sim.loader import load_many
from cric_sim.backtest import BTConfig, prepare, run_backtest, summarize
which = sys.argv[1]  # validation | test
best = json.load(open('configs/best.json'))
folders, only = POOL_FOLDERS[best['pool']]
balls, matches = load_many([f'data/{f}_json' for f in folders], full_members_only=only)
bf, st = prepare(balls, matches)
kw = dict(VAL) if which == 'validation' else dict(VAL, n_test=148, test_before=None, sims=1000, refit_every=25, random_draws=10)
out = {}
cache = {}
for fix in (False, True):
    cfg = BTConfig(features=tuple(best['features']), sigmas=tuple(best['sigmas']), half_life_days=best['half_life_days'],
                   prior_scale=best['prior_scale'], mean_preserving=fix, **kw)
    t = time.time(); df = run_backtest(bf, matches, st, cfg, fit_cache=cache); s = summarize(df)
    out[f'fix_{fix}'] = dict(pts_model=s['pts_model'], pts_blend50=s['pts_blend50']['pts'], pts_form=s['pts_form'],
                             vs_form_model=s['model_minus_form']['mean'], vs_form_blend50=s['pts_blend50']['minus_form'],
                             score_mae=s['score_mae_model'], bias_1st=float((df.pred_a - df.act_a).mean()), bias_2nd=float((df.pred_b - df.act_b).mean()),
                             spearman=s['spearman'], coverage=s['coverage_p10_p90'], secs=round(time.time() - t))
    print(which, fix, out[f'fix_{fix}'], flush=True)
json.dump(out, open(f'reports/experiments/check_fix_{which}.json', 'w'), indent=1)
