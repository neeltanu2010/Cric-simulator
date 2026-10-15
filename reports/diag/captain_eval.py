"""Compare captain / vice-captain rules. The XI is the same for every rule (picked by the 50/50 value);
only the C and VC change. Learned rules are trained only on seasons before the one they pick for."""
import sys; sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cric_sim.optimizer import pick_team
from cric_sim.simulator import Player

f95 = pd.read_csv('reports/diag/form_d095_k2.csv', dtype={'match_id': str})
def load(label):
    p = pd.read_csv(f'reports/diag/cap_players_{label}.csv', dtype={'match_id': str})
    p = p.merge(f95, on=['match_id', 'name'], how='left')
    p['f95'] = p['f95'].fillna(p['form'])
    p['value'] = 0.5 * p['pred'] + 0.5 * p['f95']
    p['season'] = pd.to_datetime(p['date']).dt.year
    p['upside'] = p['pred_p90'] - p['pred']
    return p
rows = pd.concat([load(l) for l in sys.argv[1].split(',')], ignore_index=True)

FEATS = ['pred', 'pred_p90', 'pred_p75', 'pred_std', 'p_top1', 'p_top3', 'p_ge75', 'p_ge100', 'form', 'f95',
         'team_win_p', 'plan_pos', 'bowl_weight', 'bats_first', 'is_sub', 'credits', 'hist_bowl_balls', 'hist_bat_balls']
def X(d):
    x = d[FEATS].astype(float).fillna(0.0).copy()
    x['plan_pos'] = d['plan_pos'].fillna(11) % 11
    for r in ('WK', 'BAT', 'AR', 'BOWL'):
        x['role_' + r] = (d['role'] == r).astype(float)
    x['log_matches'] = np.log1p(d['hist_matches'])
    return x

# labels: real top scorer / top 3 of each match
rows['rank'] = rows.groupby('match_id')['actual'].rank(ascending=False, method='first')
rows['is_top1'] = (rows['rank'] == 1).astype(int)
rows['is_top3'] = (rows['rank'] <= 3).astype(int)

# walk-forward learned captain scores
for kind in ('logit1', 'logit3', 'gbm1', 'gbm3'):
    rows[kind] = np.nan
    for s in sorted(rows['season'].unique()):
        tr, te = rows[(rows['season'] < s) & rows['played']], rows[rows['season'] == s]
        if tr['match_id'].nunique() < 150:
            continue
        y = tr['is_top1'] if kind.endswith('1') else tr['is_top3']
        m = (make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)) if kind.startswith('logit')
             else HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=80, random_state=0))
        m.fit(X(tr), y)
        rows.loc[te.index, kind] = m.predict_proba(X(te))[:, 1]

RULES = {
    'current (50/50 value)': 'value', 'simulator average': 'pred', 'chance of top score (sim)': 'p_top1',
    'chance of top-3 (sim)': 'p_top3', 'chance of 100+ (sim)': 'p_ge100', 'chance of 75+ (sim)': 'p_ge75',
    'value + 0.5 x upside': 'v_up05', 'value + 1 x upside': 'v_up10', 'learned P(top scorer), logistic': 'logit1',
    'learned P(top 3), logistic': 'logit3', 'learned P(top scorer), trees': 'gbm1', 'learned P(top 3), trees': 'gbm3'}
rows['v_up05'] = rows['value'] + 0.5 * rows['upside']
rows['v_up10'] = rows['value'] + 1.0 * rows['upside']

def evaluate(d, split=False):
    out = {k: [] for k in RULES}
    out_best = []
    for mid, g in d.groupby('match_id'):
        g = g[g['played']].reset_index(drop=True)
        if len(g) < 18 or g[list(RULES.values())].isna().any().any():
            continue
        pl = [Player(n, r, c, t) for n, r, c, t in zip(g['name'], g['role'], g['credits'], g['team'])]
        xi = pick_team(pl, g['value'].to_numpy()[None, :])
        idx = [i for i, q in enumerate(pl) if q in xi['team']]
        act = g['actual'].to_numpy(); top1 = act.argmax(); top3 = set(np.argsort(-act)[:3])
        base = act[idx].sum()
        best_c = max(idx, key=lambda i: act[i]); best_v = max([i for i in idx if i != best_c], key=lambda i: act[i])
        out_best.append(base + act[best_c] + 0.5 * act[best_v])
        for name, col in RULES.items():
            sc = g[col].to_numpy()
            order = sorted(idx, key=lambda i: -sc[i])
            c = order[0]
            v = next(i for i in order[1:] if (pl[i].team != pl[c].team) or not split)
            out[name].append((c == top1, c in top3, top1 in (c, v), act[c], base + act[c] + 0.5 * act[v]))
    return out, np.mean(out_best)

for split in (False, True):
    res, best = evaluate(rows[rows['season'].isin([int(x) for x in sys.argv[2].split(',')])], split)
    n = len(next(iter(res.values())))
    print(f"\nSeasons {sys.argv[2]} | {n} matches | C and VC from {'DIFFERENT teams' if split else 'any team'}")
    print(f"{'captain rule':36s}{'C = top':>8s}{'C top3':>8s}{'C/VC=top':>9s}{'C pts':>7s}{'team pts':>9s}")
    for name, r in sorted(res.items(), key=lambda kv: -np.mean([x[4] for x in kv[1]])):
        a = np.array(r, dtype=float)
        print(f"{name:36s}{a[:,0].mean():8.1%}{a[:,1].mean():8.1%}{a[:,2].mean():9.1%}{a[:,3].mean():7.1f}{a[:,4].mean():9.1f}")
    print(f"{'(best possible C/VC inside our XI)':36s}{'100%':>8s}{'':>8s}{'':>9s}{'':>7s}{best:9.1f}")
rows.to_csv(f"reports/diag/captain_scores_{sys.argv[2].replace(',', '_')}.csv", index=False)
