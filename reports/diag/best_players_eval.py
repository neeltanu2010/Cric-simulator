"""How well does the best strategy find the best Dream11 players in each IPL 2025-26 match?"""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from cric_sim.lineup import form_points
from cric_sim.optimizer import pick_team
from cric_sim.simulator import Player

p = pd.read_csv('reports/diag/ipl2526_players.csv', dtype={'match_id': str})
share = 0.5
# value our strategy uses: 50% simulator + 50% recent form (decay 0.95). The saved 'form' column is the
# reported baseline (decay 0.9); recompute the 0.95 form only if available, else use the saved one.
try:
    f95 = pd.read_csv('reports/diag/form_d095_k2.csv', dtype={'match_id': str})
    p = p.merge(f95, on=['match_id', 'name'], how='left')
    p['mixform'] = p['f95'].fillna(p['form'])
except FileNotFoundError:
    p['mixform'] = p['form']
p['ours'] = share * p['pred'] + (1 - share) * p['mixform']

rows = []
for mid, g in p.groupby('match_id'):
    g = g.reset_index(drop=True)
    if g['played'].sum() < 18:
        continue
    act = g['actual'].to_numpy()
    players = [Player(n, r, c, t) for n, r, c, t in zip(g['name'], g['role'], g['credits'], g['team'])]
    best = pick_team(players, act[None, :])                  # best possible XI in hindsight
    ours = pick_team(players, g['ours'].to_numpy()[None, :])
    form = pick_team(players, g['form'].to_numpy()[None, :])
    real_rank = (-act).argsort().argsort()                  # 0 = real top scorer
    top1 = g['name'][act.argmax()]
    top5 = set(g['name'][real_rank < 5])
    bestxi = {q.name for q in best['team']}
    r = {'match_id': mid, 'top1': top1, 'top1_pts': act.max()}
    for tag, pick, val in (('ours', ours, g['ours'].to_numpy()), ('form', form, g['form'].to_numpy())):
        xi = {q.name for q in pick['team']}
        cap = pick['captain'].name
        r[f'{tag}_top1_in_xi'] = top1 in xi
        r[f'{tag}_top1_is_c'] = cap == top1
        r[f'{tag}_top1_is_c_or_vc'] = top1 in (cap, pick['vice_captain'].name)
        r[f'{tag}_top5_in_xi'] = len(top5 & xi)
        r[f'{tag}_bestxi_overlap'] = len(bestxi & xi)
        r[f'{tag}_cap_real_rank'] = int(real_rank[g.index[g['name'] == cap][0]]) + 1
        r[f'{tag}_pred_top1_correct'] = g['name'][val.argmax()] == top1
    rows.append(r)
d = pd.DataFrame(rows)
n = len(d)
print(f"IPL 2025-26 matches analysed: {n}\n")
print(f"{'':52s}{'Our strategy':>14s}{'Form rule':>11s}")
def line(label, a, b, fmt):
    print(f"{label:52s}{fmt(a):>14s}{fmt(b):>11s}")
pc = lambda x: f"{x:.0%}"
line("Match's real top Dream11 scorer was in our XI", d.ours_top1_in_xi.mean(), d.form_top1_in_xi.mean(), pc)
line("  ... and was our captain", d.ours_top1_is_c.mean(), d.form_top1_is_c.mean(), pc)
line("  ... and was our captain or vice-captain", d.ours_top1_is_c_or_vc.mean(), d.form_top1_is_c_or_vc.mean(), pc)
line("Our no.1 predicted player was the real top scorer", d.ours_pred_top1_correct.mean(), d.form_pred_top1_correct.mean(), pc)
line("Real top-5 players in our XI (out of 5, average)", d.ours_top5_in_xi.mean(), d.form_top5_in_xi.mean(), lambda x: f"{x:.2f}")
line("Players shared with the best possible XI (of 11)", d.ours_bestxi_overlap.mean(), d.form_bestxi_overlap.mean(), lambda x: f"{x:.2f}")
line("Captain finished in the real top 3", (d.ours_cap_real_rank <= 3).mean(), (d.form_cap_real_rank <= 3).mean(), pc)
line("Captain finished in the real top half (11 of 22)", (d.ours_cap_real_rank <= 11).mean(), (d.form_cap_real_rank <= 11).mean(), pc)
print("\nHow many of the real top 5 our XI contained, per match:")
vc = d.ours_top5_in_xi.value_counts().sort_index()
print("  " + "   ".join(f"{k} of 5: {v} matches ({v/n:.0%})" for k, v in vc.items()))
print("\nPlayers shared with the best possible XI, per match:")
vc = d.ours_bestxi_overlap.value_counts().sort_index()
print("  " + "   ".join(f"{k}: {v}" for k, v in vc.items()))
d.to_csv('reports/diag/ipl2526_best_players.csv', index=False)
