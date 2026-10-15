"""How did the decider team's players do in past T20s at M Chinnaswamy Stadium, Bengaluru, compared with their
usual T20 numbers? Dream11 points use the official Dream11 roles from the app."""
import sys, json; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from cric_sim.loader import load_many
from cric_sim.backtest import prepare
from cric_sim.actual import points_of
F = ['ipl', 'bbl', 'psl', 'sat', 'wc', 't20s']
b, m = load_many([f'data/{f}_json' for f in F])
_, st = prepare(b, m)
st = st.merge(m[['match_id', 'league']], on='match_id')
sq = json.load(open('squads/ind_vs_wi_5th_t20i_2026-10-17_official_roles.json'))
roles = {p['name']: p['role'] for t in ('team_a', 'team_b') for p in sq[t]['players']}
st = st[st['name'].isin(roles)].copy()
st['pts'] = points_of(st, roles)
TEAM = ["SV Samson", "Abhishek Sharma", "Ishan Kishan", "SS Iyer", "Nithish Kumar Reddy", "Arshdeep Singh",
        "SD Hope", "J Andrew", "SO Hetmyer", "G Motie", "S Joseph"]
EXTRA = ["Prince Yadav", "RL Chase", "SE Rutherford", "KMA Paul", "Tilak Varma", "AR Patel"]
venue = 'M Chinnaswamy Stadium'
def line(n):
    g = st[st['name'] == n]; v = g[g['venue_key'] == venue]
    if v.empty:
        return [n, roles[n], 0, '-', round(g['pts'].mean(), 1) if len(g) else '-', '-', '-', '-', '-', 'never played here']
    bat = v[v['balls'] > 0]; bowl = v[v['bowl_balls'] > 0]
    runs = int(bat['runs'].sum()); balls = int(bat['balls'].sum()); outs = int(bat['out'].sum())
    bat_s = f"{runs} runs, avg {runs / max(outs, 1):.0f}, SR {100 * runs / max(balls, 1):.0f}" if len(bat) else "-"
    wk, rc, bb = int(bowl['wkts'].sum()), int(bowl['bowl_runs'].sum()), int(bowl['bowl_balls'].sum())
    bowl_s = f"{wk} wkts, econ {6 * rc / max(bb, 1):.1f}" if bb >= 6 else "-"
    best = v.loc[v['pts'].idxmax()]
    best_s = f"{best['pts']:.0f} pts ({int(best['runs'])}r" + (f", {int(best['wkts'])}w" if best['wkts'] else "") + f", {best['date'].year} {best['league'].upper()})"
    last = v['date'].max().strftime('%b %Y')
    return [n, roles[n], len(v), round(v['pts'].mean(), 1), round(g['pts'].mean(), 1), bat_s, bowl_s, best_s, last,
            int((v['pts'] >= 70).sum())]
cols = ['player', 'role', 'games here', 'avg pts here', 'avg pts all T20s', 'batting here', 'bowling here', 'best game here', 'last game here', 'games with 70+ pts here']
for title, names in (("THE TEAM", TEAM), ("BACKUPS / CLOSE CALLS", EXTRA)):
    d = pd.DataFrame([line(n) for n in names], columns=cols)
    d['diff'] = [round(a - b_, 1) if isinstance(a, float) and isinstance(b_, float) else '-' for a, b_ in zip(d['avg pts here'], d['avg pts all T20s'])]
    print(f"\n=== {title} ===")
    print(d[['player', 'role', 'games here', 'avg pts here', 'avg pts all T20s', 'diff', 'games with 70+ pts here', 'last game here']].to_string(index=False))
    print(d[['player', 'batting here', 'bowling here', 'best game here']].to_string(index=False))
# context: how high-scoring is the ground? first-innings totals since 2023 (full innings only: 19+ overs or all out)
inn = b.groupby(['match_id', 'innings', 'venue_key', 'league', 'date'], as_index=False).agg(
    runs=('runs_total', 'sum'), last_over=('over', 'max'), wkts=('wicket', 'sum'))
inn = inn[(inn['innings'] == 1) & (inn['date'] >= '2023-01-01') & ((inn['last_over'] >= 18) | (inn['wkts'] >= 10))]
print("\n=== THE GROUND: average first-innings total since 2023 ===")
for lg in sorted(inn.loc[inn['venue_key'] == venue, 'league'].unique()):
    h = inn[(inn['league'] == lg) & (inn['venue_key'] == venue)]; o = inn[(inn['league'] == lg) & (inn['venue_key'] != venue)]
    print(f"{lg.upper()}: {h['runs'].mean():.0f} here ({len(h)} games) vs {o['runs'].mean():.0f} at other grounds ({len(o)} games)")
