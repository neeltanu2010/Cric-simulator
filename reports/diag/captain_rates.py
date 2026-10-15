"""Captain / vice-captain success for the decider's players: in their past T20 matches, how often was each one
the top Dream11 scorer of the whole match (a perfect captain) or in the top 2 (a perfect captain or vice-captain)?
Also: the top scorers in each game of this IND vs WI series."""
import sys, json; sys.path.insert(0, '.')
import pandas as pd
from cric_sim.experiments import POOL_FOLDERS
from cric_sim.loader import load_many
from cric_sim.backtest import prepare
from cric_sim.scorecards import add_scorecards
from cric_sim.actual import points_of
from cric_sim.lineup import history_table, infer_roles

folders, only = POOL_FOLDERS['leagues']
b, m = load_many([f'data/{f}_json' for f in folders], full_members_only=only)
_, st = prepare(b, m)
m, st = add_scorecards(m, st, ['data/manual/ind_wi_2026_t20i_2to4.json'])
roles = infer_roles(history_table(st))
sq = json.load(open('squads/ind_vs_wi_5th_t20i_2026-10-17_official_roles.json'))
squad = {p['name']: (sq[t]['name'], p['role']) for t in ('team_a', 'team_b') for p in sq[t]['players']}
roles.update({n: r for n, (_, r) in squad.items()})
st['pts'] = points_of(st, roles)
st['rank'] = st.groupby('match_id')['pts'].rank(ascending=False, method='min')  # 1 = top scorer of the match

RECENT = pd.Timestamp('2025-01-01')
rows = []
for name, (team, role) in squad.items():
    g = st[st['name'] == name]
    r = g[g['date'] >= RECENT]
    rows.append({'player': name, 'team': team, 'role': role,
                 'games since 2025': len(r), 'top scorer % (C)': 100 * (r['rank'] == 1).mean() if len(r) else float('nan'),
                 'top 2 % (C or VC)': 100 * (r['rank'] <= 2).mean() if len(r) else float('nan'),
                 'avg pts': r['pts'].mean() if len(r) else float('nan'),
                 'all games': len(g), 'all top scorer %': 100 * (g['rank'] == 1).mean() if len(g) else float('nan'),
                 'all top 2 %': 100 * (g['rank'] <= 2).mean() if len(g) else float('nan')})
d = pd.DataFrame(rows).sort_values(['top 2 % (C or VC)', 'top scorer % (C)'], ascending=False)
pd.set_option('display.width', 200)
print("=== HOW OFTEN EACH PLAYER WAS THE BEST CAPTAIN (top scorer of the match) OR BEST C/VC (top 2) ===")
print("(a random player out of 22 would be top scorer 4.5% and top 2 9.1% of the time)\n")
print(d.round(1).to_string(index=False))

series = m[((m['team_a'] == 'India') & (m['team_b'] == 'West Indies') | (m['team_a'] == 'West Indies') & (m['team_b'] == 'India'))
           & (m['date'] >= '2026-10-01')].sort_values('date')
print("\n=== THIS SERIES: TOP 3 SCORERS IN EACH GAME (2nd-4th from scorecards: fours, sixes and dots partly estimated) ===")
for k, (_, mm) in enumerate(series.iterrows(), 1):
    g = st[st['match_id'] == mm['match_id']].sort_values('pts', ascending=False).head(3)
    top = ", ".join(f"{n} {p:.0f}" for n, p in zip(g['name'], g['pts']))
    print(f"T20I {k} ({mm['date']:%d %b}, {mm['venue_key']}, won by {mm['winner']}): {top}")
