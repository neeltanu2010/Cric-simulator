"""Command line: simulate a match from a squad file and print the best Dream11 team."""
from __future__ import annotations

import argparse

import numpy as np

from .loader import load_matches
from .model import RateModel
from .optimizer import pick_team
from .simulator import simulate
from .squads import load_squads


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("squad", help="squad JSON file for the match")
    ap.add_argument("--data", default="data/ipl_json", help="folder of Cricsheet JSON matches")
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--risk", type=float, default=0.0, help="0 = safe, 1 = chase upside")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    balls = load_matches(a.data)
    model = RateModel().fit(balls)
    team_a, team_b = load_squads(a.squad)
    unknown = [p.name for t in (team_a, team_b) for p in t.batting_order if not model.known(p.name)]
    if unknown:
        print("Warning: no history for (using league averages):", ", ".join(unknown))

    sim = simulate(model, team_a, team_b, a.sims, a.seed)
    print(f"\n{team_a.name} avg {sim['avg_score_a']:.0f}  win {sim['p_a_wins']:.0%}   |   "
          f"{team_b.name} avg {sim['avg_score_b']:.0f}  win {sim['p_b_wins']:.0%}\n")

    players, pts = sim["players"], sim["points"]
    print(f"{'Player':<22}{'Role':<6}{'Cr':>5}{'Avg':>7}{'P10':>6}{'P90':>6}")
    for i in np.argsort(-pts.mean(axis=0)):
        p = players[i]
        print(f"{p.name:<22}{p.role:<6}{p.credits:>5.1f}{pts[:, i].mean():>7.1f}"
              f"{np.percentile(pts[:, i], 10):>6.0f}{np.percentile(pts[:, i], 90):>6.0f}")

    best = pick_team(players, pts, risk=a.risk)
    print(f"\nBest XI ({best['credits']:.1f} credits, expected {best['expected_points']:.0f} pts)")
    for p in best["team"]:
        tag = " (C)" if p is best["captain"] else " (VC)" if p is best["vice_captain"] else ""
        print(f"  {p.name:<22}{p.role:<6}{p.team}{tag}")


if __name__ == "__main__":
    main()
