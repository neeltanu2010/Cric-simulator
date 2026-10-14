"""Command line.

  python -m cric_sim.cli predict squads/match.json --data data/ipl_json
  python -m cric_sim.cli backtest --data data/ipl_json --n-test 150 --ablate
  python -m cric_sim.cli types --data data/ipl_json --out data/players.csv
  python -m cric_sim.cli venues --data data/ipl_json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .backtest import (BTConfig, STAGES, format_ablation, format_report, prepare, run_ablation,
                       run_backtest, summarize)
from .loader import load_matches
from .model import ALL_FEATURES, RateModel
from .optimizer import pick_team
from .players import bowl_types, write_template
from .simulator import simulate
from .squads import load_squads


def _load(a):
    balls, matches = load_matches(a.data)
    bf, st = prepare(balls, matches)
    return balls, matches, bf, st


def cmd_predict(a):
    _, matches, bf, _ = _load(a)
    model = RateModel().fit(bf, bowl_types(bf, a.types_csv))
    team_a, team_b, venue = load_squads(a.squad)
    venue = a.venue or venue
    unknown = [p.name for t in (team_a, team_b) for p in t.roster if not model.known(p.name)]
    if unknown:
        print("Warning: no history for (league averages used):", ", ".join(unknown))
    if venue and venue not in model.ids["venue"]:
        print(f"Warning: ground '{venue}' not in the data; ground effects are off for this match.")
    elif venue:
        vs = model.venue_stats.get(venue)
        m = matches[matches["venue_key"] == venue]
        chase = float((m["winner"] != m["team_a"]).mean()) if len(m) else float("nan")
        print(f"\nGround: {venue}  ({vs['balls']} balls of data)  runs/over vs league {vs['runs_per_over_vs_league']:+.2f}, "
              f"wicket rate x{vs['wicket_rate_ratio']:.2f}, chasing side won {chase:.0%} of {len(m)} matches")

    sim = simulate(model, team_a, team_b, a.sims, a.seed, venue)
    print(f"\n{team_a.name} (bats first) avg {sim['avg_score_a']:.0f}  win {sim['p_a_wins']:.0%}   |   "
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


def cmd_backtest(a):
    balls, matches, bf, st = _load(a)
    cfg = BTConfig(n_test=a.n_test, min_history=a.min_history, sims=a.sims, refit_every=a.refit_every,
                   seed=a.seed, types_csv=a.types_csv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    if a.ablate:
        res = run_ablation(bf, matches, st, cfg)
        (out / "ablation.json").write_text(json.dumps(res, indent=1, default=float))
        text = "# Which feature helps?\n\n" + format_ablation(res) + "\n\n" + format_report(res[STAGES[-1][0]], "Full model")
    else:
        df = run_backtest(bf, matches, st, cfg, progress=True)
        df.to_csv(out / "backtest_matches.csv", index=False)
        text = format_report(summarize(df), "Backtest")
    (out / "report.md").write_text(text)
    print(text)
    print(f"\nSaved to {out}/ ({time.time() - t:.0f}s)")


def cmd_types(a):
    balls, _ = load_matches(a.data)
    n = write_template(balls, a.out)
    print(f"Wrote {n} bowlers to {a.out}. Open it, fix the bowl_type column (pace/spin) where our guess is wrong,"
          f" then pass --types-csv {a.out}")


def cmd_venues(a):
    _, _, bf, _ = _load(a)
    model = RateModel().fit(bf, bowl_types(bf, a.types_csv))
    print(f"{'Ground':<40}{'Balls':>7}{'Runs/over vs league':>21}{'Wicket x':>10}")
    for v, s in sorted(model.venue_stats.items(), key=lambda kv: -kv[1]["runs_per_over_vs_league"]):
        if s["balls"] >= 600:
            print(f"{v:<40}{s['balls']:>7}{s['runs_per_over_vs_league']:>+21.2f}{s['wicket_rate_ratio']:>10.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("predict", cmd_predict), ("backtest", cmd_backtest), ("types", cmd_types), ("venues", cmd_venues)):
        sp = sub.add_parser(name)
        sp.add_argument("--data", default="data/ipl_json", help="folder of Cricsheet JSON matches")
        sp.add_argument("--types-csv", default=None, help="CSV: name,bowl_type,role (your corrections)")
        sp.add_argument("--seed", type=int, default=0)
        sp.set_defaults(fn=fn)
        if name == "predict":
            sp.add_argument("squad")
            sp.add_argument("--venue", default=None)
            sp.add_argument("--sims", type=int, default=5000)
            sp.add_argument("--risk", type=float, default=0.0, help="0 = safe, 1 = chase upside")
        if name == "backtest":
            sp.add_argument("--n-test", type=int, default=150, help="how many of the latest matches to test")
            sp.add_argument("--min-history", type=int, default=100)
            sp.add_argument("--sims", type=int, default=1000)
            sp.add_argument("--refit-every", type=int, default=10)
            sp.add_argument("--ablate", action="store_true", help="add features one at a time")
            sp.add_argument("--out", default="reports")
        if name == "types":
            sp.add_argument("--out", default="data/players.csv")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
