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
import pandas as pd

from .backtest import (BTConfig, STAGES, format_ablation, format_report, prepare, run_ablation,
                       run_backtest, summarize)
from .loader import load_many, load_matches
from .model import ALL_FEATURES, RateModel
from .optimizer import pick_team
from .players import bowl_types, write_template
from .simulator import simulate
from .squads import load_squads


def _load(a):
    balls, matches = load_many(a.data or ["data/ipl_json"])
    bf, st = prepare(balls, matches)
    matches, st = _extras(a, matches, st)
    return balls, matches, bf, st


def _extras(a, matches, st):
    """Matches known only from published scorecards (``--extra-scorecards``)."""
    paths = getattr(a, "extra_scorecards", None)
    if not paths:
        return matches, st
    from .scorecards import add_scorecards
    n0 = len(matches)
    matches, st = add_scorecards(matches, st, paths)
    print(f"Added {len(matches) - n0} scorecard-only matches from {', '.join(paths)} "
          "(they update form, batting positions and credits; ball-by-ball ratings are unchanged)")
    return matches, st


def cmd_predict(a):
    cfg = json.loads(Path(a.config).read_text()) if a.config and Path(a.config).exists() else {}
    if cfg and not a.data:
        from .experiments import POOL_FOLDERS
        folders, only = POOL_FOLDERS[cfg["pool"]]
        balls, matches = load_many([f"data/{f}_json" for f in folders], full_members_only=only)
        bf, st = prepare(balls, matches)
        matches, st = _extras(a, matches, st)
    else:
        a.data = a.data or ["data/ipl_json"]
        _, matches, bf, st = _load(a)
    kw = dict(features=cfg["features"], sigmas=tuple(cfg["sigmas"]), half_life_days=cfg["half_life_days"],
              prior_scale=cfg["prior_scale"]) if cfg else {}
    share = cfg.get("simulator_share", 1.0)
    print(f"Setting: {a.config if cfg else 'defaults'}  (simulator share {share:.0%} of the team pick)")
    model = RateModel(**kw).fit(bf, bowl_types(bf, a.types_csv))
    team_a, team_b, venue = load_squads(a.squad)
    if cfg.get("empirical_lineup"):
        from .lineup import lineup_history, phase_overs_table
        bph = phase_overs_table(bf)
        for t in (team_a, team_b):
            t.pos_hist, t.phase_overs = lineup_history(st, bph, [p.name for p in t.roster])
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

    sq = json.loads(Path(a.squad).read_text())
    pitch = dict(sq.get("pitch") or {})
    for k in ("scoring", "spin", "pace"):
        v = getattr(a, f"pitch_{k}")
        if v is not None:
            pitch[k] = float(v) if v.replace(".", "", 1).isdigit() else v
    if any(str(v) not in ("normal", "1", "1.0") for v in pitch.values()):
        print("Pitch report used:", ", ".join(f"{k} = {v}" for k, v in pitch.items()))
    adj = None
    if a.recent_pitch and venue:
        rec = bf[(bf["venue_key"] == venue) & (bf["date"] >= bf["date"].max() - pd.Timedelta(days=60))]
        if len(rec):
            adj = model.recent_pitch(rec)
            print(f"Recent pitch ({rec['match_id'].nunique()} matches here in the last 60 days of data): "
                  f"fours x{adj[4]:.2f}, sixes x{adj[5]:.2f}, wickets x{adj[6]:.2f}")
    sim = simulate(model, team_a, team_b, a.sims, a.seed, venue, league=a.league, afternoon=a.afternoon,
                   pitch_adj=adj, pitch_report=pitch or None)
    if a.toss_unknown:  # also play it with the other side batting first; the pick must work either way
        sim2 = simulate(model, team_b, team_a, a.sims, a.seed + 1, venue, league=a.league, afternoon=a.afternoon,
                        pitch_adj=adj, pitch_report=pitch or None)
        col = {p.name: i for i, p in enumerate(sim2["players"])}
        order = [col[p.name] for p in sim["players"]]
        print(f"\nIf {team_b.name} bat first: {team_b.name} avg {sim2['avg_score_a']:.0f}  win {sim2['p_a_wins']:.0%}   |   "
              f"{team_a.name} avg {sim2['avg_score_b']:.0f}  win {sim2['p_b_wins']:.0%}")
        sim = dict(sim, points=np.vstack([sim["points"], sim2["points"][:, order]]))  # toss: 50/50 either way
    print(f"\n{'If ' if a.toss_unknown else ''}{team_a.name} {'bat first' if a.toss_unknown else '(bats first)'}: {team_a.name} avg {sim['avg_score_a']:.0f}  win {sim['p_a_wins']:.0%}   |   "
          f"{team_b.name} avg {sim['avg_score_b']:.0f}  win {sim['p_b_wins']:.0%}\n")
    players, pts = sim["players"], sim["points"]
    fv = None
    if share < 1.0:  # recent form, the other half of the pick value
        from .lineup import form_points, history_table, infer_roles
        roles = infer_roles(history_table(st))
        form, role_mean, glob = form_points(st, roles, decay=cfg.get("mix_form_decay", 0.9))
        fv = np.array([form.get(p.name, role_mean.get(p.role, glob)) for p in players])
    value = pts.mean(axis=0) if fv is None else share * pts.mean(axis=0) + (1 - share) * fv
    print(f"{'Player':<22}{'Role':<6}{'Cr':>5}{'Sim avg':>8}{'P10':>5}{'P90':>5}" + (f"{'Form':>7}{'Pick value':>11}" if fv is not None else ""))
    for i in np.argsort(-value):
        p = players[i]
        extra = f"{fv[i]:>7.1f}{value[i]:>11.1f}" if fv is not None else ""
        print(f"{p.name:<22}{p.role:<6}{p.credits:>5.1f}{pts[:, i].mean():>8.1f}"
              f"{np.percentile(pts[:, i], 10):>5.0f}{np.percentile(pts[:, i], 90):>5.0f}" + extra)
    values = pts if fv is None else value[None, :]
    if fv is not None:
        print(f"\nPick value = {share:.0%} simulator + {1 - share:.0%} recent form")
    cap_val = pts.mean(axis=0) if cfg.get("captain_by") == "simulator" else None
    if a.exclude:  # doubtful players: keep them out of every team
        out = {x.strip() for x in a.exclude.split(",") if x.strip()}
        bad = np.array([p.name in out for p in players])
        if bad.any():
            print("Left out of the picks:", ", ".join(p.name for p in players if p.name in out))
            values = np.where(bad[None, :], -1e4, values)
            cap_val = np.where(bad, -1e4, pts.mean(axis=0) if cap_val is None else cap_val)
    best = pick_team(players, values, risk=a.risk, captain_value=cap_val)
    if a.teams > 1:
        from .multi import diverse_teams
        spread = a.spread if a.spread is not None else cfg.get("multi_team_spread", 0.3)
        teams = diverse_teams(players, values.mean(axis=0), cap_val if cap_val is not None else values.mean(axis=0),
                              pts.std(axis=0), n=a.teams, spread=spread, seed=a.seed)
    P = pts
    p_top1 = np.bincount(P.argmax(axis=1), minlength=P.shape[1]) / P.shape[0]
    rank = np.argsort(-P, axis=1)
    p_top2 = np.bincount(rank[:, :2].ravel(), minlength=P.shape[1]) / P.shape[0]  # a perfect captain or vice-captain
    p_top3 = np.bincount(rank[:, :3].ravel(), minlength=P.shape[1]) / P.shape[0]
    print("\nCaptain candidates (captain = highest simulator average; tested best on 2022-26 backtests)")
    print(f"  {'Player':<22}{'Sim avg':>8}{'Top scorer':>12}{'Top 2':>8}{'Top 3':>8}")
    for i in np.argsort(-P.mean(axis=0))[:10]:
        print(f"  {players[i].name:<22}{P[:, i].mean():>8.1f}{p_top1[i]:>12.0%}{p_top2[i]:>8.0%}{p_top3[i]:>8.0%}")
    if a.teams > 1:
        print(f"\n{len(teams)} different teams (team 1 = the best single pick; the others each suit a different version "
              f"of the match; spread {spread})")
        for k, t in enumerate(teams, 1):
            others = ", ".join(p.name for p in t["team"] if p is not t["captain"] and p is not t["vice_captain"])
            print(f"  Team {k:2d} ({t['credits']:.1f} cr): C {t['captain'].name}, VC {t['vice_captain'].name} | {others}")
    print(f"\nBest XI ({best['credits']:.1f} credits, expected {best['expected_points']:.0f} pts)")
    for p in best["team"]:
        tag = " (C)" if p is best["captain"] else " (VC)" if p is best["vice_captain"] else ""
        print(f"  {p.name:<22}{p.role:<6}{p.team}{tag}")


def cmd_backtest(a):
    balls, matches, bf, st = _load(a)
    cfg = BTConfig(n_test=a.n_test, min_history=a.min_history, sims=a.sims, refit_every=a.refit_every,
                   seed=a.seed, types_csv=a.types_csv, test_league=a.test_league)
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


def cmd_squad(a):
    from .build_squad import build_squad, write_squad
    _, matches, bf, st = _load(a)
    split = lambda x: [p for p in (x or "").split(",") if p.strip()]
    nums = lambda x: [float(v) for v in split(x)] or None
    squad, notes = build_squad(st, matches, a.team_a, split(a.xi_a), a.team_b, split(a.xi_b), a.venue, a.sub_a, a.sub_b,
                               nums(a.credits_a), nums(a.credits_b),
                               {"scoring": a.pitch_scoring, "spin": a.pitch_spin, "pace": a.pitch_pace},
                               keepers=split(a.keepers))
    print("\n".join(notes))
    write_squad(squad, a.out)
    print(f"\nWrote {a.out} (team A = {squad['team_a']['name']} bats first). Check the matches above, then run:\n"
          f"  python -m cric_sim.cli predict {a.out}")


def cmd_types(a):
    balls, _ = load_many(a.data or ["data/ipl_json"])
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
    for name, fn in (("predict", cmd_predict), ("backtest", cmd_backtest), ("types", cmd_types), ("venues", cmd_venues),
                     ("squad", cmd_squad)):
        sp = sub.add_parser(name)
        sp.add_argument("--data", nargs="+", default=None,
                        help="one or more folders of Cricsheet JSON matches (all are used for training); "
                             "default: ipl_json, or the data pool named in configs/best.json for predict")
        sp.add_argument("--types-csv", default=None, help="CSV: name,bowl_type,role (your corrections)")
        sp.add_argument("--extra-scorecards", nargs="+", default=None,
                        help="JSON files of recent matches known only as scorecards (e.g. data/manual/*.json)")
        sp.add_argument("--seed", type=int, default=0)
        sp.set_defaults(fn=fn)
        if name == "predict":
            sp.add_argument("squad")
            sp.add_argument("--venue", default=None)
            sp.add_argument("--league", default="ipl", help="which league's scoring level to use")
            sp.add_argument("--config", default="configs/best.json", help="setting chosen by the experiments")
            sp.add_argument("--pitch-scoring", default=None, help="pitch report: very high / high / normal / low / very low, or a number")
            sp.add_argument("--pitch-spin", default=None, help="help for spinners: big / some / normal / little / none, or a number")
            sp.add_argument("--pitch-pace", default=None, help="help for pace bowlers: big / some / normal / little / none, or a number")
            sp.add_argument("--recent-pitch", action="store_true", help="adjust for how this ground played in its last 60 days")
            sp.add_argument("--afternoon", action="store_true", help="afternoon game (no evening dew); needs the daynight effect")
            sp.add_argument("--sims", type=int, default=5000)
            sp.add_argument("--risk", type=float, default=0.0, help="0 = safe, 1 = chase upside")
            sp.add_argument("--teams", type=int, default=1, help="also list this many different teams (e.g. 3, 6, 11, 20)")
            sp.add_argument("--toss-unknown", action="store_true", help="simulate both batting orders; pick a team that works either way")
            sp.add_argument("--exclude", default=None, help="doubtful players to leave out of the picks (data names, comma separated)")
            sp.add_argument("--spread", type=float, default=None, help="how far extra teams move from the main pick (default 0.3)")
        if name == "backtest":
            sp.add_argument("--n-test", type=int, default=150, help="how many of the latest matches to test")
            sp.add_argument("--min-history", type=int, default=100)
            sp.add_argument("--sims", type=int, default=1000)
            sp.add_argument("--refit-every", type=int, default=10)
            sp.add_argument("--ablate", action="store_true", help="add features one at a time")
            sp.add_argument("--test-league", default=None, help="only test matches from this league, e.g. ipl")
            sp.add_argument("--out", default="reports")
        if name == "squad":
            sp.add_argument("--team-a", required=True, help="team batting first, e.g. RCB or 'Royal Challengers'")
            sp.add_argument("--xi-a", required=True, help="the announced XI, comma separated, as written in the app")
            sp.add_argument("--team-b", required=True, help="team batting second")
            sp.add_argument("--xi-b", required=True)
            sp.add_argument("--venue", default=None)
            sp.add_argument("--sub-a", default=None, help="likely impact sub of team A (optional)")
            sp.add_argument("--sub-b", default=None)
            sp.add_argument("--credits-a", default=None, help="Dream11 credits in the same order as --xi-a, comma separated")
            sp.add_argument("--credits-b", default=None)
            sp.add_argument("--pitch-scoring", default="normal")
            sp.add_argument("--pitch-spin", default="normal")
            sp.add_argument("--pitch-pace", default="normal")
            sp.add_argument("--keepers", default=None, help="players Dream11 lists as wicket-keepers, comma separated")
            sp.add_argument("--out", default="squads/match.json")
        if name == "types":
            sp.add_argument("--out", default="data/players.csv")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
