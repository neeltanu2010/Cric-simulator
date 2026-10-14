"""Run MANY model settings on the same matches, record every result, and pick the best one.

Settings that vary
  features   which of 9 effects are switched on (lineup, form, venue, type, innings, state, extras, impact, league)
  sigmas     how big the day-to-day swings are
  half_life  how fast old matches fade
  prior      how hard player ratings are pulled toward average
  pool       which data is used for learning (IPL only / all leagues, associate nations removed / everything)
  blend      how much of the simulator vs the recent-form baseline is used to pick the team

Everything is scored on the SAME validation matches (IPL 2023-24). The best setting is then tested once on
IPL 2025-26, which was not used to choose it.

  python -m cric_sim.experiments worker features|hyperA|hyperB|pools
  python -m cric_sim.experiments merge
  python -m cric_sim.experiments final
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BLENDS, BTConfig, format_report, prepare, run_backtest, summarize
from .loader import load_many
from .model import ALL_FEATURES

DATA = Path("data")
OUT = Path("reports/experiments")
POOL_FOLDERS = {
    "ipl_only": (["ipl"], ("t20s",)),
    "leagues": (["ipl", "bbl", "psl", "sat", "wc", "t20s"], ("t20s",)),      # associate-nation matches removed
    "everything": (["ipl", "bbl", "psl", "sat", "wc", "t20s"], ()),          # every match you gave me
}
SIGMAS = [(0.3, 0.15, 0.1), (0.5, 0.25, 0.15), (0.7, 0.35, 0.2), (0.9, 0.45, 0.25)]
HALF_LIVES = [365.0, 730.0, 1500.0, 3000.0]
PRIORS = [1.0, 2.0, 4.0]
DEFAULT = dict(sigmas=(0.7, 0.35, 0.2), half_life_days=1500.0, prior_scale=2.0)
VAL = dict(n_test=145, test_before="2025-01-01", min_history=100, test_league="ipl", sims=500, refit_every=80,
           random_draws=4, seed=0)


def space_size() -> dict:
    toggles = 2 ** len(ALL_FEATURES)
    blends = len(BLENDS) + 2  # + simulator alone + form alone
    full = toggles * len(SIGMAS) * len(HALF_LIVES) * len(PRIORS) * len(POOL_FOLDERS) * blends
    return {"feature_switch_sets": toggles, "sigmas": len(SIGMAS), "half_lives": len(HALF_LIVES),
            "priors": len(PRIORS), "data_pools": len(POOL_FOLDERS), "team_pick_blends": blends,
            "total_on_paper": full}


def feature_configs():
    """No effects, all effects, each effect removed from the full model, each effect on its own."""
    out = [("none", ()), ("full", ALL_FEATURES)]
    out += [(f"minus_{f}", tuple(x for x in ALL_FEATURES if x != f)) for f in ALL_FEATURES]
    out += [(f"only_{f}", (f,)) for f in ALL_FEATURES]
    return out


def _rows(label, pool, features, cfg: BTConfig, df: pd.DataFrame, secs: float):
    s = summarize(df)
    base = dict(label=label, pool=pool, features="+".join(features) or "none", n_features=len(features),
                sigmas=str(cfg.sigmas), half_life_days=cfg.half_life_days, prior_scale=cfg.prior_scale,
                matches=s["matches"], seconds=round(secs), spearman=s["spearman"], score_mae=s["score_mae_model"],
                coverage=s["coverage_p10_p90"], brier=s.get("brier_model", np.nan), pts_form=s["pts_form"],
                pts_random=s["pts_random"], pts_oracle=s["pts_oracle"])
    rows = [dict(base, blend=1.0, pts_xi=s["pts_model"], minus_form=s["model_minus_form"]["mean"],
                 ci_lo=s["model_minus_form"]["ci95"][0], ci_hi=s["model_minus_form"]["ci95"][1])]
    for a in BLENDS:
        b = s[f"pts_blend{int(a * 100)}"]
        rows.append(dict(base, blend=a, pts_xi=b["pts"], minus_form=b["minus_form"], ci_lo=b["ci95"][0], ci_hi=b["ci95"][1]))
    return rows


def worker(group: str):
    OUT.mkdir(parents=True, exist_ok=True)
    pools = ["ipl_only", "everything"] if group == "pools" else ["leagues"]
    for pool in pools:
        folders, only = POOL_FOLDERS[pool]
        balls, matches = load_many([DATA / f"{f}_json" for f in folders], full_members_only=only)
        bf, st = prepare(balls, matches)
        print(f"[{group}] pool {pool}: {matches['match_id'].nunique()} matches, {len(balls)} balls", flush=True)
        if group == "features":
            jobs = [(name, feats, DEFAULT["sigmas"], DEFAULT["half_life_days"], DEFAULT["prior_scale"])
                    for name, feats in feature_configs()]
        elif group in ("hyperA", "hyperB"):
            hls = HALF_LIVES[:2] if group == "hyperA" else HALF_LIVES[2:]
            jobs = [(f"hl{int(h)}_prior{p:g}_sig{i}", ALL_FEATURES, sg, h, p)
                    for h, p in itertools.product(hls, PRIORS) for i, sg in enumerate(SIGMAS)]
        else:
            jobs = [("full", ALL_FEATURES, DEFAULT["sigmas"], DEFAULT["half_life_days"], DEFAULT["prior_scale"])]
        cache: dict = {}
        last_fit = None
        with open(OUT / f"results_{group}.jsonl", "a") as fh:
            for name, feats, sg, hl, pr in jobs:
                fit_key = (feats, hl, pr)
                if fit_key != last_fit:
                    cache.clear()  # keep memory small: fitted models are reused only across sigma variants
                    last_fit = fit_key
                cfg = BTConfig(features=feats, sigmas=sg, half_life_days=hl, prior_scale=pr, label=name, **VAL)
                t = time.time()
                df = run_backtest(bf, matches, st, cfg, fit_cache=cache)
                for r in _rows(name, pool, feats, cfg, df, time.time() - t):
                    fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
                print(f"[{group}] {pool} {name}: done in {time.time() - t:.0f}s", flush=True)


def merge() -> pd.DataFrame:
    rows = []
    for f in sorted(OUT.glob("results_*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    df = pd.DataFrame(rows).drop_duplicates(["label", "pool", "blend", "sigmas", "half_life_days", "prior_scale", "features"])
    df = df.sort_values("pts_xi", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "results.csv", index=False)
    return df


def leaderboard(df: pd.DataFrame, top: int = 15) -> str:
    sz = space_size()
    runs = df.groupby(["label", "pool", "sigmas", "half_life_days", "prior_scale", "features"]).ngroups
    L = [f"Possible combinations on paper: {sz['total_on_paper']:,}  "
         f"({sz['feature_switch_sets']} effect on/off sets x {sz['sigmas']} swing sizes x {sz['half_lives']} memory lengths x "
         f"{sz['priors']} pull-to-average strengths x {sz['data_pools']} data pools x {sz['team_pick_blends']} team-pick mixes)",
         f"Settings actually run: {runs}, each scored with {df['blend'].nunique()} team-pick mixes = {len(df)} recorded results", "",
         "| Rank | Setting | Data | Effects | Swing | Memory | Pull | Sim share | XI points | vs form (95% range) | Rank corr | Score err | Cover |",
         "|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for i, r in df.head(top).iterrows():
        eff = r["features"] if r["n_features"] < len(ALL_FEATURES) else "all"
        L.append(f"| {i + 1} | {r['label']} | {r['pool']} | {eff} | {r['sigmas']} | {r['half_life_days']:.0f} | {r['prior_scale']:g} | "
                 f"{r['blend']:.0%} | {r['pts_xi']:.1f} | {r['minus_form']:+.1f} ({r['ci_lo']:+.0f} to {r['ci_hi']:+.0f}) | "
                 f"{r['spearman']:.3f} | {r['score_mae']:.1f} | {r['coverage']:.0%} |")
    return "\n".join(L)


def final():
    df = merge()
    best = df.iloc[0]
    print(leaderboard(df))
    folders, only = POOL_FOLDERS[best["pool"]]
    feats = tuple(best["features"].split("+")) if best["features"] != "none" else ()
    sg = tuple(float(x) for x in best["sigmas"].strip("()").split(","))
    cfgd = dict(features=list(feats), sigmas=list(sg), half_life_days=float(best["half_life_days"]),
                prior_scale=float(best["prior_scale"]), pool=best["pool"], simulator_share=float(best["blend"]),
                chosen_on="IPL 2023-24 validation", validation_xi_points=float(best["pts_xi"]))
    balls, matches = load_many([DATA / f"{f}_json" for f in folders], full_members_only=only)
    bf, st = prepare(balls, matches)
    cfg = BTConfig(features=feats, sigmas=sg, half_life_days=cfgd["half_life_days"], prior_scale=cfgd["prior_scale"],
                   n_test=148, test_before=None, min_history=100, test_league="ipl", sims=1000, refit_every=25,
                   random_draws=10, label="final")
    df_t = run_backtest(bf, matches, st, cfg, progress=True)
    df_t.to_csv(OUT / "final_test_matches.csv", index=False)
    s = summarize(df_t)
    a = cfgd["simulator_share"]
    key = "pts_model" if a == 1.0 else f"pts_blend{int(a * 100)}"
    cfgd["test_xi_points"] = float(s[key] if a == 1.0 else s[key]["pts"])
    cfgd["test_form_points"] = float(s["pts_form"])
    Path("configs").mkdir(exist_ok=True)
    Path("configs/best.json").write_text(json.dumps(cfgd, indent=1))
    text = [f"# Final: best setting from {df['label'].nunique()} tried, tested on IPL 2025-26", "",
            "```json", json.dumps(cfgd, indent=1), "```", "", format_report(s, "Held-out test, best setting")]
    (OUT / "final_report.md").write_text("\n".join(text))
    print("\n".join(text))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["worker", "merge", "final", "space"])
    ap.add_argument("group", nargs="?", choices=["features", "hyperA", "hyperB", "pools"])
    a = ap.parse_args(argv)
    if a.cmd == "worker":
        worker(a.group)
    elif a.cmd == "merge":
        print(leaderboard(merge()))
    elif a.cmd == "space":
        print(json.dumps(space_size(), indent=1))
    else:
        final()


if __name__ == "__main__":
    main()
