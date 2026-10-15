"""Add matches that exist only as published scorecards (no ball-by-ball data yet).

A scorecard gives each player's runs, balls, fours, sixes, dismissal, overs, runs conceded and wickets.
That is enough for the parts of the model that work per match: recent form, batting positions, roles
and estimated credits. The ball-by-ball ratings are NOT updated from scorecards (there are no balls).
Missing fours/sixes are estimated from the player's usual boundary share, and dot balls from economy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import venue_key

# dot-ball share of a spell vs runs per ball, fitted on 11,637 T20 spells (2023-2026)
DOT_A, DOT_B = 0.617, -0.171
FOUR_SHARE, SIX_SHARE = 0.349, 0.276  # share of T20 runs that come from fours / sixes (2023-2026)


def _balls(overs: str | float) -> int:
    o = str(overs)
    whole, _, part = o.partition(".")
    return int(whole) * 6 + (int(part) if part else 0)


def _boundary_shares(stats: pd.DataFrame, name: str) -> tuple[float, float]:
    g = stats[(stats["name"] == name) & (stats["balls"] > 0)]
    r = g["runs"].sum()
    if r < 150:  # too little history: use the overall T20 shares
        return FOUR_SHARE, SIX_SHARE
    return float((g["fours"] * 4).sum() / r), float((g["sixes"] * 6).sum() / r)


def load_scorecards(paths: list[str | Path], history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (matches rows, per-player stats rows) in the same format as ``loader`` / ``actual_stats``.
    ``history`` = existing per-player stats, used only to estimate missing fours and sixes."""
    mrows, srows = [], []
    for path in paths:
        for m in json.loads(Path(path).read_text())["matches"]:
            date = pd.Timestamp(m["date"])
            vk = venue_key(m["venue"])
            for team, xi in m["xi"].items():
                bat = {r[0]: (i, r) for i, r in enumerate(m["batting"].get(team, []))}
                opp = [t for t in m["xi"] if t != team][0]
                bowl = {r[0]: r for r in m["bowling"].get(team, [])}
                for name in xi:
                    row = dict(match_id=m["match_id"], name=name, team=team, date=date, venue_key=vk, pos=np.nan,
                               runs=0, balls=0, fours=0, sixes=0, out=False, wkts=0, bl_wkts=0, dots=0, maidens=0,
                               bowl_runs=0, bowl_balls=0, catches=0, stumpings=0, run_outs=0, from_scorecard=True)
                    if name in bat:
                        i, (_, runs, balls, fours, sixes, out) = bat[name]
                        row.update(pos=i, runs=runs, balls=balls, out=bool(out))
                        if fours is None or sixes is None:
                            f4, f6 = _boundary_shares(history, name)
                            fours = int(round(runs * f4 / 4)) if fours is None else fours
                            sixes = int(round(runs * f6 / 6)) if sixes is None else sixes
                            while fours * 4 + sixes * 6 > runs and (fours or sixes):
                                if fours:
                                    fours -= 1
                                else:
                                    sixes -= 1
                        row.update(fours=fours, sixes=sixes)
                    if name in bowl:
                        _, overs, maidens, runs_c, wkts = bowl[name]
                        b = _balls(overs)
                        dots = int(round(b * min(max(DOT_A + DOT_B * runs_c / max(b, 1), 0.05), 0.75)))
                        row.update(bowl_balls=b, bowl_runs=runs_c, wkts=wkts, maidens=maidens, dots=dots,
                                   bl_wkts=m.get("bowled_lbw", {}).get(name, 0))
                    srows.append(row)
                _ = opp
            for name, kind in m.get("fielding", []):
                for r in srows:
                    if r["match_id"] == m["match_id"] and r["name"] == name:
                        key = {"catch": "catches", "stumping": "stumpings", "run_out": "run_outs"}[kind]
                        r[key] += 1
            a, b_ = m["team_a"], m["team_b"]
            mrows.append(dict(match_id=m["match_id"], date=date, venue=m["venue"], venue_key=vk, city=None,
                              season=str(date.year), team_a=a, team_b=b_, toss_winner=None, toss_decision=None,
                              winner=m.get("winner"), players_a=m["xi"][a], players_b=m["xi"][b_], replacements=[],
                              match_number=None, league=m.get("league", "t20s"), afternoon=False))
    return pd.DataFrame(mrows), pd.DataFrame(srows)


def add_scorecards(matches: pd.DataFrame, stats: pd.DataFrame, paths: list[str | Path] | None):
    """Append scorecard matches to ``matches`` and ``stats`` (skipping any match already present)."""
    if not paths:
        return matches, stats
    m2, s2 = load_scorecards(paths, stats)
    m2 = m2[~m2["match_id"].isin(set(matches["match_id"]))]
    s2 = s2[s2["match_id"].isin(set(m2["match_id"]))]
    if m2.empty:
        return matches, stats
    matches = pd.concat([matches, m2], ignore_index=True).sort_values(["date", "match_id"]).reset_index(drop=True)
    stats = pd.concat([stats, s2[stats.columns.intersection(s2.columns)]], ignore_index=True)
    return matches, stats
