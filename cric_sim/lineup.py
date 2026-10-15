"""Things you would know BEFORE a match, worked out from past matches only:
expected batting order, who bowls, each player's role, form, and estimated Dream11 credits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .actual import points_of


def history_table(stats: pd.DataFrame, roles: dict[str, str] | None = None, half_life_matches: float = 15.0):
    """Per-player summary from past matches. ``stats`` = actual_stats rows strictly before the match."""
    s = stats.sort_values("date")
    out = {}
    for name, g in s.groupby("name"):
        n = len(g)
        age = np.arange(n)[::-1]
        w = 0.5 ** (age / half_life_matches)
        batted = g["balls"] > 0
        out[name] = {
            "matches": n,
            "bowl_balls": float((g["bowl_balls"] * w).sum() / w.sum()),
            "bat_balls": float((g["balls"] * w).sum() / w.sum()),
            "pos": float((g.loc[batted, "pos"].fillna(7) * w[batted.to_numpy()]).sum() / max(w[batted.to_numpy()].sum(), 1e-9)) if batted.any() else 9.0,
            "stumpings": float(g["stumpings"].sum()),
            "team": g["team"].iloc[-1],
        }
    return out


def infer_roles(hist: dict[str, dict], overrides: dict[str, dict] | None = None) -> dict[str, str]:
    roles = {}
    for name, h in hist.items():
        bb, pos = h["bowl_balls"], h["pos"]
        if h["stumpings"] >= 3 and h["stumpings"] / max(h["matches"], 1) >= 0.1:
            roles[name] = "WK"
        elif bb >= 12:
            roles[name] = "AR" if (pos <= 7.5 and h["bat_balls"] >= 8) else "BOWL"
        else:
            roles[name] = "BAT"
    for name, o in (overrides or {}).items():
        if o.get("role"):
            roles[name] = o["role"]
    return roles


def form_points(stats: pd.DataFrame, roles: dict[str, str], decay: float = 0.9, prior_matches: float = 2.0):
    """Form baseline: recency-weighted average of past Dream11 points, shrunk toward the role average."""
    s = stats.sort_values("date").copy()
    s["pts"] = points_of(s, roles)
    role_mean = s.assign(role=s["name"].map(roles).fillna("BAT")).groupby("role")["pts"].mean().to_dict()
    glob = float(s["pts"].mean()) if len(s) else 30.0
    out = {}
    for name, g in s.groupby("name"):
        n = len(g)
        w = decay ** np.arange(n)[::-1]
        prior = role_mean.get(roles.get(name, "BAT"), glob)
        out[name] = float(((g["pts"].to_numpy() * w).sum() + prior * prior_matches) / (w.sum() + prior_matches))
    return out, role_mean, glob


def credits_from_rank(pct) -> np.ndarray:
    """Dream11-style credit bands from a 0-1 form rank inside the match (1 = best):
    premium 9.5-11.0 (top ~25%), mid-tier 8.5-9.0 (next ~35%), budget 7.0-8.0 (bottom ~40%)."""
    pct = np.asarray(pct, dtype=float)
    cr = np.where(pct >= 0.75, 9.5 + (pct - 0.75) / 0.25 * 1.5,
                  np.where(pct >= 0.40, 8.5 + (pct - 0.40) / 0.35 * 0.5, 7.0 + pct / 0.40 * 1.0))
    return np.clip(np.round(cr * 2) / 2, 7.0, 11.0)


def estimate_credits(names: list[str], roles: dict[str, str], form: dict[str, float], role_mean: dict, glob: float):
    """Real Dream11 credits are set per match and are not in any public history, so estimate them from each
    player's recent-form rank among the 22, using Dream11's bands (premium 9.5-11, mid 8.5-9, budget 7-8).
    Type the real credits in whenever you have them."""
    vals = np.array([form.get(n, role_mean.get(roles.get(n, "BAT"), glob)) for n in names])
    pct = vals.argsort().argsort() / max(len(names) - 1, 1)
    return {n: float(c) for n, c in zip(names, credits_from_rank(pct))}


def order_by_position(names: list[str], hist: dict[str, dict], roles: dict[str, str]) -> list[str]:
    default = {"WK": 5.0, "BAT": 4.0, "AR": 6.5, "BOWL": 9.5}
    return sorted(names, key=lambda n: hist[n]["pos"] if n in hist and hist[n]["bat_balls"] > 0
                  else default.get(roles.get(n, "BAT"), 7))


def pick_bowlers(names: list[str], hist: dict[str, dict], roles: dict[str, str], k: int = 6) -> list[str]:
    """The XI's most likely bowlers, in the order they are likely to start."""
    score_ = {n: hist[n]["bowl_balls"] if n in hist else (14.0 if roles.get(n) in ("BOWL", "AR") else 0.0)
              for n in names}
    ranked = sorted(names, key=lambda n: -score_[n])
    chosen = [n for n in ranked if score_[n] >= 6][:k]
    for n in ranked:
        if len(chosen) >= 5:
            break
        if n not in chosen:
            chosen.append(n)
    return chosen


def phase_overs_table(balls_f: pd.DataFrame) -> pd.DataFrame:
    """Overs bowled per match per bowler in powerplay / middle / death (columns 0, 1, 2)."""
    lg = balls_f[~(balls_f["wide"] | balls_f["noball"])]
    t = lg.groupby(["match_id", "bowler", "phase"]).size().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0) / 6.0
    return t.reset_index().rename(columns={"bowler": "name"})


def lineup_history(stats: pd.DataFrame, bph: pd.DataFrame, names: list[str], n_pos: int = 10, n_overs: int = 15,
                   team_of: dict[str, str] | None = None, same_team_pos: int = 5, same_team_overs: int = 10):
    """Recent real batting positions and usual overs per phase, for the realistic-lineup simulator.
    ``stats`` must only hold matches before the one being predicted. With ``team_of`` (player -> team), a
    player's recent matches FOR THIS TEAM are used when there are enough of them: a batter can open for his
    league side but bat at six for his country."""
    ps = stats[stats["name"].isin(names)].sort_values("date")
    pos_hist, phase_ov = {}, {}
    known = set(bph["match_id"])  # scorecard-only matches have no phase split: leave them out here
    for n, g in ps.groupby("name"):
        same = g[g["team"] == team_of[n]] if team_of and n in team_of else g.iloc[0:0]
        sp = same["pos"].dropna()
        pos_hist[n] = sp.tail(same_team_pos).tolist() if len(sp) >= 2 else g["pos"].dropna().tail(n_pos).tolist()
        sm = same.loc[same["match_id"].isin(known), "match_id"].tail(same_team_overs)
        mids = sm if len(sm) >= 3 else g.loc[g["match_id"].isin(known), "match_id"].tail(n_overs)
        b = bph[(bph["name"] == n) & bph["match_id"].isin(mids)]
        if len(mids):
            phase_ov[n] = tuple(b[[0, 1, 2]].sum().to_numpy() / len(mids))
    return pos_hist, phase_ov
