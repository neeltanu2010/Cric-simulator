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


def estimate_credits(names: list[str], roles: dict[str, str], form: dict[str, float], role_mean: dict, glob: float):
    """Dream11 credits are not public history. Estimate from form rank inside this match:
    ~11 for the top player down to ~7 for the weakest."""
    vals = np.array([form.get(n, role_mean.get(roles.get(n, "BAT"), glob)) for n in names])
    pct = vals.argsort().argsort() / max(len(names) - 1, 1)
    cr = np.round((7 + 4 * pct ** 2) * 2) / 2
    return {n: float(c) for n, c in zip(names, cr)}


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
