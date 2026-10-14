"""Dream11-style T20 scoring.

IMPORTANT: Dream11 changes its rules from time to time. The values below are
the commonly published T20 rules. Check them against the current Dream11 app
and edit ``T20_RULES`` if anything differs.
"""
from __future__ import annotations

import numpy as np

T20_RULES = {
    "playing_xi": 4,
    "run": 1,
    "four_bonus": 4,
    "six_bonus": 6,
    "bonus_30": 4,
    "bonus_50": 8,  # replaces the 30 bonus
    "bonus_100": 16,  # replaces the 50 bonus
    "duck": -2,  # not for pure bowlers
    "dot_ball": 1,
    "wicket": 25,  # not for run outs
    "bowled_lbw_bonus": 8,
    "wkts_3": 4,
    "wkts_4": 8,
    "wkts_5": 16,
    "maiden": 12,
    "catch": 8,
    "catches_3_bonus": 4,
    "stumping": 12,
    "run_out": 8,  # average of direct (12) and indirect (6) in the real rules
    # (min overs 2) economy bands: (upper_bound_exclusive, points)
    "economy": [(5, 6), (6, 4), (7, 2), (10, 0), (11, -2), (12, -4), (1e9, -6)],
    # (min balls 10, not for bowlers) strike rate bands: (lower_bound_inclusive, points)
    "strike_rate": [(170, 6), (150, 4), (130, 2), (70, 0), (60, -2), (50, -4), (-1, -6)],
}


def score(stats: dict[str, np.ndarray], roles: list[str], rules: dict = T20_RULES) -> np.ndarray:
    """Turn simulated match stats into Dream11 points.

    ``stats`` holds arrays of shape (n_sims, n_players). ``roles`` is one of
    WK / BAT / AR / BOWL per player (same player order).
    """
    r = rules
    runs, balls = stats["runs"], stats["balls"]
    played = stats.get("played", np.ones(runs.shape, dtype=bool))
    pts = np.full(runs.shape, float(r["playing_xi"]))

    # batting
    pts += runs * r["run"] + stats["fours"] * r["four_bonus"] + stats["sixes"] * r["six_bonus"]
    pts += np.where(runs >= 100, r["bonus_100"], np.where(runs >= 50, r["bonus_50"],
                    np.where(runs >= 30, r["bonus_30"], 0)))
    not_bowler = np.array([role != "BOWL" for role in roles])
    duck = stats["out"] & (runs == 0)
    pts += np.where(duck & not_bowler[None, :], r["duck"], 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(balls > 0, runs / np.maximum(balls, 1) * 100, 0)
    sr_pts = np.zeros(runs.shape)
    for lower, p in reversed(r["strike_rate"]):
        sr_pts = np.where(sr >= lower, p, sr_pts)
    pts += np.where((balls >= 10) & not_bowler[None, :], sr_pts, 0)

    # bowling
    w = stats["wkts"]
    pts += stats["dots"] * r["dot_ball"] + w * r["wicket"] + stats["bl_wkts"] * r["bowled_lbw_bonus"]
    pts += np.where(w >= 5, r["wkts_5"], np.where(w == 4, r["wkts_4"], np.where(w == 3, r["wkts_3"], 0)))
    pts += stats["maidens"] * r["maiden"]
    overs = stats["bowl_balls"] / 6.0
    with np.errstate(divide="ignore", invalid="ignore"):
        econ = np.where(overs > 0, stats["bowl_runs"] / np.maximum(overs, 1e-9), 0)
    e_pts = np.zeros(runs.shape)
    for upper, p in reversed(r["economy"]):
        e_pts = np.where(econ < upper, p, e_pts)
    pts += np.where(overs >= 2, e_pts, 0)

    # fielding
    c = stats["catches"]
    pts += c * r["catch"] + np.where(c >= 3, r["catches_3_bonus"], 0)
    pts += stats["stumpings"] * r["stumping"] + stats["run_outs"] * r["run_out"]
    return np.where(played, pts, 0.0)
