"""Pick the best Dream11 team from simulated points.

Rules used (Dream11): 11 players, at most 100 credits, at most 10 from one real team,
role limits (WK 1-4, BAT 3-6, AR 1-4, BOWL 3-6), one captain (2x points)
and one vice-captain (1.5x points).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROLE_LIMITS = {"WK": (1, 4), "BAT": (3, 6), "AR": (1, 4), "BOWL": (3, 6)}


def pick_team(players, points: np.ndarray, credit_cap: float = 100.0, max_per_team: int = 10,
              risk: float = 0.0, role_limits: dict | None = None, captain_value: np.ndarray | None = None,
              split_cv: bool = False):
    """Choose the best XI.

    ``points`` is (n_sims, n_players). ``risk`` sets what we maximise:
    0.0 = average points (safe, small leagues). Above 0 adds that many
    standard deviations, which favours high-ceiling players (big grand leagues).
    """
    limits = ROLE_LIMITS if role_limits is None else role_limits
    n = len(players)
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    value = mean + risk * std

    # variables: x (in team), c (captain), v (vice-captain). The captain slots can use their own value
    # (for example chance of being the match's top scorer, scaled to points).
    cap = value if captain_value is None else np.asarray(captain_value, dtype=float)
    cost = -np.concatenate([value, cap, 0.5 * cap])  # minimise negative
    rows, lo, hi = [], [], []

    def add(row, low, high):
        rows.append(row)
        lo.append(low)
        hi.append(high)

    z = np.zeros(n)
    add(np.concatenate([np.ones(n), z, z]), 11, 11)
    add(np.concatenate([[p.credits for p in players], z, z]), 0, credit_cap)
    add(np.concatenate([z, np.ones(n), z]), 1, 1)
    add(np.concatenate([z, z, np.ones(n)]), 1, 1)
    for i in range(n):  # captain / vice must be in the team and differ
        e = np.zeros(n)
        e[i] = 1
        add(np.concatenate([-e, e, z]), -np.inf, 0)
        add(np.concatenate([-e, z, e]), -np.inf, 0)
        add(np.concatenate([z, e, e]), 0, 1)
    for team in {p.team for p in players}:
        add(np.concatenate([[1.0 if p.team == team else 0.0 for p in players], z, z]), 0, max_per_team)
        if split_cv:  # captain and vice-captain from different teams
            t = np.array([1.0 if p.team == team else 0.0 for p in players])
            add(np.concatenate([z, t, t]), 0, 1)
    for role, (lo_r, hi_r) in limits.items():
        add(np.concatenate([[1.0 if p.role == role else 0.0 for p in players], z, z]), lo_r, hi_r)

    res = milp(c=cost, constraints=LinearConstraint(np.array(rows), lo, hi),
               integrality=np.ones(3 * n), bounds=Bounds(0, 1))
    if not res.success:
        if role_limits is None:
            # role guesses can be wrong (for example no wicket-keeper found); relax the limits
            return pick_team(players, points, credit_cap, max_per_team, risk,
                             {r: (0, 11) for r in ROLE_LIMITS}, captain_value, split_cv)
        raise RuntimeError(f"Could not build a valid team: {res.message}")
    x = np.round(res.x).astype(int)
    team = [i for i in range(n) if x[i]]
    cap = next(i for i in range(n) if x[n + i])
    vice = next(i for i in range(n) if x[2 * n + i])
    return {
        "team": [players[i] for i in team],
        "captain": players[cap],
        "vice_captain": players[vice],
        "credits": sum(players[i].credits for i in team),
        "expected_points": float(sum(mean[i] for i in team) + mean[cap] + 0.5 * mean[vice]),
        "means": {players[i].name: float(mean[i]) for i in range(n)},
    }
