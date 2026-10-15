"""Several different teams for one match (Dream11 lets you enter more than one team in a contest).

Team 1 is the normal best pick. Each further team is built for a different plausible version of the match:
every player's value is redrawn from his predicted spread (value + spread x z x std, z ~ Normal(0, 1)), and the
best team for that version is picked. Duplicates are skipped.

Backtests (Dream11 rules, spread 0.3, chosen on IPL 2022-24 and checked on 2025-26): the best of 3 teams
reached 71.8% of the best possible team on IPL 2025-26, of 6 teams 74.7%, of 20 teams 78.9%, against 65.9%
for a single team. The average team in the set is lower (705 vs 753 points with 20 teams), because each one
bets on a different version of the match. Every extra team costs another entry fee.
"""
from __future__ import annotations

import numpy as np

from .optimizer import pick_team


def _key(t):
    return frozenset(p.name for p in t["team"]), t["captain"].name, t["vice_captain"].name


def diverse_teams(players, value: np.ndarray, captain_value: np.ndarray, std: np.ndarray, n: int = 20,
                  spread: float = 0.3, seed: int = 0, max_tries: int = 300, **pick_kw) -> list[dict]:
    rng = np.random.default_rng(seed)
    value, captain_value, std = (np.asarray(x, dtype=float) for x in (value, captain_value, std))
    teams = [pick_team(players, value[None, :], captain_value=captain_value, **pick_kw)]
    seen = {_key(teams[0])}
    tries = 0
    while len(teams) < n and tries < max_tries:
        tries += 1
        z = rng.standard_normal(len(players))
        keep_out = value < -100  # players left out on purpose stay out
        v = np.where(keep_out, value, np.maximum(value + spread * z * std, 0.0))
        c = np.where(keep_out, captain_value, np.maximum(captain_value + spread * z * std, 0.0))
        t = pick_team(players, v[None, :], captain_value=c, **pick_kw)
        if _key(t) not in seen:
            seen.add(_key(t))
            teams.append(t)
    return teams
