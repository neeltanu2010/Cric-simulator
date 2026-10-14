"""Ball-by-ball Monte Carlo simulation of a T20 match.

All simulations run together as numpy arrays, so 5,000 matches take seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dream11 import T20_RULES, score
from .model import OUTCOMES, RUNS_OF, RateModel

# Chance a wicket is: bowled/lbw, caught, stumped, run out
DISMISSAL_P = np.array([0.29, 0.58, 0.05, 0.08])


@dataclass
class Player:
    name: str
    role: str  # WK / BAT / AR / BOWL
    credits: float = 8.0
    team: str = ""


@dataclass
class Team:
    name: str
    batting_order: list[Player]  # 11 players, in batting order
    bowlers: list[Player]  # 5+ players who bowl, in the order they start

    def __post_init__(self):
        if len(self.batting_order) != 11:
            raise ValueError(f"{self.name}: need exactly 11 players")
        if len(self.bowlers) < 5:
            raise ValueError(f"{self.name}: need at least 5 bowlers")


def _bowling_plan(n_bowlers: int) -> list[int]:
    """20 overs, bowlers rotate, nobody bowls two overs in a row or more than 4."""
    return [o % n_bowlers for o in range(20)]


def _stat_arrays(n: int, k: int) -> dict[str, np.ndarray]:
    return {key: np.zeros((n, k)) for key in ("runs", "balls", "fours", "sixes", "wkts", "bl_wkts",
                                              "dots", "maidens", "bowl_runs", "bowl_balls",
                                              "catches", "stumpings", "run_outs")} | {
        "out": np.zeros((n, k), dtype=bool)}


def _innings(model: RateModel, bat: Team, bowl: Team, n: int, rng: np.random.Generator,
             target: np.ndarray | None):
    bowlers = bowl.bowlers
    plan = _bowling_plan(len(bowlers))
    # probability table: [batter 0-10][bowler slot][phase] -> 7 cumulative chances
    table = np.zeros((11, len(bowlers), 3, len(OUTCOMES)))
    for i, b in enumerate(bat.batting_order):
        for j, w in enumerate(bowlers):
            for ph in range(3):
                table[i, j, ph] = np.cumsum(model.matchup(b.name, w.name, ph))

    bat_s = _stat_arrays(n, 11)
    bowl_idx = {p.name: i for i, p in enumerate(bowl.batting_order)}  # fielders / bowlers live here
    field_s = _stat_arrays(n, 11)
    keeper = next((i for i, p in enumerate(bowl.batting_order) if p.role == "WK"), 0)

    striker = np.zeros(n, dtype=int)
    non_striker = np.ones(n, dtype=int)
    next_in = np.full(n, 2)
    wkts = np.zeros(n, dtype=int)
    score_ = np.zeros(n, dtype=int)
    legal_balls = np.zeros(n, dtype=int)
    idx = np.arange(n)

    for over in range(20):
        slot = plan[over]
        bidx = bowl_idx[bowlers[slot].name]
        phase = 0 if over < 6 else (1 if over < 16 else 2)
        over_runs = np.zeros(n)
        over_balls = np.zeros(n)
        for _ in range(6):
            active = wkts < 10
            if target is not None:
                active &= score_ < target
            if not active.any():
                break
            cum = table[striker, slot, phase]
            o = (rng.random(n)[:, None] > cum).sum(axis=1).clip(max=len(OUTCOMES) - 1)
            runs = RUNS_OF[o] * active
            is_w = (o == 6) & active
            a = active.astype(float)

            np.add.at(bat_s["balls"], (idx, striker), a)
            np.add.at(bat_s["runs"], (idx, striker), runs)
            np.add.at(bat_s["fours"], (idx, striker), ((o == 4) & active).astype(float))
            np.add.at(bat_s["sixes"], (idx, striker), ((o == 5) & active).astype(float))
            field_s["bowl_balls"][:, bidx] += a
            field_s["bowl_runs"][:, bidx] += runs
            field_s["dots"][:, bidx] += ((o == 0) & active)
            over_runs += runs
            over_balls += a
            score_ += runs
            legal_balls += active

            # wickets
            if is_w.any():
                kind = rng.choice(4, size=n, p=DISMISSAL_P)
                np.add.at(bat_s["out"], (idx[is_w], striker[is_w]), True)
                bowler_credit = is_w & (kind < 3)
                field_s["wkts"][:, bidx] += bowler_credit
                field_s["bl_wkts"][:, bidx] += is_w & (kind == 0)
                fielder = rng.integers(0, 11, size=n)
                m = is_w & (kind == 1)
                np.add.at(field_s["catches"], (idx[m], fielder[m]), 1.0)
                m = is_w & (kind == 2)
                field_s["stumpings"][m, keeper] += 1.0
                m = is_w & (kind == 3)
                np.add.at(field_s["run_outs"], (idx[m], fielder[m]), 1.0)
                wkts += is_w
                new_ok = is_w & (next_in < 11)
                striker = np.where(new_ok, next_in, striker)
                next_in = next_in + new_ok

            # odd runs swap strike
            swap = active & (runs % 2 == 1) & ~is_w
            striker, non_striker = (np.where(swap, non_striker, striker),
                                    np.where(swap, striker, non_striker))
        # maiden: six legal balls and no runs
        field_s["maidens"][:, bidx] += (over_balls == 6) & (over_runs == 0)
        striker, non_striker = non_striker.copy(), striker.copy()  # end of over
    return bat_s, field_s, score_, wkts


def simulate(model: RateModel, team_a: Team, team_b: Team, n_sims: int = 5000, seed: int = 0,
             rules: dict = T20_RULES) -> dict:
    """Simulate A batting first, B chasing. Returns points per player and win chances."""
    rng = np.random.default_rng(seed)
    a_bat, b_field, a_score, _ = _innings(model, team_a, team_b, n_sims, rng, None)
    b_bat, a_field, b_score, _ = _innings(model, team_b, team_a, n_sims, rng, a_score + 1)

    def merge(bat, field):
        stats = {k: bat[k] for k in ("runs", "balls", "fours", "sixes", "out")}
        for k in ("wkts", "bl_wkts", "dots", "maidens", "bowl_runs", "bowl_balls",
                  "catches", "stumpings", "run_outs"):
            stats[k] = field[k]
        return stats

    # a player's batting stats come from their own innings; bowling/fielding from the other
    stats_a = merge(a_bat, a_field)
    stats_b = merge(b_bat, b_field)
    pts_a = score(stats_a, [p.role for p in team_a.batting_order], rules)
    pts_b = score(stats_b, [p.role for p in team_b.batting_order], rules)
    players = list(team_a.batting_order) + list(team_b.batting_order)
    return {
        "players": players,
        "points": np.hstack([pts_a, pts_b]),  # (n_sims, 22)
        "p_a_wins": float((a_score > b_score).mean()),
        "p_b_wins": float((b_score >= a_score + 1).mean()),
        "avg_score_a": float(a_score.mean()),
        "avg_score_b": float(b_score.mean()),
    }
