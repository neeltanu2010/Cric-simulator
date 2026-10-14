"""Ball-by-ball Monte Carlo simulation of a T20 match.

All simulations run together as numpy arrays, so thousands of matches take seconds.

What is modelled
- Every batter vs bowler matchup, by phase (powerplay / middle / death), ground, innings,
  pace vs spin, new batter, and match situation (required rate and wickets).
- Extras: wides, no-balls (with a free hit next ball, no bowler wicket) and byes / leg byes.
- Run outs, and how wickets happen (bowled/lbw, caught, stumped) by bowler type.
- Impact player substitutions (IPL): a bowler can replace a batter who has batted, or a batter
  can replace a bowler who has bowled. Whether the sub is used is random per simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dream11 import T20_RULES, score
from .features import newbat_bucket, pressure_key
from .model import RUNS_OF, RateModel

BIG = 99


@dataclass
class Player:
    name: str
    role: str  # WK / BAT / AR / BOWL
    credits: float = 8.0
    team: str = ""


@dataclass
class Impact:
    """An impact-player sub. mode 'bowl': comes on when the team bowls second, replacing
    ``replaces`` (a batter who has already batted). mode 'bat': comes in when the team
    chases, taking the batting place of ``replaces`` (a bowler who has already bowled)."""
    player: Player
    mode: str
    replaces: str
    p_used: float = 0.8


@dataclass
class Team:
    name: str
    batting_order: list[Player]  # 11 players, in expected batting order
    bowlers: list[Player]  # 5+ players from the XI who bowl, in bowling order
    impact: Impact | None = None
    bowl_plan: list[str] | None = None  # optional: bowler name for each of the 20 overs

    def __post_init__(self):
        if len(self.batting_order) != 11:
            raise ValueError(f"{self.name}: need exactly 11 players")
        if len(self.bowlers) < 5:
            raise ValueError(f"{self.name}: need at least 5 bowlers")

    @property
    def roster(self) -> list[Player]:
        return list(self.batting_order) + ([self.impact.player] if self.impact else [])


def _plan(bowlers: list[Player], explicit: list[str] | None) -> list[int]:
    if explicit and len(explicit) == 20:
        names = [b.name for b in bowlers]
        if all(n in names for n in explicit):
            return [names.index(n) for n in explicit]
    return [o % len(bowlers) for o in range(20)]


def _zeros(n: int, k: int) -> dict[str, np.ndarray]:
    keys = ("runs", "balls", "fours", "sixes", "wkts", "bl_wkts", "dots", "maidens", "bowl_runs",
            "bowl_balls", "catches", "stumpings", "run_outs")
    z = {key: np.zeros((n, k)) for key in keys}
    z["out"] = np.zeros((n, k), dtype=bool)
    return z


def _lineups(team: Team, uses_impact: bool, batting_first: bool):
    """Who bats, bowls and fields in each innings. Returns (bat, bowl_for_innings2, field_for_innings2).
    Slots: 0-10 = starting XI (by batting order), 11 = impact player."""
    roster_names = [p.name for p in team.batting_order]
    bat = [(i, p) for i, p in enumerate(team.batting_order)]
    bowl = [(roster_names.index(b.name), b) for b in team.bowlers]
    field = list(range(11))
    bat_chase = bat
    if uses_impact and team.impact:
        im, rep = team.impact, team.impact.replaces
        rep_slot = roster_names.index(rep) if rep in roster_names else None
        if im.mode == "bowl" and batting_first and rep_slot is not None:
            # innings 2: sub bowls, the replaced batter is off the field
            bowl = [(s, p) for s, p in bowl if s != rep_slot] + [(11, im.player)]
            field = [s for s in range(11) if s != rep_slot] + [11]
        elif im.mode == "bat" and not batting_first and rep_slot is not None:
            # innings 2: sub bats in the replaced bowler's place
            bat_chase = [(11 if s == rep_slot else s, im.player if s == rep_slot else p) for s, p in bat]
    return bat, bat_chase, bowl, field


def _innings(model: RateModel, venue: str | None, inn: int, bat: list, bowl: list, field: list,
             keeper: int, plan: list[int], n: int, rng: np.random.Generator, target, n_roster: tuple[int, int]):
    nb = len(bat)
    nbowl = len(bowl)
    bat_slot = np.array([s for s, _ in bat])
    bowl_slot = np.array([s for s, _ in bowl])
    field_arr = np.array(field)

    # base outcome table: [batter position][bowler][phase] -> 7 chances
    base = np.zeros((nb, nbowl, 3, 7))
    for i, (_, b) in enumerate(bat):
        for j, (_, w) in enumerate(bowl):
            for ph in range(3):
                base[i, j, ph] = model.base_probs(b.name, w.name, ph, venue, inn, i)
    nb_tab, pr_tab = model.dynamic_tables()
    extras = np.array([[model.extra_rates(w.name, ph) for ph in range(3)] for _, w in bowl])  # [bowler][phase][3]
    kind_p = np.array([model.kind_p[model.bowl_type(w.name)] for _, w in bowl])
    p_ro = model.p_runout if model.has("extras") else np.zeros(3)
    use_extras = model.has("extras")

    bs = _zeros(n, nb)  # batting stats by batting position
    bo = _zeros(n, n_roster[1])  # bowling + fielding stats by roster slot of the fielding team
    idx = np.arange(n)
    striker = np.zeros(n, dtype=int)
    non_striker = np.ones(n, dtype=int)
    next_in = np.full(n, 2)
    wkts = np.zeros(n, dtype=int)
    score_ = np.zeros(n, dtype=int)
    legal_balls = np.zeros(n, dtype=int)
    free_hit = np.zeros(n, dtype=bool)

    for over in range(20):
        j = plan[over]
        bslot = bowl_slot[j]
        ph = 0 if over < 6 else (1 if over < 16 else 2)
        pw, pn, pb = extras[j, ph]
        lb = np.zeros(n, dtype=int)  # legal balls bowled this over
        conceded = np.zeros(n)
        for _attempt in range(14):
            alive = (wkts < 10) & (lb < 6)
            if target is not None:
                alive &= score_ < target
            if not alive.any():
                break
            left = np.maximum(120 - legal_balls, 1)
            if target is None:
                pk = pressure_key(np.zeros(n), wkts, False)
            else:
                pk = pressure_key((target - score_) / left * 6, wkts, True)
            bf = bs["balls"][idx, striker]
            probs = base[striker, j, ph] * nb_tab[newbat_bucket(bf)] * pr_tab[pk]
            nb_ball = ~free_hit  # on a free hit (and on no-balls) the batter cannot be bowled/caught out
            probs[:, 6] = np.where(nb_ball, probs[:, 6], 0.0)
            probs /= probs.sum(axis=1, keepdims=True)
            u = rng.random(n)
            is_wide = alive & (u < pw)
            is_nb = alive & ~is_wide & (u < pw + pn)
            is_bye = alive & ~is_wide & ~is_nb & (u < pw + pn + pb)
            normal = alive & ~is_wide & ~is_nb & ~is_bye
            # outcome off the bat (no-ball: batter can't be dismissed by the bowler)
            pm = probs.copy()
            pm[is_nb, 6] = 0.0
            pm /= pm.sum(axis=1, keepdims=True)
            o = (rng.random(n)[:, None] > np.cumsum(pm, axis=1)).sum(axis=1).clip(max=6)
            faced = (normal | is_nb | is_bye)
            bat_runs = np.where(normal | is_nb, RUNS_OF[o], 0)
            wicket = normal & (o == 6)
            wide_runs = np.where(is_wide, _draw(rng, model.wide_runs_p, n), 0)
            bye_runs = np.where(is_bye, _draw(rng, model.bye_runs_p, n), 0)
            nb_runs = is_nb.astype(int)
            run_out = faced & ~wicket & (rng.random(n) < p_ro[ph])
            if not use_extras:
                run_out &= False
            legal = normal | is_bye
            total = bat_runs + wide_runs + bye_runs + nb_runs

            a = faced.astype(float)
            np.add.at(bs["balls"], (idx, striker), a)
            np.add.at(bs["runs"], (idx, striker), bat_runs.astype(float))
            np.add.at(bs["fours"], (idx, striker), ((o == 4) & (normal | is_nb)).astype(float))
            np.add.at(bs["sixes"], (idx, striker), ((o == 5) & (normal | is_nb)).astype(float))
            charged = bat_runs + wide_runs + nb_runs  # runs the bowler concedes (not byes)
            bo["bowl_runs"][:, bslot] += charged
            bo["bowl_balls"][:, bslot] += legal
            bo["dots"][:, bslot] += legal & (charged == 0)
            conceded += charged
            score_ += total
            lb += legal
            legal_balls += legal
            free_hit = is_nb | (free_hit & is_wide)  # free hit after a no-ball; a wide does not use it up

            dismissed_striker = wicket | (run_out & (rng.random(n) >= model.q_nonstriker))
            dismissed_ns = run_out & ~dismissed_striker
            if wicket.any():
                kind = rng.choice(3, size=n, p=kind_p[j])
                bo["wkts"][wicket, bslot] += 1
                bo["bl_wkts"][wicket & (kind == 0), bslot] += 1
                fld = field_arr[rng.integers(0, len(field_arr), size=n)]
                m = wicket & (kind == 1)
                np.add.at(bo["catches"], (idx[m], fld[m]), 1.0)
                m = wicket & (kind == 2)
                bo["stumpings"][m, keeper] += 1.0
            if run_out.any():
                fld = field_arr[rng.integers(0, len(field_arr), size=n)]
                np.add.at(bo["run_outs"], (idx[run_out], fld[run_out]), 1.0)
            np.add.at(bs["out"], (idx[dismissed_striker], striker[dismissed_striker]), True)
            np.add.at(bs["out"], (idx[dismissed_ns], non_striker[dismissed_ns]), True)

            swap = faced & ((bat_runs + bye_runs) % 2 == 1) & ~dismissed_striker & ~dismissed_ns
            wkts += dismissed_striker
            wkts += dismissed_ns
            can = next_in < nb
            new_s = dismissed_striker & can
            striker = np.where(new_s, next_in, striker)
            next_in = next_in + new_s
            can = next_in < nb
            new_n = dismissed_ns & can
            non_striker = np.where(new_n, next_in, non_striker)
            next_in = next_in + new_n
            striker, non_striker = (np.where(swap, non_striker, striker), np.where(swap, striker, non_striker))
        bo["maidens"][:, bslot] += (lb == 6) & (conceded == 0)
        striker, non_striker = non_striker.copy(), striker.copy()

    # move batting stats from batting-position order to roster slots
    out = _zeros(n, n_roster[0])
    for key in ("runs", "balls", "fours", "sixes", "out"):
        out[key][:, bat_slot] = bs[key]
    return out, bo, score_, wkts


def _draw(rng, p: np.ndarray, n: int) -> np.ndarray:
    """Draw values 1..len(p) with chances p."""
    return 1 + (rng.random(n)[:, None] > np.cumsum(p)[None, :]).sum(axis=1).clip(max=len(p) - 1)


def _keeper(team: Team) -> int:
    return next((i for i, p in enumerate(team.batting_order) if p.role == "WK"), 0)


def _stats(bat_part, bo_part, played):
    s = {k: bat_part[k] for k in ("runs", "balls", "fours", "sixes", "out")}
    for k in ("wkts", "bl_wkts", "dots", "maidens", "bowl_runs", "bowl_balls", "catches", "stumpings", "run_outs"):
        s[k] = bo_part[k]
    s["played"] = played
    return s


def simulate(model: RateModel, team_a: Team, team_b: Team, n_sims: int = 5000, seed: int = 0,
             venue: str | None = None, rules: dict = T20_RULES) -> dict:
    """Simulate with team A batting first. Returns points for every player and win chances."""
    rng = np.random.default_rng(seed)
    use_imp = model.has("impact")
    # a sub can only matter if the innings order suits it: bowl-subs for the team batting first,
    # bat-subs for the chasing team
    pa = team_a.impact.p_used if (use_imp and team_a.impact and team_a.impact.mode == "bowl") else 0.0
    pb = team_b.impact.p_used if (use_imp and team_b.impact and team_b.impact.mode == "bat") else 0.0
    counts = rng.multinomial(n_sims, [(1 - pa) * (1 - pb), pa * (1 - pb), (1 - pa) * pb, pa * pb])
    ra, rb = len(team_a.roster), len(team_b.roster)
    pts_a, pts_b, a_scores, b_scores = [], [], [], []

    for g, cnt in enumerate(counts):
        if cnt == 0:
            continue
        ua, ub = bool(g & 1), bool(g & 2)
        a_bat, _, a_bowl, a_field = _lineups(team_a, ua, True)
        _, b_bat, b_bowl, b_field = _lineups(team_b, ub, False)
        # innings 1: A bats, B bowls and fields (its starting XI)
        plan_b = _plan([p for _, p in b_bowl], team_b.bowl_plan)
        bs1, bo1, sc1, _ = _innings(model, venue, 1, a_bat, b_bowl, b_field, _keeper(team_b), plan_b,
                                    cnt, rng, None, (ra, rb))
        # innings 2: B chases (maybe with its impact batter), A bowls (maybe with its impact bowler)
        plan_a = _plan([p for _, p in a_bowl], team_a.bowl_plan)
        bs2, bo2, sc2, _ = _innings(model, venue, 2, b_bat, a_bowl, a_field, _keeper(team_a), plan_a,
                                    cnt, rng, sc1 + 1, (rb, ra))
        played_a = np.ones((cnt, ra), dtype=bool)
        played_b = np.ones((cnt, rb), dtype=bool)
        if team_a.impact:
            played_a[:, 11] = ua
        if team_b.impact:
            played_b[:, 11] = ub
        pts_a.append(score(_stats(bs1, bo2, played_a), [p.role for p in team_a.roster], rules))
        pts_b.append(score(_stats(bs2, bo1, played_b), [p.role for p in team_b.roster], rules))
        a_scores.append(sc1)
        b_scores.append(sc2)

    pa_pts, pb_pts = np.vstack(pts_a), np.vstack(pts_b)
    sa, sb = np.concatenate(a_scores), np.concatenate(b_scores)
    return {
        "players": team_a.roster + team_b.roster,
        "points": np.hstack([pa_pts, pb_pts]),
        "p_a_wins": float((sa > sb).mean()),
        "p_b_wins": float((sb > sa).mean()),
        "avg_score_a": float(sa.mean()),
        "avg_score_b": float(sb.mean()),
    }
