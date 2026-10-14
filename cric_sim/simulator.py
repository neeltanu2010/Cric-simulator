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
from .model import RUNS_OF, TILT, RateModel

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
    bowl_weights: list[float] | None = None  # how likely each bowler is to bowl (e.g. usual overs per match)
    pos_hist: dict[str, list[float]] | None = None  # recent real batting positions (0 = opener) per player
    phase_overs: dict[str, tuple[float, float, float]] | None = None  # usual overs per match in PP / middle / death

    def __post_init__(self):
        if len(self.batting_order) != 11:
            raise ValueError(f"{self.name}: need exactly 11 players")
        if self.bowl_weights is not None and len(self.bowl_weights) != len(self.bowlers):
            raise ValueError(f"{self.name}: bowl_weights must match bowlers")
        if len(self.bowlers) < 5:
            raise ValueError(f"{self.name}: need at least 5 bowlers")

    @property
    def roster(self) -> list[Player]:
        return list(self.batting_order) + ([self.impact.player] if self.impact else [])


def _plans(bowl: list, weights: list[float] | None, explicit: list[str] | None, n: int,
           rng: np.random.Generator, random_lineup: bool) -> np.ndarray:
    """Which bowler (index into ``bowl``) bowls each over, for each simulation: shape (n, 20).
    A set plan if given; otherwise pick 5 (sometimes 6) bowlers at random, weighted by how much each
    usually bowls, and rotate them so nobody bowls twice in a row or more than 4 overs."""
    nb = len(bowl)
    names = [p.name for _, p in bowl]
    if explicit and len(explicit) == 20 and all(x in names for x in explicit):
        return np.tile([names.index(x) for x in explicit], (n, 1))
    if not random_lineup:
        return np.tile([o % nb for o in range(20)], (n, 1))
    w = np.array(weights if weights is not None else np.linspace(2.0, 1.0, nb), dtype=float).clip(min=0.05)
    key = np.log(w)[None, :] + rng.gumbel(size=(n, nb))  # weighted random order without repeats
    ranked = np.argsort(-key, axis=1)
    six = rng.random(n) < 0.3 if nb >= 6 else np.zeros(n, dtype=bool)
    k = np.where(six, 6, 5)
    plan = np.zeros((n, 20), dtype=int)
    for o in range(20):
        plan[:, o] = ranked[np.arange(n), o % k]
    return plan


def _phase_plans(bowl: list, phase_overs: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Over-by-over bowler choice for each simulation, from how much each bowler really bowls in the
    powerplay (overs 1-6), middle (7-16) and death (17-20). Max 4 overs each, never two in a row."""
    nb = len(bowl)
    typical = np.mean([v for v in phase_overs.values()], axis=0) if phase_overs else np.array([1.0, 1.5, 0.7])
    per_over = np.array([np.array(phase_overs.get(p.name, typical * 0.5), dtype=float) for _, p in bowl])
    per_over = per_over / np.array([6.0, 10.0, 4.0])  # chance-like weight per over in each phase
    per_over = per_over.clip(min=1e-3)
    plan = np.zeros((n, 20), dtype=int)
    used = np.zeros((n, nb))
    prev = np.full(n, -1)
    rows = np.arange(n)
    for o in range(20):
        ph = 0 if o < 6 else (1 if o < 16 else 2)
        w = np.tile(per_over[:, ph], (n, 1))
        w[used >= 4] = 0.0
        w[rows, prev.clip(min=0)] *= (prev < 0)  # no bowler twice in a row
        # make sure enough overs remain: a bowler who must bowl to finish the innings gets priority
        left = (4 - used).clip(min=0)
        dead = w.sum(axis=1) <= 0
        if dead.any():
            w[dead] = (left[dead] > 0).astype(float)
            w[dead, prev[dead].clip(min=0)] *= (prev[dead] < 0)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        pick = (rng.random(n)[:, None] > np.cumsum(w, axis=1)).sum(axis=1).clip(max=nb - 1)
        plan[:, o] = pick
        used[rows, pick] += 1
        prev = pick
    return plan


def _emp_order(bat: list, pos_hist: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Batting order for each simulation from each player's recent real positions."""
    nb = len(bat)
    keys = np.zeros((n, nb))
    for i, (_, p) in enumerate(bat):
        h = [x for x in (pos_hist.get(p.name) or []) if x == x]
        if len(h) >= 3:
            keys[:, i] = np.array(h)[rng.integers(0, len(h), size=n)]
        else:
            keys[:, i] = i + rng.normal(size=n) * (0.25 + 0.2 * i)
    keys += rng.random((n, nb)) * 0.5  # break ties at random
    return np.argsort(keys, axis=1)


def _order(nb: int, n: int, rng: np.random.Generator, random_lineup: bool) -> np.ndarray:
    """Batting order for each simulation as indices into the expected batting list, shape (n, nb).
    Openers rarely change; players lower down move around more."""
    base = np.tile(np.arange(nb), (n, 1))
    if not random_lineup:
        return base
    sd = 0.25 + 0.2 * np.arange(nb)
    return np.argsort(base + rng.normal(size=(n, nb)) * sd, axis=1)


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
             keeper: int, plan: np.ndarray, order: np.ndarray, n: int, rng: np.random.Generator, target,
             n_roster: tuple[int, int], league: str | None = None, cond: np.ndarray | None = None):
    nb = len(bat)
    nbowl = len(bowl)
    bat_slot = np.array([s for s, _ in bat])
    bowl_slot = np.array([s for s, _ in bowl])
    field_arr = np.array(field)

    # base outcome table: [batter][bowler][phase] -> 7 chances (batter index = place in expected order)
    base = np.zeros((nb, nbowl, 3, 7))
    for i, (_, b) in enumerate(bat):
        for j, (_, w) in enumerate(bowl):
            for ph in range(3):
                base[i, j, ph] = model.base_probs(b.name, w.name, ph, venue, inn, i, league)
    nb_tab, pr_tab = model.dynamic_tables()
    extras = np.array([[model.extra_rates(w.name, ph) for ph in range(3)] for _, w in bowl])  # [bowler][phase][3]
    kind_p = np.array([model.kind_p[model.bowl_type(w.name)] for _, w in bowl])
    p_ro = model.p_runout if model.has("extras") else np.zeros(3)
    use_extras = model.has("extras")
    # day-to-day form: each batter and bowler has a good or bad day, shared pitch conditions in ``cond``
    if model.has("form"):
        fb = rng.normal(size=(n, nb)) * model.sigmas[0]
        fw = rng.normal(size=(n, nbowl)) * model.sigmas[1]
    else:
        fb, fw = np.zeros((n, nb)), np.zeros((n, nbowl))
    cond = np.zeros(n) if cond is None else cond

    bs = _zeros(n, nb)  # batting stats by place in the expected order
    bo = _zeros(n, n_roster[1])  # bowling + fielding stats by roster slot of the fielding team
    idx = np.arange(n)
    striker = np.zeros(n, dtype=int)  # positions in this simulation's batting order
    non_striker = np.ones(n, dtype=int)
    next_in = np.full(n, 2)
    wkts = np.zeros(n, dtype=int)
    score_ = np.zeros(n, dtype=int)
    legal_balls = np.zeros(n, dtype=int)
    free_hit = np.zeros(n, dtype=bool)

    for over in range(20):
        j = plan[:, over]  # bowler index for each simulation
        bslot = bowl_slot[j]
        ph = 0 if over < 6 else (1 if over < 16 else 2)
        pw, pn, pb = extras[j, ph, 0], extras[j, ph, 1], extras[j, ph, 2]
        kp = np.cumsum(kind_p[j], axis=1)
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
            sb = order[idx, striker]  # who is on strike
            nsb = order[idx, non_striker]
            bf = bs["balls"][idx, sb]
            shift = cond + fb[idx, sb] - fw[idx, j] - model.shift0
            probs = base[sb, j, ph] * nb_tab[newbat_bucket(bf)] * pr_tab[pk] * np.exp(TILT[None, :] * shift[:, None])
            probs[:, 6] = np.where(~free_hit, probs[:, 6], 0.0)  # no bowler wicket on a free hit
            probs /= probs.sum(axis=1, keepdims=True)
            u = rng.random(n)
            is_wide = alive & (u < pw)
            is_nb = alive & ~is_wide & (u < pw + pn)
            is_bye = alive & ~is_wide & ~is_nb & (u < pw + pn + pb)
            normal = alive & ~is_wide & ~is_nb & ~is_bye
            pm = probs.copy()
            pm[is_nb, 6] = 0.0  # nor on a no-ball
            pm /= pm.sum(axis=1, keepdims=True)
            o = (rng.random(n)[:, None] > np.cumsum(pm, axis=1)).sum(axis=1).clip(max=6)
            faced = (normal | is_nb | is_bye)
            bat_runs = np.where(normal | is_nb, RUNS_OF[o], 0)
            wicket = normal & (o == 6)
            wide_runs = np.where(is_wide, _draw(rng, model.wide_runs_p, n), 0)
            bye_runs = np.where(is_bye, _draw(rng, model.bye_runs_p, n), 0)
            nb_runs = is_nb.astype(int)
            run_out = faced & ~wicket & (rng.random(n) < p_ro[ph]) if use_extras else np.zeros(n, dtype=bool)
            legal = normal | is_bye
            total = bat_runs + wide_runs + bye_runs + nb_runs

            np.add.at(bs["balls"], (idx, sb), faced.astype(float))
            np.add.at(bs["runs"], (idx, sb), bat_runs.astype(float))
            np.add.at(bs["fours"], (idx, sb), ((o == 4) & (normal | is_nb)).astype(float))
            np.add.at(bs["sixes"], (idx, sb), ((o == 5) & (normal | is_nb)).astype(float))
            charged = bat_runs + wide_runs + nb_runs  # runs the bowler concedes (not byes)
            np.add.at(bo["bowl_runs"], (idx, bslot), charged.astype(float))
            np.add.at(bo["bowl_balls"], (idx, bslot), legal.astype(float))
            np.add.at(bo["dots"], (idx, bslot), (legal & (charged == 0)).astype(float))
            conceded += charged
            score_ += total
            lb += legal
            legal_balls += legal
            free_hit = is_nb | (free_hit & is_wide)  # free hit after a no-ball; a wide does not use it up

            dismissed_striker = wicket | (run_out & (rng.random(n) >= model.q_nonstriker))
            dismissed_ns = run_out & ~dismissed_striker
            if wicket.any():
                kind = (rng.random(n)[:, None] > kp).sum(axis=1).clip(max=2)
                np.add.at(bo["wkts"], (idx[wicket], bslot[wicket]), 1.0)
                m = wicket & (kind == 0)
                np.add.at(bo["bl_wkts"], (idx[m], bslot[m]), 1.0)
                fld = field_arr[rng.integers(0, len(field_arr), size=n)]
                m = wicket & (kind == 1)
                np.add.at(bo["catches"], (idx[m], fld[m]), 1.0)
                m = wicket & (kind == 2)
                bo["stumpings"][m, keeper] += 1.0
            if run_out.any():
                fld = field_arr[rng.integers(0, len(field_arr), size=n)]
                np.add.at(bo["run_outs"], (idx[run_out], fld[run_out]), 1.0)
            np.add.at(bs["out"], (idx[dismissed_striker], sb[dismissed_striker]), True)
            np.add.at(bs["out"], (idx[dismissed_ns], nsb[dismissed_ns]), True)

            swap = faced & ((bat_runs + bye_runs) % 2 == 1) & ~dismissed_striker & ~dismissed_ns
            wkts += dismissed_striker
            wkts += dismissed_ns
            new_s = dismissed_striker & (next_in < nb)
            striker = np.where(new_s, next_in, striker)
            next_in = next_in + new_s
            new_n = dismissed_ns & (next_in < nb)
            non_striker = np.where(new_n, next_in, non_striker)
            next_in = next_in + new_n
            striker, non_striker = (np.where(swap, non_striker, striker), np.where(swap, striker, non_striker))
        m = (lb == 6) & (conceded == 0)
        np.add.at(bo["maidens"], (idx[m], bslot[m]), 1.0)  # a maiden: six legal balls, no runs
        striker, non_striker = non_striker.copy(), striker.copy()

    # move batting stats from expected-order places to roster slots
    out = _zeros(n, n_roster[0])
    for key in ("runs", "balls", "fours", "sixes", "out"):
        out[key][:, bat_slot] = bs[key]
    return out, bo, score_, wkts


def _draw(rng, p: np.ndarray, n: int) -> np.ndarray:
    """Draw values 1..len(p) with chances p."""
    return 1 + (rng.random(n)[:, None] > np.cumsum(p)[None, :]).sum(axis=1).clip(max=len(p) - 1)


def _weights(team: Team, bowl: list) -> list[float] | None:
    """Usual-overs weights for the bowlers in ``bowl`` (the impact sub, who has none, gets a middling weight)."""
    if team.bowl_weights is None:
        return None
    w = {b.name: x for b, x in zip(team.bowlers, team.bowl_weights)}
    typical = float(np.mean(team.bowl_weights))
    return [w.get(p.name, typical) for _, p in bowl]


def _keeper(team: Team) -> int:
    return next((i for i, p in enumerate(team.batting_order) if p.role == "WK"), 0)


def _stats(bat_part, bo_part, played):
    s = {k: bat_part[k] for k in ("runs", "balls", "fours", "sixes", "out")}
    for k in ("wkts", "bl_wkts", "dots", "maidens", "bowl_runs", "bowl_balls", "catches", "stumpings", "run_outs"):
        s[k] = bo_part[k]
    s["played"] = played
    return s


def calibrate_swings(model: RateModel, n: int = 1500) -> float:
    """Find the offset that makes the average score with day-to-day swings equal the average without them.
    Uses an average team (players the model has never seen), so it depends only on the swing sizes."""
    key = (tuple(model.sigmas), tuple(sorted(model.features)))
    if model._calibrated_for == key:
        return model.shift0
    mk = lambda tag: Team(tag, [Player(f"{tag}{i}", ["WK", "BAT", "BAT", "BAT", "AR", "AR", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL"][i])
                                for i in range(11)], [Player(f"{tag}{i}", "BOWL") for i in range(5, 11)])
    run = lambda: simulate(model, mk("x"), mk("y"), n, 12345)["avg_score_a"] + 0.0
    saved, model.mean_preserving, model.shift0 = model.sigmas, False, 0.0
    model.sigmas = (0.0, 0.0, 0.0)
    target = run()
    model.sigmas = saved
    d0, f0 = 0.0, run() - target
    d1 = 0.15
    model.shift0 = d1
    f1 = run() - target
    for _ in range(3):  # secant steps
        if abs(f1 - f0) < 1e-9:
            break
        d2 = float(np.clip(d1 - f1 * (d1 - d0) / (f1 - f0), -1.5, 1.5))
        d0, f0, d1 = d1, f1, d2
        model.shift0 = d1
        f1 = run() - target
    model.mean_preserving = True
    model._calibrated_for = key
    return model.shift0


def simulate(model: RateModel, team_a: Team, team_b: Team, n_sims: int = 5000, seed: int = 0,
             venue: str | None = None, rules: dict = T20_RULES, league: str | None = None) -> dict:
    """Simulate with team A batting first. Returns points for every player and win chances."""
    if model.mean_preserving and model.has("form") and model._calibrated_for != (tuple(model.sigmas), tuple(sorted(model.features))):
        calibrate_swings(model)
    rng = np.random.default_rng(seed)
    use_imp = model.has("impact")
    # a sub can only matter if the innings order suits it: bowl-subs for the team batting first,
    # bat-subs for the chasing team
    pa = team_a.impact.p_used if (use_imp and team_a.impact and team_a.impact.mode == "bowl") else 0.0
    pb = team_b.impact.p_used if (use_imp and team_b.impact and team_b.impact.mode == "bat") else 0.0
    counts = rng.multinomial(n_sims, [(1 - pa) * (1 - pb), pa * (1 - pb), (1 - pa) * pb, pa * pb])
    ra, rb = len(team_a.roster), len(team_b.roster)
    pts_a, pts_b, a_scores, b_scores = [], [], [], []
    parts_a, parts_b = [], []

    for g, cnt in enumerate(counts):
        if cnt == 0:
            continue
        ua, ub = bool(g & 1), bool(g & 2)
        a_bat, _, a_bowl, a_field = _lineups(team_a, ua, True)
        _, b_bat, b_bowl, b_field = _lineups(team_b, ub, False)
        # innings 1: A bats, B bowls and fields (its starting XI)
        rl = model.has("lineup")
        cond = rng.normal(size=cnt) * model.sigmas[2] if model.has("form") else np.zeros(cnt)
        plan_b = (_phase_plans(b_bowl, team_b.phase_overs, cnt, rng) if rl and team_b.phase_overs and not team_b.bowl_plan
                  else _plans(b_bowl, _weights(team_b, b_bowl), team_b.bowl_plan, cnt, rng, rl))
        ord_a = _emp_order(a_bat, team_a.pos_hist, cnt, rng) if rl and team_a.pos_hist else _order(len(a_bat), cnt, rng, rl)
        bs1, bo1, sc1, _ = _innings(model, venue, 1, a_bat, b_bowl, b_field, _keeper(team_b), plan_b,
                                    ord_a, cnt, rng, None, (ra, rb), league, cond)
        # innings 2: B chases (maybe with its impact batter), A bowls (maybe with its impact bowler)
        plan_a = (_phase_plans(a_bowl, team_a.phase_overs, cnt, rng) if rl and team_a.phase_overs and not team_a.bowl_plan
                  else _plans(a_bowl, _weights(team_a, a_bowl), team_a.bowl_plan, cnt, rng, rl))
        ord_b = _emp_order(b_bat, team_b.pos_hist, cnt, rng) if rl and team_b.pos_hist else _order(len(b_bat), cnt, rng, rl)
        bs2, bo2, sc2, _ = _innings(model, venue, 2, b_bat, a_bowl, a_field, _keeper(team_a), plan_a,
                                    ord_b, cnt, rng, sc1 + 1, (rb, ra), league, cond)
        played_a = np.ones((cnt, ra), dtype=bool)
        played_b = np.ones((cnt, rb), dtype=bool)
        if team_a.impact:
            played_a[:, 11] = ua
        if team_b.impact:
            played_b[:, 11] = ub
        ta_, pa_ = score(_stats(bs1, bo2, played_a), [p.role for p in team_a.roster], rules, parts=True)
        tb_, pb_ = score(_stats(bs2, bo1, played_b), [p.role for p in team_b.roster], rules, parts=True)
        pts_a.append(ta_); pts_b.append(tb_); parts_a.append(pa_); parts_b.append(pb_)
        a_scores.append(sc1)
        b_scores.append(sc2)

    pa_pts, pb_pts = np.vstack(pts_a), np.vstack(pts_b)
    comp = {k: np.hstack([np.vstack([x[k] for x in parts_a]), np.vstack([x[k] for x in parts_b])]) for k in parts_a[0]}
    sa, sb = np.concatenate(a_scores), np.concatenate(b_scores)
    return {
        "players": team_a.roster + team_b.roster,
        "points": np.hstack([pa_pts, pb_pts]),
        "parts": comp,  # per simulation: base / bat / bowl / field points
        "p_a_wins": float((sa > sb).mean()),
        "p_b_wins": float((sb > sa).mean()),
        "avg_score_a": float(sa.mean()),
        "avg_score_b": float(sb.mean()),
    }
