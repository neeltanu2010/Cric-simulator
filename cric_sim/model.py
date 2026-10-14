"""Per-ball outcome model.

A ball has 7 outcomes: dot, 1, 2, 3, four, six, wicket (a wicket credited to the bowler;
run outs are modelled separately).

The chance of each outcome is the league average for that phase and bowler type,
multiplied by a set of effects (all "ratio to normal"):

    batter, batter-by-phase, batter-vs-pace/spin,
    bowler, bowler-by-phase,
    ground, ground-by-pace/spin, ground-by-innings (this is where toss / chase advantage lives),
    innings, batting position, new-batter-at-crease, game situation (required rate, wickets).

Every effect is shrunk toward "no effect" when there is little data, and the effects are
fitted by repeated passes so one effect does not steal credit from another.
Extras (wides, no-balls, leg byes), run outs and how wickets happen are modelled too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import N_PRESSURE
from .players import PACE, SPIN, UNKNOWN

OUTCOMES = ["0", "1", "2", "3", "4", "6", "W"]
RUNS_OF = np.array([0, 1, 2, 3, 4, 6, 0])
N_OUT = 7
ALL_FEATURES = ("lineup", "form", "venue", "type", "innings", "state", "extras", "impact", "league", "daynight")

# How a "good day" shifts the 7 outcomes (log-odds per unit): fewer dots and wickets, more runs.
# Used for match-to-match variation in a batter's form, a bowler's form, and pitch/conditions.
TILT = np.array([-0.35, 0.15, 0.15, 0.15, 0.6, 0.85, -0.7])


def _outcome_index(runs_batter: np.ndarray, bowler_wicket: np.ndarray) -> np.ndarray:
    o = np.zeros(len(runs_batter), dtype=int)
    o[runs_batter == 1] = 1
    o[runs_batter == 2] = 2
    o[runs_batter == 3] = 3
    o[(runs_batter == 4) | (runs_batter == 5)] = 4
    o[runs_batter >= 6] = 5
    o[bowler_wicket] = 6
    return o


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class RateModel:
    # strength of the pull toward "no effect", in balls
    PRIOR = {"batter": 120, "bat_phase": 90, "bat_type": 100, "bowler": 150, "bowl_phase": 120,
             "venue": 1500, "venue_type": 1500, "venue_inn": 1500, "innings": 50,
             "newbat": 50, "pos": 50, "pressure": 100, "league": 300, "daynight": 500}

    def __init__(self, features=ALL_FEATURES, half_life_days: float | None = 730.0, rounds: int = 3,
                 sigmas: tuple[float, float, float] = (0.25, 0.15, 0.12), prior_scale: float = 1.0,
                 mean_preserving: bool = False):
        """``sigmas`` = size of the day-to-day swing in (batter form, bowler form, pitch/conditions),
        in log-odds. Needed because a player is not exactly his average in every match."""
        self.features = set(features)
        self.sigmas = sigmas
        self.prior_scale = prior_scale  # >1 pulls batter and bowler ratings harder toward the average
        # Day-to-day swings are not neutral: a batter on a good day also stays in longer, so good days add
        # more runs than bad days remove (+11 runs per first innings on real IPL 2025-26). With this flag
        # on, the simulator finds the offset that cancels the drift before predicting. Off by default so the
        # recorded experiments stay reproducible.
        self.mean_preserving = mean_preserving
        self.shift0 = 0.0  # offset subtracted from every swing (set by ``simulator.calibrate_swings``)
        self._calibrated_for = None
        self.half_life_days = half_life_days
        self.rounds = rounds
        self.types: dict[str, int] = {}
        self.league = np.full((3, 3, N_OUT), 1 / N_OUT)  # [phase][bowler type][outcome]
        self.tables: dict[str, np.ndarray] = {}
        self.ids: dict[str, dict] = {}
        # extras and other events
        self.p_wide = np.zeros(3)
        self.p_noball = np.zeros(3)
        self.p_bye = np.zeros(3)
        self.wide_ratio: dict[str, float] = {}
        self.noball_ratio: dict[str, float] = {}
        self.wide_runs_p = np.array([1.0, 0, 0, 0, 0])
        self.bye_runs_p = np.array([1.0, 0, 0, 0])
        self.p_runout = np.zeros(3)
        self.q_nonstriker = 0.4
        self.kind_p = np.tile([0.3, 0.62, 0.08], (3, 1))  # [bowler type][bowled/lbw, caught, stumped]
        self.venue_stats: dict[str, dict] = {}

    def has(self, feature: str) -> bool:
        return feature in self.features

    # ------------------------------------------------------------------ fitting
    def fit(self, balls: pd.DataFrame, bowl_types: dict[str, int] | None = None) -> "RateModel":
        """``balls`` must already have the columns from ``features.add_features``."""
        self.types = dict(bowl_types or {})
        use_type = self.has("type")
        allb = balls
        if self.half_life_days:
            age = (allb["date"].max() - allb["date"]).dt.days.to_numpy()
            w_all = 0.5 ** (age / self.half_life_days)
        else:
            w_all = np.ones(len(allb))
        self._fit_extras(allb, w_all)

        keep = ~(allb["wide"] | (allb["bye"] > 0) | (allb["legbye"] > 0)).to_numpy()
        d = allb[keep]
        w = w_all[keep]
        O = _outcome_index(d["runs_batter"].to_numpy(), d["bowler_wicket"].to_numpy())
        ph = d["phase"].to_numpy()
        bt = d["bowler"].map(self.types).fillna(UNKNOWN).astype(int).to_numpy() if use_type \
            else np.full(len(d), UNKNOWN)
        n = len(d)

        # league rates [phase][bowler type]; the "unknown" row is the plain phase average
        for p in range(3):
            m = ph == p
            c = np.bincount(O[m], weights=w[m], minlength=N_OUT) + 1e-3
            self.league[p, UNKNOWN] = c / c.sum()
            for t in (PACE, SPIN):
                mm = m & (bt == t)
                c2 = np.bincount(O[mm], weights=w[mm], minlength=N_OUT) + 20 * self.league[p, UNKNOWN]
                self.league[p, t] = c2 / c2.sum()
        share = np.bincount(O, weights=w, minlength=N_OUT) + 1e-3
        share = share / share.sum()

        bat_names = sorted(d["batter"].unique())
        bowl_names = sorted(d["bowler"].unique())
        venues = sorted(d["venue_key"].unique())
        leagues = sorted(d["league"].unique()) if "league" in d.columns else ["all"]
        self.ids = {"league": {x: i for i, x in enumerate(leagues)},
                    "bat": {x: i for i, x in enumerate(bat_names)},
                    "bowl": {x: i for i, x in enumerate(bowl_names)},
                    "venue": {x: i for i, x in enumerate(venues)}}
        bi = d["batter"].map(self.ids["bat"]).to_numpy()
        wi = d["bowler"].map(self.ids["bowl"]).to_numpy()
        vi = d["venue_key"].map(self.ids["venue"]).to_numpy()
        inn = (d["innings"].to_numpy() - 1).clip(0, 1)
        nbat, nbowl, nven = len(bat_names), len(bowl_names), len(venues)

        li = d["league"].map(self.ids["league"]).to_numpy() if "league" in d.columns else np.zeros(n, dtype=int)
        spec: dict[str, tuple[np.ndarray, int]] = {
            "batter": (bi, nbat), "bat_phase": (bi * 3 + ph, nbat * 3),
            "bowler": (wi, nbowl), "bowl_phase": (wi * 3 + ph, nbowl * 3),
        }
        if use_type:
            spec["bat_type"] = (bi * 3 + bt, nbat * 3)
        if self.has("venue"):
            spec["venue"] = (vi, nven)
            if use_type:
                spec["venue_type"] = (vi * 3 + bt, nven * 3)
        if self.has("daynight") and "afternoon" in d.columns:
            spec["daynight"] = (d["afternoon"].to_numpy().astype(int) * 2 + inn, 4)
        if self.has("league") and len(leagues) > 1:
            spec["league"] = (li, len(leagues))
        if self.has("innings"):
            spec["innings"] = (inn, 2)
            if self.has("venue"):
                spec["venue_inn"] = (vi * 2 + inn, nven * 2)
        if self.has("state"):
            spec["newbat"] = (d["nb_bucket"].to_numpy(), 4)
            spec["pos"] = (d["pos"].to_numpy(), 11)
            spec["pressure"] = (d["pkey"].to_numpy(), N_PRESSURE)

        logT = {k: np.zeros((g, N_OUT)) for k, (_, g) in spec.items()}
        tot = np.log(self.league[ph, bt])  # (n, 7)
        for _ in range(self.rounds):
            for name, (gid, G) in spec.items():
                base = tot - logT[name][gid]
                P = _softmax(base)
                obs = np.stack([np.bincount(gid, weights=w * (O == o), minlength=G) for o in range(N_OUT)], axis=1)
                exp = np.stack([np.bincount(gid, weights=w * P[:, o], minlength=G) for o in range(N_OUT)], axis=1)
                k = self.prior_scale if name in ("batter", "bat_phase", "bat_type", "bowler", "bowl_phase") else 1.0
                prior = self.PRIOR[name] * k * share
                logT[name] = np.log((obs + prior) / (exp + prior))
                tot = base + logT[name][gid]
        self.tables = {k: np.exp(v) for k, v in logT.items()}
        self._fit_venue_stats(d, w)
        return self

    def _fit_extras(self, b: pd.DataFrame, w: np.ndarray) -> None:
        ph = b["phase"].to_numpy()
        wide = b["wide"].to_numpy()
        nb = b["noball"].to_numpy()
        bye = ((b["bye"] > 0) | (b["legbye"] > 0)).to_numpy()
        for p in range(3):
            m = ph == p
            tot = w[m].sum() + 1e-9
            self.p_wide[p] = (w[m] * wide[m]).sum() / tot
            self.p_noball[p] = (w[m] * nb[m]).sum() / tot
            self.p_bye[p] = (w[m] * bye[m]).sum() / tot
        for col, store, rate in (("wide", self.wide_ratio, self.p_wide), ("noball", self.noball_ratio, self.p_noball)):
            e = pd.DataFrame({"bowler": b["bowler"].to_numpy(), "ph": ph, "x": b[col].to_numpy() * w, "w": w})
            g = e.groupby("bowler").agg(obs=("x", "sum"))
            g["exp"] = e.assign(r=rate[ph] * w).groupby("bowler")["r"].sum()
            for name, r in g.iterrows():
                store[name] = (r["obs"] + 4.0) / (r["exp"] + 4.0)
        wr = b.loc[b["wide"], "wide_runs"].clip(1, 5).value_counts()
        arr = np.array([wr.get(i, 0) for i in range(1, 6)], dtype=float) + 0.5
        self.wide_runs_p = arr / arr.sum()
        by = (b["bye"] + b["legbye"])[bye].clip(1, 4).value_counts()
        arr = np.array([by.get(i, 0) for i in range(1, 5)], dtype=float) + 0.5
        self.bye_runs_p = arr / arr.sum()
        # run outs
        notwide = ~wide
        for p in range(3):
            m = (ph == p) & notwide
            self.p_runout[p] = (w[m] * b["runout"].to_numpy()[m]).sum() / (w[m].sum() + 1e-9)
        ro = b[b["runout"]]
        if len(ro):
            self.q_nonstriker = float(np.clip((ro["player_out"] != ro["batter"]).mean(), 0.05, 0.9))
        # how wickets happen, by bowler type
        bw = b[b["bowler_wicket"]].copy()
        grp = bw["kind"].map({"bowled": 0, "lbw": 0, "hit wicket": 0, "caught": 1,
                              "caught and bowled": 1, "stumped": 2})
        bt = bw["bowler"].map(self.types).fillna(UNKNOWN).astype(int) if self.has("type") \
            else pd.Series(UNKNOWN, index=bw.index)
        allk = np.bincount(grp.dropna().astype(int), minlength=3) + 1.0
        self.kind_p[UNKNOWN] = allk / allk.sum()
        for t in (PACE, SPIN):
            c = np.bincount(grp[bt == t].dropna().astype(int), minlength=3) + 15 * self.kind_p[UNKNOWN]
            self.kind_p[t] = c / c.sum()

    def _fit_venue_stats(self, d: pd.DataFrame, w: np.ndarray) -> None:
        """Readable summary per ground: how much it changes scoring vs the league."""
        if not self.has("venue") or "venue" not in self.tables:
            return
        league_rr = 6 * float((self.league[:, UNKNOWN].mean(axis=0) * RUNS_OF).sum())
        for v, i in self.ids["venue"].items():
            n = int((d["venue_key"] == v).sum())
            p = self.league[:, UNKNOWN].mean(axis=0) * self.tables["venue"][i]
            p = p / p.sum()
            self.venue_stats[v] = {"balls": n, "runs_per_over_vs_league": 6 * float((p * RUNS_OF).sum()) - league_rr,
                                   "wicket_rate_ratio": float(self.tables["venue"][i][6])}

    # --------------------------------------------------------------- prediction
    def bowl_type(self, name: str) -> int:
        return self.types.get(name, UNKNOWN) if self.has("type") else UNKNOWN

    def base_probs(self, batter: str, bowler: str, phase: int, venue: str | None, innings: int,
                   pos: int, league: str | None = None, afternoon: bool = False) -> np.ndarray:
        """Outcome chances before the in-match effects (new batter, game situation)."""
        bt = self.bowl_type(bowler)
        logp = np.log(self.league[phase, bt])
        T, ids = self.tables, self.ids
        bi, wi = ids["bat"].get(batter), ids["bowl"].get(bowler)
        if bi is not None:
            logp = logp + np.log(T["batter"][bi]) + np.log(T["bat_phase"][bi * 3 + phase])
            if "bat_type" in T:
                logp = logp + np.log(T["bat_type"][bi * 3 + bt])
        if wi is not None:
            logp = logp + np.log(T["bowler"][wi]) + np.log(T["bowl_phase"][wi * 3 + phase])
        vi = ids["venue"].get(venue) if venue else None
        if vi is not None:
            if "venue" in T:
                logp = logp + np.log(T["venue"][vi])
            if "venue_type" in T:
                logp = logp + np.log(T["venue_type"][vi * 3 + bt])
        li = ids["league"].get(league) if league else None
        if li is not None and "league" in T:
            logp = logp + np.log(T["league"][li])
        if "innings" in T:
            logp = logp + np.log(T["innings"][innings - 1])
            if vi is not None and "venue_inn" in T:
                logp = logp + np.log(T["venue_inn"][vi * 2 + innings - 1])
        if "pos" in T:
            logp = logp + np.log(T["pos"][min(pos, 10)])
        if "daynight" in T:
            logp = logp + np.log(T["daynight"][int(afternoon) * 2 + innings - 1])
        p = np.exp(logp - logp.max())
        return p / p.sum()

    def recent_pitch(self, recent: pd.DataFrame, S: float = 1500.0) -> np.ndarray:
        """Data version of a pitch report: how the ground has played in its last few matches compared with
        what this model expects there. ``recent`` = ball rows (with feature columns) at this ground from
        matches BEFORE the one being predicted. Returns 7 outcome ratios, shrunk toward 1 (S = prior balls)."""
        d = recent[~(recent["wide"] | (recent["bye"] > 0) | (recent["legbye"] > 0))]
        if d.empty:
            return np.ones(N_OUT)
        O = _outcome_index(d["runs_batter"].to_numpy(), d["bowler_wicket"].to_numpy())
        nb_tab, pr_tab = self.dynamic_tables()
        exp = np.zeros(N_OUT)
        cache = {}
        for r in d.itertuples(index=False):
            key = (r.batter, r.bowler, r.phase, r.venue_key, r.innings, min(int(r.pos), 10), r.league, bool(getattr(r, "afternoon", False)))
            p = cache.get(key)
            if p is None:
                p = cache[key] = self.base_probs(r.batter, r.bowler, r.phase, r.venue_key, r.innings, min(int(r.pos), 10),
                                                 r.league, bool(getattr(r, "afternoon", False)))
            q = p * nb_tab[r.nb_bucket] * pr_tab[r.pkey]
            exp += q / q.sum()
        obs = np.bincount(O, minlength=N_OUT).astype(float)
        share = self.league[:, UNKNOWN].mean(axis=0)
        return (obs + S * share) / (exp + S * share)

    def dynamic_tables(self) -> tuple[np.ndarray, np.ndarray]:
        """(new-batter ratios [4,7], game-situation ratios [N_PRESSURE,7]); ones if switched off."""
        nb = self.tables.get("newbat", np.ones((4, N_OUT)))
        pr = self.tables.get("pressure", np.ones((N_PRESSURE, N_OUT)))
        return nb, pr

    def extra_rates(self, bowler: str, phase: int) -> tuple[float, float, float]:
        """(wide, no-ball, bye/leg-bye) chance per delivery."""
        if not self.has("extras"):
            return 0.0, 0.0, 0.0
        pw = min(0.2, self.p_wide[phase] * self.wide_ratio.get(bowler, 1.0))
        pn = min(0.1, self.p_noball[phase] * self.noball_ratio.get(bowler, 1.0))
        return float(pw), float(pn), float(self.p_bye[phase])

    def known(self, name: str) -> bool:
        return name in self.ids.get("bat", {}) or name in self.ids.get("bowl", {})
