"""What really happened: per-player match stats and Dream11 points from ball-by-ball data.
Used as the "truth" in the backtest and as each player's form."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dream11 import T20_RULES, score

STAT_COLS = ["runs", "balls", "fours", "sixes", "out", "wkts", "bl_wkts", "dots", "maidens",
             "bowl_runs", "bowl_balls", "catches", "stumpings", "run_outs"]


def actual_stats(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per player per match."""
    b = balls
    legal = ~(b["wide"] | b["noball"])
    charged = b["runs_total"] - b["bye"] - b["legbye"]

    bat = b.assign(faced=(~b["wide"]).astype(int), four=(b["runs_batter"] == 4).astype(int),
                   six=(b["runs_batter"] >= 6).astype(int)) \
        .groupby(["match_id", "batter"]).agg(runs=("runs_batter", "sum"), balls=("faced", "sum"),
                                              fours=("four", "sum"), sixes=("six", "sum"))
    bat.index.names = ["match_id", "name"]
    # first-appearance position, for knowing who bats where
    pos = b.groupby(["match_id", "batter"])["pos"].first() if "pos" in b.columns else None
    outs = b[b["wicket"] & b["player_out"].notna()].groupby(["match_id", "player_out"]).size().rename("out")
    outs.index.names = ["match_id", "name"]

    bw = b.assign(charged=charged, legal=legal.astype(int),
                  dot=(legal & (charged == 0)).astype(int),
                  wk=b["bowler_wicket"].astype(int),
                  blwk=b["kind"].isin(["bowled", "lbw"]).astype(int))
    bowl = bw.groupby(["match_id", "bowler"]).agg(wkts=("wk", "sum"), bl_wkts=("blwk", "sum"), dots=("dot", "sum"),
                                                  bowl_runs=("charged", "sum"), bowl_balls=("legal", "sum"))
    bowl.index.names = ["match_id", "name"]
    ov = bw.groupby(["match_id", "innings", "over", "bowler"]).agg(c=("charged", "sum"), n=("legal", "sum")).reset_index()
    maid = ov[(ov["n"] == 6) & (ov["c"] == 0)].groupby(["match_id", "bowler"]).size().rename("maidens")
    maid.index.names = ["match_id", "name"]

    # fielding
    w = b[b["wicket"]]
    rows = []
    for mid, kind, bowler, fielders in zip(w["match_id"], w["kind"], w["bowler"], w["fielders"]):
        if kind == "caught" and fielders:
            rows.append((mid, fielders[0], "catches"))
        elif kind == "caught and bowled":
            rows.append((mid, bowler, "catches"))
        elif kind == "stumped" and fielders:
            rows.append((mid, fielders[0], "stumpings"))
        elif kind == "run out":
            for f in fielders[:2]:
                rows.append((mid, f, "run_outs"))
    if rows:
        fld = pd.DataFrame(rows, columns=["match_id", "name", "k"]).groupby(["match_id", "name", "k"]).size() \
            .unstack(fill_value=0)
    else:
        fld = pd.DataFrame(columns=["catches", "stumpings", "run_outs"])

    # everyone who played, from the players lists
    base = []
    for r in matches.itertuples():
        for team, names in ((r.team_a, r.players_a), (r.team_b, r.players_b)):
            for nme in names:
                base.append((r.match_id, nme, team, r.date, r.venue_key))
    df = pd.DataFrame(base, columns=["match_id", "name", "team", "date", "venue_key"]).set_index(["match_id", "name"])
    for part in (bat, outs, bowl, maid, fld):
        df = df.join(part, how="left")
    if pos is not None:
        df = df.join(pos.rename("pos").rename_axis(["match_id", "name"]), how="left")
    for c in STAT_COLS + [c for c in ("catches", "stumpings", "run_outs") if c not in STAT_COLS]:
        if c not in df.columns:
            df[c] = 0
    df[STAT_COLS] = df[STAT_COLS].fillna(0)
    df["out"] = df["out"] > 0
    return df.reset_index()


def points_of(stats: pd.DataFrame, roles: dict[str, str], rules: dict = T20_RULES) -> np.ndarray:
    """Dream11 points for every row of ``stats``, given a role per player name."""
    arr = {c: stats[c].to_numpy(dtype=float)[:, None] for c in STAT_COLS if c != "out"}
    arr["out"] = stats["out"].to_numpy()[:, None]
    r = [roles.get(n, "BAT") for n in stats["name"]]
    # score() takes one role per column; do it row-wise by scoring a (rows, 1) block per role group
    out = np.zeros(len(stats))
    roles_arr = np.array(r)
    for role in set(r):
        m = roles_arr == role
        sub = {k: v[m] for k, v in arr.items()}
        out[m] = score(sub, [role], rules)[:, 0]
    return out


def points_parts(stats: pd.DataFrame, roles: dict[str, str], rules: dict = T20_RULES) -> pd.DataFrame:
    """Real Dream11 points split into base / bat / bowl / field for every row of ``stats``."""
    arr = {c: stats[c].to_numpy(dtype=float)[:, None] for c in STAT_COLS if c != "out"}
    arr["out"] = stats["out"].to_numpy()[:, None]
    r = np.array([roles.get(n, "BAT") for n in stats["name"]])
    out = {k: np.zeros(len(stats)) for k in ("base", "bat", "bowl", "field")}
    for role in set(r):
        m = r == role
        _, parts = score({k: v[m] for k, v in arr.items()}, [role], rules, parts=True)
        for k in out:
            out[k][m] = parts[k][:, 0]
    return pd.DataFrame(out, index=stats.index)
