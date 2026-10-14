"""Add game-state columns to the ball table (score, wickets, required rate, batting position...)."""
from __future__ import annotations

import numpy as np
import pandas as pd

NEWBAT_EDGES = [6, 12, 25]  # balls faced -> bucket 0..3
RRR_EDGES = [5, 7, 9, 11, 13, 16]  # required run rate -> bucket 1..7 (0 = first innings)
N_PRESSURE = 8 * 3


def newbat_bucket(balls_faced):
    return np.searchsorted(NEWBAT_EDGES, balls_faced, side="right")


def pressure_key(rrr, wkts, chasing):
    """Game-state key: (required-rate bucket or 0 in first innings) x (wickets lost bucket)."""
    rb = np.where(chasing, 1 + np.searchsorted(RRR_EDGES, rrr, side="right"), 0)
    wb = np.searchsorted([3, 6], wkts, side="right")
    return rb * 3 + wb


def add_features(balls: pd.DataFrame) -> pd.DataFrame:
    d = balls.sort_values(["match_id", "innings", "over", "ball"], kind="stable").reset_index(drop=True)
    d["legal"] = ~(d["wide"] | d["noball"])
    g = d.groupby(["match_id", "innings"], sort=False)
    d["score_before"] = g["runs_total"].cumsum() - d["runs_total"]
    d["wkts_before"] = g["wicket"].cumsum() - d["wicket"].astype(int)
    d["legal_before"] = g["legal"].cumsum() - d["legal"].astype(int)

    totals = g["runs_total"].sum().rename("inn_total").reset_index()
    first = totals[totals["innings"] == 1].set_index("match_id")["inn_total"]
    d["target"] = np.where(d["innings"] == 2, d["match_id"].map(first) + 1, np.nan)
    left = (120 - d["legal_before"]).clip(lower=1)
    d["rrr"] = np.where(d["innings"] == 2, (d["target"] - d["score_before"]) / left * 6, 0.0)
    d["pkey"] = pressure_key(d["rrr"].to_numpy(), d["wkts_before"].to_numpy(), (d["innings"] == 2).to_numpy())

    # balls the striker has already faced this innings (wides are not faced)
    d["faced"] = (~d["wide"]).astype(int)
    d["bf_before"] = d.groupby(["match_id", "innings", "batter"], sort=False)["faced"].cumsum() - d["faced"]
    d["nb_bucket"] = newbat_bucket(d["bf_before"].to_numpy())

    # batting position = order of first appearance (as striker or non-striker)
    idx = np.arange(len(d))
    a = pd.DataFrame({"match_id": d["match_id"], "innings": d["innings"], "name": d["batter"], "ord": idx * 2})
    b = pd.DataFrame({"match_id": d["match_id"], "innings": d["innings"], "name": d["non_striker"], "ord": idx * 2 + 1})
    ap = pd.concat([a, b]).dropna(subset=["name"])
    first_ord = ap.groupby(["match_id", "innings", "name"])["ord"].min().reset_index()
    first_ord["pos"] = first_ord.groupby(["match_id", "innings"])["ord"].rank(method="first").astype(int) - 1
    d = d.merge(first_ord[["match_id", "innings", "name", "pos"]],
                left_on=["match_id", "innings", "batter"], right_on=["match_id", "innings", "name"], how="left")
    d = d.drop(columns="name")
    d["pos"] = d["pos"].fillna(10).clip(upper=10).astype(int)
    return d
