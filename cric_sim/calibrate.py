"""Learned correction layer ("stacking").

The simulator gives, for every player, expected batting / bowling / fielding points and a range. Recent
form gives another estimate. This layer learns from PAST matches how to turn those signals into a better
prediction of real Dream11 points. It is trained only on matches before the ones it predicts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimizer import pick_team
from .simulator import Player

ROLES = ("WK", "BAT", "AR", "BOWL")


def features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in ("pred", "pred_bat", "pred_bowl", "pred_field", "pred_std", "pred_p90", "pred_p10", "form"):
        X[c] = df[c]
    X["plan_pos"] = df["plan_pos"].fillna(11) % 11  # place in its own team's expected order
    X["bowl_weight"] = df["bowl_weight"].fillna(0.0)
    X["log_matches"] = np.log1p(df["hist_matches"])
    X["hist_bowl_balls"] = df["hist_bowl_balls"]
    X["hist_bat_balls"] = df["hist_bat_balls"]
    X["hist_pos"] = df["hist_pos"].fillna(9)
    X["is_sub"] = df["is_sub"].astype(float)
    X["bats_first"] = df["bats_first"].astype(float)
    for r in ROLES:
        X[f"role_{r}"] = (df["role"] == r).astype(float)
    return X


class Ridge:
    """Plain ridge regression on standardised features (no extra libraries needed)."""

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        A = X.to_numpy(float)
        self.mu, self.sd = A.mean(0), A.std(0) + 1e-9
        Z = (A - self.mu) / self.sd
        self.b0 = y.mean()
        self.w = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (y - self.b0))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.b0 + ((X.to_numpy(float) - self.mu) / self.sd) @ self.w


def make_model(kind: str):
    if kind == "ridge":
        return Ridge(10.0)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=60,
                                         l2_regularization=1.0, random_state=0)


def walk_forward(rows: pd.DataFrame, kind: str = "gbm", min_train: int = 1500, step: str = "season") -> pd.Series:
    """Predict every row with a model trained only on rows from earlier seasons (or earlier dates)."""
    d = rows[rows["played"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["season"] = d["date"].dt.year
    out = pd.Series(np.nan, index=d.index)
    for s in sorted(d["season"].unique()):
        tr, te = d[d["season"] < s], d[d["season"] == s]
        if len(tr) < min_train:
            continue
        m = make_model(kind).fit(features(tr), tr["actual"].to_numpy())
        out.loc[te.index] = m.predict(features(te))
    return out


def xi_points(rows: pd.DataFrame, value_col: str) -> pd.Series:
    """Real Dream11 points of the XI picked with ``value_col`` as the value, per match."""
    res = {}
    for mid, g in rows.groupby("match_id"):
        g = g[g[value_col].notna()]
        if len(g) < 15:
            continue
        players = [Player(n, r, c, t) for n, r, c, t in zip(g["name"], g["role"], g["credits"], g["team"])]
        best = pick_team(players, g[value_col].to_numpy()[None, :])
        act = dict(zip(g["name"], g["actual"]))
        res[mid] = sum(act[p.name] for p in best["team"]) + act[best["captain"].name] + 0.5 * act[best["vice_captain"].name]
    return pd.Series(res)
