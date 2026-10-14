"""Per-ball outcome model for each batter and bowler, by phase of the innings.

Outcomes of a ball (7 classes): dot, 1, 2, 3, four, six, wicket.

Each player's own record is shrunk toward the league average with a prior
("empirical Bayes"), so a player with only 20 balls does not look extreme.
Batter and bowler are then combined by multiplying their odds ratios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ["0", "1", "2", "3", "4", "6", "W"]
RUNS_OF = np.array([0, 1, 2, 3, 4, 6, 0])
N_OUT = len(OUTCOMES)
N_PHASE = 3


def _outcome_index(df: pd.DataFrame) -> np.ndarray:
    """Map every delivery to one of the 7 outcome classes."""
    r = df["runs_batter"].to_numpy()
    idx = np.zeros(len(df), dtype=int)
    idx[r == 1] = 1
    idx[r == 2] = 2
    idx[r == 3] = 3
    idx[r == 4] = 4
    idx[r >= 6] = 5
    idx[r == 5] = 4  # rare; treat as a four
    # a wicket on the ball overrides runs; run outs are not the bowler's wicket
    idx[df["wicket"].to_numpy()] = 6
    return idx


class RateModel:
    """Learns league, batter and bowler outcome rates from ball-by-ball data."""

    def __init__(
        self,
        bat_prior: float = 60.0,
        bowl_prior: float = 80.0,
        half_life_days: float | None = 730.0,
    ):
        self.bat_prior = bat_prior
        self.bowl_prior = bowl_prior
        self.half_life_days = half_life_days
        self.league = np.full((N_PHASE, N_OUT), 1 / N_OUT)
        self._bat: dict[tuple[str, int], np.ndarray] = {}
        self._bowl: dict[tuple[str, int], np.ndarray] = {}

    def fit(self, balls: pd.DataFrame) -> "RateModel":
        d = balls[~balls["wide"]].copy()  # wides are not a batter-bowler outcome
        d["o"] = _outcome_index(d)
        if self.half_life_days:
            age = (d["date"].max() - d["date"]).dt.days.to_numpy()
            d["w"] = 0.5 ** (age / self.half_life_days)
        else:
            d["w"] = 1.0

        for ph in range(N_PHASE):
            sub = d[d["phase"] == ph]
            c = np.bincount(sub["o"], weights=sub["w"], minlength=N_OUT) + 1e-3
            self.league[ph] = c / c.sum()

        for role, col, store in (("bat", "batter", self._bat), ("bowl", "bowler", self._bowl)):
            g = d.groupby([col, "phase", "o"])["w"].sum().unstack(fill_value=0.0)
            g = g.reindex(columns=range(N_OUT), fill_value=0.0)
            for (name, ph), row in g.iterrows():
                store[(name, ph)] = row.to_numpy()
        return self

    def _rate(self, store, prior, name: str, ph: int) -> np.ndarray:
        counts = store.get((name, ph), np.zeros(N_OUT))
        shrunk = counts + prior * self.league[ph]
        return shrunk / shrunk.sum()

    def batter_rate(self, name: str, ph: int) -> np.ndarray:
        return self._rate(self._bat, self.bat_prior, name, ph)

    def bowler_rate(self, name: str, ph: int) -> np.ndarray:
        return self._rate(self._bowl, self.bowl_prior, name, ph)

    def matchup(self, batter: str, bowler: str, ph: int) -> np.ndarray:
        """Outcome chances for this batter facing this bowler in this phase."""
        lg = self.league[ph]
        p = lg * (self.batter_rate(batter, ph) / lg) * (self.bowler_rate(bowler, ph) / lg)
        return p / p.sum()

    def known(self, name: str) -> bool:
        return any((name, ph) in self._bat or (name, ph) in self._bowl for ph in range(N_PHASE))
