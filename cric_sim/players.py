"""Player facts that Cricsheet does not give: bowling type (pace / spin).

Cricsheet has no bowling style, so we do two things:
1. ``guess_bowl_types`` guesses from WHEN a bowler bowls (spinners bowl the middle overs,
   rarely the death). It is only a guess.
2. You can override any player in a CSV (``name,bowl_type,role``). Your file always wins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PACE, SPIN, UNKNOWN = 0, 1, 2
TYPE_NAMES = {PACE: "pace", SPIN: "spin", UNKNOWN: "unknown"}
_FROM_NAME = {"pace": PACE, "fast": PACE, "seam": PACE, "spin": SPIN, "unknown": UNKNOWN, "": UNKNOWN}


def guess_bowl_types(balls: pd.DataFrame, min_balls: int = 90) -> dict[str, int]:
    legal = balls[~(balls["wide"] | balls["noball"])]
    tab = legal.groupby(["bowler", "phase"]).size().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    out: dict[str, int] = {}
    for name, row in tab.iterrows():
        n = row.sum()
        if n < min_balls:
            out[name] = UNKNOWN
            continue
        pp, mid, death = row[0] / n, row[1] / n, row[2] / n
        out[name] = SPIN if (mid >= 0.62 and death <= 0.14) else PACE
    return out


def load_player_csv(path: str | Path | None) -> dict[str, dict]:
    """Columns: name, bowl_type (pace/spin), role (WK/BAT/AR/BOWL). All but name optional."""
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path).fillna("")
    out = {}
    for _, r in df.iterrows():
        out[r["name"]] = {
            "bowl_type": _FROM_NAME.get(str(r.get("bowl_type", "")).strip().lower(), UNKNOWN),
            "role": str(r.get("role", "")).strip().upper(),
        }
    return out


def bowl_types(balls: pd.DataFrame, csv_path: str | Path | None = None) -> dict[str, int]:
    types = guess_bowl_types(balls)
    for name, info in load_player_csv(csv_path).items():
        if info["bowl_type"] != UNKNOWN or name not in types:
            types[name] = info["bowl_type"]
    return types


def write_template(balls: pd.DataFrame, path: str | Path) -> int:
    """Write a CSV of every bowler with our guess, for you to correct by hand."""
    guess = guess_bowl_types(balls)
    legal = balls[~(balls["wide"] | balls["noball"])]
    n = legal.groupby("bowler").size()
    rows = [{"name": k, "bowl_type": TYPE_NAMES[v], "role": "", "balls_bowled": int(n.get(k, 0))}
            for k, v in guess.items()]
    pd.DataFrame(rows).sort_values("balls_bowled", ascending=False).to_csv(path, index=False)
    return len(rows)
