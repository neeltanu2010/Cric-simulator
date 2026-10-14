"""Read Cricsheet ball-by-ball JSON files into one flat table of deliveries."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Phase of a T20 innings, by over number (0-based).
# 0 = powerplay (overs 1-6), 1 = middle (7-16), 2 = death (17-20)
PHASE_BREAKS = (6, 16)


def phase_of_over(over: int) -> int:
    if over < PHASE_BREAKS[0]:
        return 0
    if over < PHASE_BREAKS[1]:
        return 1
    return 2


def _parse_match(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    info = m.get("info", {})
    match_id = path.stem
    date = (info.get("dates") or [None])[0]
    venue = info.get("venue")
    rows: list[dict] = []
    for inn_no, inn in enumerate(m.get("innings", []), start=1):
        if inn.get("super_over"):
            continue  # super overs are not part of normal match scoring
        team = inn.get("team")
        for over in inn.get("overs", []):
            ov = over["over"]
            for ball_no, d in enumerate(over["deliveries"], start=1):
                extras = d.get("extras", {})
                wickets = d.get("wickets", [])
                w = wickets[0] if wickets else {}
                rows.append(
                    {
                        "match_id": match_id,
                        "date": date,
                        "venue": venue,
                        "innings": inn_no,
                        "team": team,
                        "over": ov,
                        "ball": ball_no,
                        "phase": phase_of_over(ov),
                        "batter": d["batter"],
                        "bowler": d["bowler"],
                        "runs_batter": d["runs"]["batter"],
                        "runs_total": d["runs"]["total"],
                        "wide": "wides" in extras,
                        "noball": "noballs" in extras,
                        "wicket": bool(wickets),
                        "player_out": w.get("player_out"),
                        "kind": w.get("kind"),
                        "fielder": (w.get("fielders") or [{}])[0].get("name"),
                    }
                )
    return rows


def load_matches(folder: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Load every ``*.json`` Cricsheet match in ``folder``."""
    files = sorted(Path(folder).glob("*.json"))
    if limit:
        files = files[:limit]
    rows: list[dict] = []
    for p in files:
        rows.extend(_parse_match(p))
    if not rows:
        raise FileNotFoundError(f"No Cricsheet JSON matches found in {folder}")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df
