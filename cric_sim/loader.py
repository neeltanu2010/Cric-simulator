"""Read Cricsheet ball-by-ball JSON files into two tables.

``balls``   one row per delivery (including wides and no-balls)
``matches`` one row per match: venue, toss, winner, the players, impact subs
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Phase of a T20 innings, by over number (0-based).
# 0 = powerplay (overs 1-6), 1 = middle (7-16), 2 = death (17-20)
PHASE_BREAKS = (6, 16)
BOWLER_KINDS = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}


def phase_of_over(over: int) -> int:
    if over < PHASE_BREAKS[0]:
        return 0
    if over < PHASE_BREAKS[1]:
        return 1
    return 2


def venue_key(venue: str | None) -> str:
    """'Wankhede Stadium, Mumbai' and 'Wankhede Stadium' are the same ground."""
    return (venue or "unknown").split(",")[0].strip()


def _parse_match(path: Path) -> tuple[list[dict], dict | None]:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    info = m.get("info", {})
    match_id = path.stem
    date = (info.get("dates") or [None])[0]
    venue = info.get("venue")
    vkey = venue_key(venue)
    rows: list[dict] = []
    teams_batting: list[str] = []
    for inn_no, inn in enumerate(m.get("innings", []), start=1):
        if inn.get("super_over"):
            continue
        team = inn.get("team")
        teams_batting.append(team)
        for over in inn.get("overs", []):
            ov = over["over"]
            for k, d in enumerate(over["deliveries"]):
                ex = d.get("extras", {})
                wk = d.get("wickets", [])
                w = wk[0] if wk else {}
                kind = w.get("kind")
                fielders = [f.get("name") for f in w.get("fielders", []) if f.get("name")]
                rows.append({
                    "match_id": match_id, "date": date, "venue": venue, "venue_key": vkey,
                    "innings": inn_no, "team": team, "over": ov, "ball": k,
                    "phase": phase_of_over(ov),
                    "batter": d["batter"], "bowler": d["bowler"], "non_striker": d.get("non_striker"),
                    "runs_batter": d["runs"]["batter"], "runs_extras": d["runs"].get("extras", 0),
                    "runs_total": d["runs"]["total"],
                    "wide": "wides" in ex, "noball": "noballs" in ex,
                    "bye": ex.get("byes", 0), "legbye": ex.get("legbyes", 0),
                    "wide_runs": ex.get("wides", 0), "noball_runs": ex.get("noballs", 0),
                    "wicket": bool(wk), "player_out": w.get("player_out"), "kind": kind,
                    "bowler_wicket": kind in BOWLER_KINDS,
                    "runout": kind == "run out",
                    "fielders": fielders,
                })
    if not rows:
        return rows, None
    toss = info.get("toss", {})
    outcome = info.get("outcome", {})
    teams = info.get("teams") or list(dict.fromkeys(teams_batting))
    players = info.get("players", {})
    bat_first = teams_batting[0]
    other = next((t for t in teams if t != bat_first), None)
    repl = []
    for r in (info.get("replacements") or {}).get("match", []):
        repl.append({"in": r.get("in"), "out": r.get("out"), "team": r.get("team"),
                     "reason": r.get("reason")})
    meta = {
        "match_id": match_id, "date": date, "venue": venue, "venue_key": vkey,
        "city": info.get("city"), "season": str(info.get("season", "")),
        "team_a": bat_first, "team_b": other,
        "toss_winner": toss.get("winner"), "toss_decision": toss.get("decision"),
        "winner": outcome.get("winner"),
        "players_a": players.get(bat_first, []), "players_b": players.get(other, []),
        "replacements": repl,
    }
    return rows, meta


def load_matches(folder: str | Path, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load every ``*.json`` Cricsheet match in ``folder``. Returns (balls, matches)."""
    files = sorted(Path(folder).glob("*.json"))
    if limit:
        files = files[:limit]
    rows: list[dict] = []
    metas: list[dict] = []
    for p in files:
        r, meta = _parse_match(p)
        rows.extend(r)
        if meta:
            metas.append(meta)
    if not rows:
        raise FileNotFoundError(f"No Cricsheet JSON matches found in {folder}")
    balls = pd.DataFrame(rows)
    balls["date"] = pd.to_datetime(balls["date"])
    matches = pd.DataFrame(metas)
    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    return balls, matches
