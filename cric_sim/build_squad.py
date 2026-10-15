"""Turn a confirmed lineup (the XI announced at the toss, as shown in the Dream11 or Cricbuzz app) into a
squad file. Names are matched to the data's spelling ("Virat Kohli" -> "V Kohli"); roles, batting order
and likely bowlers come from each player's history. Always check the printed name matches."""
from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import pandas as pd

from .lineup import estimate_credits, form_points, history_table, infer_roles, order_by_position, pick_bowlers

TEAM_ALIASES = {
    "mi": "Mumbai Indians", "csk": "Chennai Super Kings", "rcb": "Royal Challengers", "kkr": "Kolkata Knight Riders",
    "srh": "Sunrisers Hyderabad", "dc": "Delhi Capitals", "pbks": "Punjab Kings", "kxip": "Kings XI Punjab",
    "rr": "Rajasthan Royals", "gt": "Gujarat Titans", "lsg": "Lucknow Super Giants",
}


def _norm(x: str) -> str:
    """lower case, letters and single spaces only: "R.G. Sharma" -> "r g sharma"."""
    return " ".join(re.sub(r"[^a-z ]", " ", x.lower()).split())


def match_team(raw: str, teams: list[str]) -> str:
    """Latest full team name for an exact name, an abbreviation or part of a name
    ("India" -> "India", never "Mumbai Indians"; "RCB" -> "Royal Challengers Bengaluru")."""
    n = _norm(raw)
    for key in (n, _norm(TEAM_ALIASES.get(n.replace(" ", ""), raw))):
        exact = [t for t in teams if _norm(t) == key]
        if exact:
            return exact[-1]
        words = [t for t in teams if re.search(rf"\b{re.escape(key)}\b", _norm(t))]
        if words:
            return words[-1]  # ``teams`` is in date order: the newest name wins (Bangalore -> Bengaluru)
    best = difflib.get_close_matches(raw, teams, n=1, cutoff=0.3)
    if not best:
        raise ValueError(f"team '{raw}' not found in the data")
    return best[0]


def match_player(raw: str, team_names: list[str], all_names: list[str]) -> tuple[str, str]:
    """Return (data name, how it was matched). Prefers players seen recently for this team."""
    if raw in all_names:
        return raw, "exact"
    r = _norm(raw).split()
    if not r:
        raise ValueError("empty player name")
    surname, first = r[-1], r[0]

    def score(c: str) -> float:
        t = _norm(c).split()
        if not t or t[-1] != surname:
            return 0.0
        s = 1.0
        if len(r) > 1:  # first name or initials must agree
            initials = t[0] if len(t) > 1 else ""
            if len(t) > 1 and (t[0] == first or first in t[1:-1] or (initials and initials[0] == first[0])):
                s += 1.0
            elif len(t) > 1:
                return 0.0
        return s

    for pool, how in ((team_names, "surname, recent for this team"), (all_names, "surname")):
        cands = sorted(((score(c), c) for c in pool), reverse=True)
        if cands and cands[0][0] > 0 and (len(cands) == 1 or cands[0][0] > cands[1][0]):
            return cands[0][1], how
    close = difflib.get_close_matches(raw, team_names, n=1, cutoff=0.5) or difflib.get_close_matches(raw, all_names, n=1, cutoff=0.6)
    if close:
        return close[0], "closest spelling - CHECK"
    return raw, "NOT FOUND - no history, league averages will be used"


def build_squad(stats: pd.DataFrame, matches: pd.DataFrame, team_a: str, xi_a: list[str], team_b: str, xi_b: list[str],
                venue: str | None = None, sub_a: str | None = None, sub_b: str | None = None,
                credits_a: list[float] | None = None, credits_b: list[float] | None = None,
                pitch: dict | None = None, keepers: list[str] | None = None,
                estimate: bool = True) -> tuple[dict, list[str]]:
    """Team A bats first. Returns (squad dict, lines describing every name match)."""
    teams_in_order = list(dict.fromkeys(pd.concat([matches["team_a"], matches["team_b"]]).iloc[matches.index.argsort()]))
    all_names = sorted(stats["name"].unique())
    recent = stats[stats["date"] >= stats["date"].max() - pd.Timedelta(days=730)]
    hist = history_table(stats)
    roles = infer_roles(hist)
    notes = []
    out = {"venue": None, "team_a": None, "team_b": None,
           "pitch": pitch or {"scoring": "normal", "spin": "normal", "pace": "normal"}}
    if venue:
        vk = sorted(matches["venue_key"].unique())
        v = venue if venue in vk else (difflib.get_close_matches(venue, vk, n=1, cutoff=0.3) or [venue])[0]
        out["venue"] = v
        notes.append(f"ground: '{venue}' -> '{v}'")
    for slot, raw_team, xi, sub, credits, first in (("team_a", team_a, xi_a, sub_a, credits_a, True),
                                                     ("team_b", team_b, xi_b, sub_b, credits_b, False)):
        tname = match_team(raw_team, teams_in_order)
        notes.append(f"team: '{raw_team}' -> '{tname}'")
        tn = sorted(recent.loc[recent["team"] == tname, "name"].unique())
        names = []
        for raw in xi:
            n, how = match_player(raw.strip(), tn, all_names)
            names.append(n)
            notes.append(f"  {raw.strip():24s} -> {n:24s} ({how})")
        if len(names) != 11:
            raise ValueError(f"{tname}: need exactly 11 players, got {len(names)}")
        cr = {n: float(c) for n, c in zip(names, credits)} if credits and len(credits) == 11 else {}
        if not cr and (credits_a or credits_b):
            notes.append(f"  {tname}: no credits given, using 8.5 for everyone - type the real Dream11 credits in")
        order = order_by_position(names, hist, roles)
        bowls = set(pick_bowlers(order, hist, roles, k=7))
        team = {"name": tname, "players": [{"name": n, "role": roles.get(n, "BAT"), "credits": cr.get(n, 8.5),
                                            "bowls": n in bowls} for n in order]}
        if sub:
            sn, how = match_player(sub.strip(), tn, all_names)
            notes.append(f"  impact sub {sub.strip():13s} -> {sn:24s} ({how})")
            h = lambda n, k: hist.get(n, {}).get(k, 0.0)
            if first:  # a bowling sub comes on for a batter who has batted (not the keeper)
                pool = [n for n in order if roles.get(n) != "WK"]
                rep = min(pool, key=lambda n: (h(n, "bowl_balls"), -h(n, "bat_balls")))
            else:  # a batting sub comes in for a bowler who has bowled
                rep = max(order, key=lambda n: h(n, "bowl_balls") - 0.1 * h(n, "bat_balls"))
            team["impact"] = {"name": sn, "role": roles.get(sn, "BAT"), "credits": 7.5, "replaces": rep, "p_used": 0.8}
            notes.append(f"  (guess) {sn} replaces {rep}; edit 'replaces' and 'p_used' if you know better")
        out[slot] = team
    # Dream11 fixes who counts as a wicket-keeper; let the user say so
    for raw in keepers or []:
        for slot in ("team_a", "team_b"):
            tn = sorted(recent.loc[recent["team"] == out[slot]["name"], "name"].unique())
            k, _ = match_player(raw.strip(), tn, all_names)
            for p in out[slot]["players"]:
                if p["name"] == k:
                    p["role"] = "WK"
                    notes.append(f"keeper: {raw.strip()} -> {k} marked WK")
    # no credits typed in: estimate them from recent form so the 100-credit budget still bites
    if estimate and not credits_a and not credits_b:
        names = [p["name"] for s_ in ("team_a", "team_b") for p in out[s_]["players"]]
        names += [out[s_]["impact"]["name"] for s_ in ("team_a", "team_b") if out[s_].get("impact")]
        form, role_mean, glob = form_points(stats, roles)
        cr = estimate_credits(names, roles, form, role_mean, glob)
        for s_ in ("team_a", "team_b"):
            for p in out[s_]["players"]:
                p["credits"] = cr[p["name"]]
            if out[s_].get("impact"):
                out[s_]["impact"]["credits"] = cr[out[s_]["impact"]["name"]]
        notes.append("credits: ESTIMATED from recent form (7 to 11). Replace them with the real Dream11 credits if you have them.")
    return out, notes


def write_squad(squad: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(squad, indent=1))
