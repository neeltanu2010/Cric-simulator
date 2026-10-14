"""Squad files: a small JSON file per match that you fill in after the toss.

{
  "venue": "Wankhede Stadium",
  "team_a": {                         # team A BATS FIRST
    "name": "MI",
    "players": [                      # the 11 starters, in expected batting order
      {"name": "RG Sharma", "role": "BAT", "credits": 10, "bowls": false}, ...
    ],
    "impact": {"name": "Arshad Khan", "role": "BOWL", "credits": 7.5,
               "replaces": "RG Sharma", "p_used": 0.8}      # optional impact player
  },
  "team_b": {...}                     # team B chases
}
Names must match Cricsheet's spelling (for example "V Kohli", "JJ Bumrah").
"""
from __future__ import annotations

import json
from pathlib import Path

from .simulator import Impact, Player, Team


def _team(d: dict, chasing: bool) -> Team:
    players, bowlers = [], []
    for p in d["players"]:
        pl = Player(p["name"], p["role"], float(p.get("credits", 8)), d["name"])
        players.append(pl)
        if p.get("bowls", p["role"] in ("BOWL", "AR")):
            bowlers.append(pl)
    impact = None
    if d.get("impact"):
        i = d["impact"]
        sub = Player(i["name"], i["role"], float(i.get("credits", 7.5)), d["name"])
        # a bowling sub helps the team that bats first; a batting sub helps the chasing team
        mode = i.get("mode") or ("bat" if chasing else "bowl")
        impact = Impact(sub, mode, i["replaces"], float(i.get("p_used", 0.8)))
    return Team(d["name"], players, bowlers, impact, d.get("bowl_plan"))


def load_squads(path: str | Path) -> tuple[Team, Team, str | None]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return _team(d["team_a"], False), _team(d["team_b"], True), d.get("venue")
