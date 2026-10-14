"""Squad files: a small JSON file per match that you fill in after the XI is announced.

{
  "team_a": {"name": "MI", "players": [
      {"name": "RG Sharma", "role": "BAT", "credits": 10, "bowls": false}, ... 11 players in batting order ]},
  "team_b": {...}
}
Names must match Cricsheet's spelling (for example "V Kohli", "JJ Bumrah").
"""
from __future__ import annotations

import json
from pathlib import Path

from .simulator import Player, Team


def _team(d: dict) -> Team:
    players, bowlers = [], []
    for p in d["players"]:
        pl = Player(p["name"], p["role"], float(p.get("credits", 8)), d["name"])
        players.append(pl)
        if p.get("bowls", p["role"] in ("BOWL", "AR")):
            bowlers.append(pl)
    return Team(d["name"], players, bowlers)


def load_squads(path: str | Path) -> tuple[Team, Team]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return _team(d["team_a"]), _team(d["team_b"])
