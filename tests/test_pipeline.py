import numpy as np
import pytest

from cric_sim.dream11 import T20_RULES, score
from cric_sim.loader import load_matches
from cric_sim.model import RateModel
from cric_sim.optimizer import pick_team
from cric_sim.simulator import Player, Team, simulate
from tests.synth import make_dataset, team_names

ROLES = ["WK", "BAT", "BAT", "BAT", "AR", "AR", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL"]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    folder = tmp_path_factory.mktemp("data")
    make_dataset(folder, n=60, power={"A1": 1.6})  # A1 is a big hitter
    balls = load_matches(folder)
    return balls, RateModel().fit(balls)


def make_team(prefix):
    players = [Player(n, r, 8.5, prefix) for n, r in zip(team_names(prefix), ROLES)]
    return Team(prefix, players, players[6:])


def test_loader_columns(fitted):
    balls, _ = fitted
    assert {"batter", "bowler", "phase", "runs_batter", "wicket"} <= set(balls.columns)
    assert balls["phase"].isin([0, 1, 2]).all()


def test_rates_sum_to_one_and_league_is_sane(fitted):
    _, m = fitted
    assert np.allclose(m.league.sum(axis=1), 1)
    assert np.allclose(m.matchup("A1", "B8", 0).sum(), 1)
    assert 0.02 < m.league[1][6] < 0.08  # wicket chance per ball


def test_big_hitter_is_rated_higher(fitted):
    _, m = fitted
    hard = lambda n: m.batter_rate(n, 1)[4] + m.batter_rate(n, 1)[5]
    assert hard("A1") > hard("A2")


def test_simulation_is_realistic_and_big_hitter_scores_more(fitted):
    _, m = fitted
    sim = simulate(m, make_team("A"), make_team("B"), n_sims=1500, seed=3)
    assert 110 < sim["avg_score_a"] < 210
    assert abs(sim["p_a_wins"] + sim["p_b_wins"] - 1) < 0.06  # only ties are missing
    pts = sim["points"].mean(axis=0)
    assert pts[0] > pts[1] + 3  # A1 vs A2, A1 should score clearly more
    assert sim["points"].shape == (1500, 22)


def test_scoring_rules_by_hand():
    z = lambda v=0: np.array([[float(v)]])
    stats = {k: z() for k in ("runs", "balls", "fours", "sixes", "wkts", "bl_wkts", "dots",
                              "maidens", "bowl_runs", "bowl_balls", "catches", "stumpings", "run_outs")}
    stats["out"] = np.array([[False]])
    stats.update(runs=z(52), balls=z(30), fours=z(4), sixes=z(3))
    pts = score(stats, ["BAT"])[0, 0]
    # XI 4 + runs 52 + fours 16 + sixes 18 + half-century 8 + SR 173 -> +6
    assert pts == 4 + 52 + 16 + 18 + 8 + 6
    bowl = {k: z() for k in stats}
    bowl["out"] = np.array([[False]])
    bowl.update(wkts=z(3), bl_wkts=z(1), dots=z(10), bowl_runs=z(24), bowl_balls=z(24), maidens=z(0))
    # XI 4 + 3 wkts 75 + bowled bonus 8 + 3-wkt bonus 4 + dots 10 + economy 6.0 -> +2
    assert score(bowl, ["BOWL"])[0, 0] == 4 + 75 + 8 + 4 + 10 + 2


def test_optimizer_respects_rules(fitted):
    _, m = fitted
    a, b = make_team("A"), make_team("B")
    sim = simulate(m, a, b, n_sims=800, seed=5)
    best = pick_team(sim["players"], sim["points"])
    team = best["team"]
    assert len(team) == 11
    assert best["credits"] <= 100
    assert max(sum(p.team == t for p in team) for t in "AB") <= 7
    assert best["captain"] in team and best["vice_captain"] in team
    assert best["captain"] is not best["vice_captain"]
    for role, (lo, hi) in {"WK": (1, 4), "BAT": (3, 6), "AR": (1, 4), "BOWL": (3, 6)}.items():
        assert lo <= sum(p.role == role for p in team) <= hi
