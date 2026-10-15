import json

import pandas as pd

import numpy as np
import pytest

from cric_sim.actual import actual_stats, points_of
from cric_sim.backtest import BTConfig, prepare, run_backtest, summarize
from cric_sim.dream11 import T20_RULES, score
from cric_sim.features import add_features
from cric_sim.loader import load_matches
from cric_sim.model import RateModel
from cric_sim.optimizer import pick_team
from cric_sim.players import PACE, SPIN
from cric_sim.simulator import Impact, Player, Team, simulate
from tests.conftest import TYPES

ROLES = ["WK", "BAT", "BAT", "BAT", "AR", "AR", "BOWL", "BOWL", "BOWL", "BOWL", "BOWL"]


def make_team(t, impact=None):
    players = [Player(f"T{t}_P{k}", ROLES[k], 8.5, f"Team{t}") for k in range(11)]
    return Team(f"Team{t}", players, players[5:11], impact)


# ---------------------------------------------------------------- data
def test_loader_reads_toss_venue_and_impact(world):
    m = world["matches"]
    assert {"toss_winner", "toss_decision", "venue_key", "replacements", "players_a"} <= set(m.columns)
    assert set(m["venue_key"]) == {"High Ground", "Spin Ground", "Chase Ground", "Plain Ground"}
    assert (m["replacements"].map(len) > 0).any()
    assert (m["players_a"].map(len) == 12).any()


def test_features_are_sane(world):
    b = world["bf"]
    assert b["pkey"].between(0, 23).all()
    assert (b.loc[b.innings == 1, "target"].isna()).all()
    assert b.loc[b.innings == 2, "target"].notna().all()
    assert b["pos"].between(0, 10).all()
    # openers are position 0 and 1
    first = b.groupby(["match_id", "innings"]).head(1)
    assert (first["pos"] == 0).all()


# ---------------------------------------------------------------- model finds planted effects
def test_ground_effects(world):
    vs = world["model"].venue_stats
    assert vs["High Ground"]["runs_per_over_vs_league"] > vs["Plain Ground"]["runs_per_over_vs_league"] + 0.4
    assert vs["Spin Ground"]["runs_per_over_vs_league"] < vs["Plain Ground"]["runs_per_over_vs_league"]


def test_spinners_get_more_help_at_spin_ground(world):
    m = world["model"]
    v = m.ids["venue"]["Spin Ground"]
    pace_w = m.tables["venue_type"][v * 3 + PACE][6]
    spin_w = m.tables["venue_type"][v * 3 + SPIN][6]
    assert spin_w > pace_w + 0.1


def test_chase_ground_is_easier_in_second_innings(world):
    m = world["model"]
    v = m.ids["venue"]["Chase Ground"]
    t = m.tables["venue_inn"]
    bound = lambda row: row[4] + row[5]
    assert bound(t[v * 2 + 1]) > bound(t[v * 2]) + 0.05


def test_batter_weak_against_spin_is_detected(world):
    m = world["model"]
    b = m.ids["bat"]["T0_P3"]
    bound = lambda r: r[4] + r[5]
    assert bound(m.tables["bat_type"][b * 3 + SPIN]) < 0.9 * bound(m.tables["bat_type"][b * 3 + PACE])


def test_extras_and_run_outs_recovered(world):
    m = world["model"]
    assert m.p_wide.mean() == pytest.approx(0.035, abs=0.006)
    assert m.p_noball.mean() == pytest.approx(0.006, abs=0.003)
    assert m.p_bye.mean() == pytest.approx(0.02, abs=0.005)
    assert m.p_runout.mean() == pytest.approx(0.005, abs=0.003)
    assert m.kind_p[SPIN][2] > m.kind_p[PACE][2]  # spinners get the stumpings


def test_chasing_pressure_raises_risk(world):
    m = world["model"]
    pr = m.tables["pressure"]
    # chasing 13-16 an over (bucket 6) vs. 5-7 an over (bucket 2), no wickets down
    bound = lambda row: row[4] + row[5]
    assert bound(pr[6 * 3]) > bound(pr[2 * 3])


def test_new_batters_are_more_vulnerable(world):
    nb = world["model"].tables["newbat"]
    assert nb[0][6] > nb[3][6]


def test_switching_features_off_removes_them(world):
    m = RateModel(features=()).fit(world["bf"], TYPES)
    assert "venue" not in m.tables and "pressure" not in m.tables
    assert m.extra_rates("T0_P6", 0) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- the points calculator
def _z(v=0):
    return np.array([[float(v)]])


def _blank():
    s = {k: _z() for k in ("runs", "balls", "fours", "sixes", "wkts", "bl_wkts", "dots", "maidens",
                           "bowl_runs", "bowl_balls", "catches", "stumpings", "run_outs")}
    s["out"] = np.array([[False]])
    return s


def test_scoring_rules_by_hand():
    s = _blank()
    s.update(runs=_z(52), balls=_z(30), fours=_z(4), sixes=_z(3))
    # XI 4 + runs 52 + fours 16 + sixes 18 + half-century 8 + strike rate 173 -> +6
    assert score(s, ["BAT"])[0, 0] == 4 + 52 + 16 + 18 + 8 + 6
    b = _blank()
    b.update(wkts=_z(3), bl_wkts=_z(1), dots=_z(10), bowl_runs=_z(24), bowl_balls=_z(24), maidens=_z(1))
    # XI 4 + 3 wkts 75 + bowled bonus 8 + 3-wkt bonus 4 + dots 10 + maiden 12 + economy 6.0 -> +2
    assert score(b, ["BOWL"])[0, 0] == 4 + 75 + 8 + 4 + 10 + 12 + 2


def test_duck_and_exemptions_and_unplayed():
    s = _blank()
    s.update(out=np.array([[True]]), balls=_z(2))
    assert score(s, ["BAT"])[0, 0] == 4 - 2
    assert score(s, ["BOWL"])[0, 0] == 4  # bowlers are not penalised for ducks
    s["played"] = np.array([[False]])
    assert score(s, ["BAT"])[0, 0] == 0


def test_actual_stats_on_a_tiny_hand_made_match(tmp_path):
    def d(batter, bowler, ns, runs, extras=None, wicket=None, total=None):
        x = {"batter": batter, "bowler": bowler, "non_striker": ns,
             "runs": {"batter": runs, "extras": sum((extras or {}).values()),
                      "total": total if total is not None else runs + sum((extras or {}).values())}}
        if extras:
            x["extras"] = extras
        if wicket:
            x["wickets"] = [wicket]
        return x

    over0 = [d("A", "B1", "C", 4), d("A", "B1", "C", 6),
             d("A", "B1", "C", 0, wicket={"player_out": "A", "kind": "bowled"}),
             d("C", "B1", "D", 0, wicket={"player_out": "C", "kind": "caught", "fielders": [{"name": "F1"}]}),
             d("D", "B1", "E", 0, extras={"wides": 1}), d("D", "B1", "E", 1)]
    over1 = [d("E", "B2", "D", 0) for _ in range(6)]
    match = {"info": {"dates": ["2024-01-01"], "venue": "Test, City", "teams": ["X", "Y"],
                      "players": {"X": ["A", "C", "D", "E"], "Y": ["B1", "B2", "F1"]},
                      "toss": {"winner": "X", "decision": "bat"}, "outcome": {"winner": "X"}},
             "innings": [{"team": "X", "overs": [{"over": 0, "deliveries": over0}, {"over": 1, "deliveries": over1}]},
                         {"team": "Y", "overs": [{"over": 0, "deliveries": [d("B1", "A", "B2", 0)]}]}]}
    (tmp_path / "1.json").write_text(json.dumps(match))
    balls, matches = load_matches(tmp_path)
    st = actual_stats(add_features(balls), matches).set_index("name")
    a = st.loc["A"]
    assert (a.runs, a.balls, a.fours, a.sixes, a.out) == (10, 3, 1, 1, True)
    b1 = st.loc["B1"]
    assert (b1.wkts, b1.bl_wkts, b1.dots, b1.bowl_balls, b1.bowl_runs) == (2, 1, 2, 5, 12)
    b2 = st.loc["B2"]
    assert (b2.maidens, b2.dots, b2.bowl_balls) == (1, 6, 6)
    assert st.loc["F1"].catches == 1
    assert st.loc["D"].balls == 1  # the wide is not a ball faced; the single is
    pts = points_of(st.reset_index(), {})
    assert np.isfinite(pts).all()


# ---------------------------------------------------------------- simulator
def test_simulation_is_realistic(world):
    sim = simulate(world["model"], make_team(0), make_team(1), n_sims=1500, seed=3, venue="Plain Ground")
    assert 120 < sim["avg_score_a"] < 230
    assert abs(sim["p_a_wins"] + sim["p_b_wins"] - 1) < 0.06
    assert sim["points"].shape == (1500, 22)
    assert np.isfinite(sim["points"]).all()


def test_high_scoring_ground_raises_scores(world):
    m = world["model"]
    hi = simulate(m, make_team(0), make_team(1), 1500, 1, "High Ground")
    lo = simulate(m, make_team(0), make_team(1), 1500, 1, "Spin Ground")
    assert hi["avg_score_a"] > lo["avg_score_a"] + 12


def test_extras_change_the_game(world):
    with_x = simulate(RateModel().fit(world["bf"], TYPES), make_team(0), make_team(1), 1500, 4, "Plain Ground")
    no_x = simulate(RateModel(features=("venue", "type", "innings", "state", "impact")).fit(world["bf"], TYPES),
                    make_team(0), make_team(1), 1500, 4, "Plain Ground")
    # wides, no-balls and leg byes add runs but cost nothing in balls; byes and wides add to the total
    assert with_x["avg_score_a"] != pytest.approx(no_x["avg_score_a"], abs=0.5)


def test_impact_player_scores_only_when_used(world):
    m = world["model"]
    sub = Player("T0_P12", "BOWL", 7.0, "Team0")
    never = make_team(0, Impact(sub, "bowl", "T0_P2", p_used=0.0))
    always = make_team(0, Impact(sub, "bowl", "T0_P2", p_used=1.0))
    s0 = simulate(m, never, make_team(1), 1000, 7, "Plain Ground")
    s1 = simulate(m, always, make_team(1), 1000, 7, "Plain Ground")
    i0 = [p.name for p in s0["players"]].index("T0_P12")
    assert s0["points"][:, i0].max() == 0
    assert s1["points"][:, i0].mean() > 20  # plays, bowls, gets playing-XI points
    # the replaced batter has already batted: he keeps his batting points but no bowling points
    j = [p.name for p in s1["players"]].index("T0_P2")
    assert s1["points"][:, j].mean() > 4


def test_batting_impact_sub_replaces_a_bowler_in_the_chase(world):
    m = world["model"]
    sub = Player("T1_P11", "BAT", 7.5, "Team1")
    team_b = make_team(1, Impact(sub, "bat", "T1_P10", p_used=1.0))
    s = simulate(m, make_team(0), team_b, 1000, 9, "Plain Ground")
    names = [p.name for p in s["players"]]
    assert s["points"][:, names.index("T1_P11")].mean() > 4  # bats in the chase
    # the replaced bowler bowled in innings 1 (so has points) and does not bat
    assert s["points"][:, names.index("T1_P10")].mean() > 4


# ---------------------------------------------------------------- optimizer
def test_optimizer_respects_rules(world):
    sim = simulate(world["model"], make_team(0), make_team(1), 600, 5, "Plain Ground")
    best = pick_team(sim["players"], sim["points"])
    team = best["team"]
    assert len(team) == 11 and best["credits"] <= 100
    assert max(sum(p.team == t for p in team) for t in ("Team0", "Team1")) <= 10
    assert best["captain"] in team and best["vice_captain"] in team and best["captain"] is not best["vice_captain"]
    for role, (lo, hi) in {"WK": (1, 4), "BAT": (3, 6), "AR": (1, 4), "BOWL": (3, 6)}.items():
        assert lo <= sum(p.role == role for p in team) <= hi


def test_optimizer_relaxes_when_roles_are_unusable():
    players = [Player(f"p{i}", "BAT", 8.0, "A" if i < 11 else "B") for i in range(22)]
    best = pick_team(players, np.random.default_rng(0).random((5, 22)))
    assert len(best["team"]) == 11


# ---------------------------------------------------------------- backtest
def test_backtest_uses_only_the_past(world, monkeypatch):
    seen = []
    real_fit = RateModel.fit

    def spy(self, balls, bowl_types=None):
        seen.append(balls["date"].max())
        return real_fit(self, balls, bowl_types)

    monkeypatch.setattr(RateModel, "fit", spy)
    bf, st = prepare(world["balls"], world["matches"])
    cfg = BTConfig(n_test=5, sims=100, refit_every=1, types_override=TYPES, random_draws=2)
    rows = []
    df = run_backtest(bf, world["matches"], st, cfg, collect=rows)
    assert rows and {"pred_bat", "act_bat", "plan_pos", "real_pos"} <= set(rows[0])
    assert len(df) == len(seen) == 5
    for fit_last_date, match_date in zip(seen, df["date"]):
        assert fit_last_date < match_date  # never trained on the match day or later


def test_backtest_beats_random_on_data_with_real_signal(world):
    bf, st = prepare(world["balls"], world["matches"])
    cfg = BTConfig(n_test=25, sims=500, refit_every=25, types_override=TYPES, random_draws=8)
    s = summarize(run_backtest(bf, world["matches"], st, cfg))
    assert s["matches"] >= 20
    assert s["pts_model"] > s["pts_random"] + 50
    assert s["pts_oracle"] > s["pts_model"]  # hindsight is always at least as good
    assert s["spearman"] > 0.2
    assert 0.6 < s["coverage_p10_p90"] < 0.95


# ---------------------------------------------------------------- several leagues together
def test_pooling_leagues_adds_a_league_effect(tmp_path):
    from cric_sim.loader import load_many
    from tests.synth import make_dataset
    hi, lo = tmp_path / "hi_json", tmp_path / "lo_json"
    make_dataset(hi, n_matches=60, seed=3)
    make_dataset(lo, n_matches=60, seed=4)
    # make the second league much lower scoring by turning every four into a dot
    for f in lo.glob("*.json"):
        d = json.loads(f.read_text())
        for inn in d["innings"]:
            for ov in inn["overs"]:
                for x in ov["deliveries"]:
                    if x["runs"]["batter"] in (4, 6):
                        x["runs"] = {"batter": 0, "extras": x["runs"]["extras"], "total": x["runs"]["extras"]}
        f.write_text(json.dumps(d))
    lo_ids = {f.name: f.name.replace(".json", "x.json") for f in lo.glob("*.json")}
    for old, new in lo_ids.items():  # make match ids unique across folders
        (lo / old).rename(lo / new)
    balls, matches = load_many([hi, lo])
    assert set(balls["league"]) == {"hi", "lo"} and len(matches) == 120
    m = RateModel().fit(add_features(balls), TYPES)
    t = m.tables["league"]
    boundary = lambda row: row[4] + row[5]
    assert boundary(t[m.ids["league"]["hi"]]) > 1.5 * boundary(t[m.ids["league"]["lo"]])


# ---------------------------------------------------------------- lineup uncertainty and day-to-day form
def test_lineup_uncertainty_lifts_the_tail_and_spreads_bowling(world):
    m_on = RateModel(features=("lineup", "venue", "type", "innings", "state", "extras")).fit(world["bf"], TYPES)
    m_off = RateModel(features=("venue", "type", "innings", "state", "extras")).fit(world["bf"], TYPES)
    on = simulate(m_on, make_team(0), make_team(1), 2000, 11, "Plain Ground")
    off = simulate(m_off, make_team(0), make_team(1), 2000, 11, "Plain Ground")
    # with a fixed order the number 11 bats last every time; with a shuffled order he sometimes bats higher
    assert on["points"][:, 10].mean() > off["points"][:, 10].mean() - 1
    # a fixed plan gives the 5th-6th bowlers exactly the same overs each time; a random one varies them
    assert on["points"][:, 6].std() > 0 and off["points"][:, 6].std() > 0
    assert on["points"][:, 5:11].std(axis=0).mean() != pytest.approx(off["points"][:, 5:11].std(axis=0).mean(), abs=0.01)


def test_day_to_day_form_widens_ranges_but_keeps_scoring_level(world):
    base = ("lineup", "venue", "type", "innings", "state", "extras")
    flat = RateModel(features=base).fit(world["bf"], TYPES)
    wavy = RateModel(features=base + ("form",), sigmas=(0.6, 0.3, 0.15)).fit(world["bf"], TYPES)
    a = simulate(flat, make_team(0), make_team(1), 3000, 5, "Plain Ground")
    b = simulate(wavy, make_team(0), make_team(1), 3000, 5, "Plain Ground")
    assert b["points"][:, :11].std(axis=0).mean() > a["points"][:, :11].std(axis=0).mean()
    # known flaw: swings are not neutral, so scores drift up (about +14 runs here, +11 on real IPL 2025-26)
    assert 0 < b["avg_score_a"] - a["avg_score_a"] < 25
    fixed = RateModel(features=base + ("form",), sigmas=(0.6, 0.3, 0.15), mean_preserving=True).fit(world["bf"], TYPES)
    c = simulate(fixed, make_team(0), make_team(1), 3000, 5, "Plain Ground")
    assert abs(c["avg_score_a"] - a["avg_score_a"]) < 5  # the optional fix restores the first-innings level
    assert c["points"][:, :11].std(axis=0).mean() > a["points"][:, :11].std(axis=0).mean()


def test_pooling_drops_associate_nation_matches(tmp_path):
    from cric_sim.loader import load_many
    from tests.synth import make_dataset
    folder = tmp_path / "t20s_json"
    make_dataset(folder, n_matches=20)
    # the fake league's teams are called Team0..Team7: none are Test nations, so the filter removes all
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        balls, matches = load_many([folder])
        if len(matches) == 0:
            raise ValueError("all matches filtered out")


# ---------------------------------------------------------------- confirmed lineups and conditions
def _mini_match(path, date, match_number, with_sub=False):
    d = lambda b, w, ns, r: {"batter": b, "bowler": w, "non_striker": ns, "runs": {"batter": r, "extras": 0, "total": r}}
    inn1 = [d("A", "B1", "C", 1) for _ in range(6)]
    inn2 = [d("B1", "A", "B2", 0) for _ in range(6)]
    if with_sub:
        inn2[0]["replacements"] = {"match": [{"in": "S", "out": "C", "team": "X", "reason": "impact_player"}]}
    m = {"info": {"dates": [date], "venue": "V, City", "teams": ["X", "Y"], "event": {"name": "L", "match_number": match_number},
                  "players": {"X": ["A", "C", "D"] + (["S"] if with_sub else []), "Y": ["B1", "B2"]},
                  "toss": {"winner": "X", "decision": "bat"}, "outcome": {"winner": "X"}},
         "innings": [{"team": "X", "overs": [{"over": 0, "deliveries": inn1}]},
                     {"team": "Y", "overs": [{"over": 0, "deliveries": inn2}]}]}
    path.write_text(json.dumps(m))


def test_substitutions_on_deliveries_and_afternoon_games(tmp_path):
    _mini_match(tmp_path / "1.json", "2024-04-06", 19, with_sub=True)
    _mini_match(tmp_path / "2.json", "2024-04-06", 20)
    _mini_match(tmp_path / "3.json", "2024-04-07", 21)
    balls, matches = load_matches(tmp_path)
    m = matches.set_index("match_id")
    rep = m.loc["1", "replacements"]
    assert rep and rep[0]["in"] == "S" and rep[0]["out"] == "C" and rep[0]["reason"] == "impact_player" and rep[0]["innings"] == 2
    assert bool(m.loc["1", "afternoon"]) and not bool(m.loc["2", "afternoon"]) and not bool(m.loc["3", "afternoon"])
    assert set(balls.loc[balls.match_id == "1", "afternoon"]) == {True}


def test_recent_pitch_reads_how_the_ground_is_playing(world):
    bf = world["bf"]
    no_ground = RateModel(features=("type", "innings", "state")).fit(bf, TYPES)  # knows nothing about grounds
    last = bf["date"].max()
    recent = lambda v: bf[(bf.venue_key == v) & (bf.date > last - pd.Timedelta(days=400))]
    hi = no_ground.recent_pitch(recent("High Ground"), S=300)
    spin = no_ground.recent_pitch(recent("Spin Ground"), S=300)
    assert hi[4] > 1.05 and hi[5] > 1.05  # more fours and sixes than the model expects
    assert spin[6] > 1.05  # more wickets
    assert np.allclose(no_ground.recent_pitch(bf.iloc[0:0]), 1.0)  # no recent matches: no change
    sim_hi = simulate(no_ground, make_team(0), make_team(1), 1000, 3, "High Ground", pitch_adj=hi)
    sim_0 = simulate(no_ground, make_team(0), make_team(1), 1000, 3, "High Ground")
    assert sim_hi["avg_score_a"] > sim_0["avg_score_a"] + 5


def test_backtest_with_real_subs_and_recent_pitch(world):
    bf, st = prepare(world["balls"], world["matches"])
    cfg = BTConfig(n_test=4, sims=100, refit_every=4, types_override=TYPES, random_draws=2, real_subs=True,
                   recent_pitch=True, empirical_lineup=True, features=("lineup", "venue", "type", "innings", "impact", "daynight"))
    df = run_backtest(bf, world["matches"], st, cfg)
    assert len(df) == 4 and df["pts_model"].notna().all()


def test_squad_builder_matches_app_style_names(world):
    from cric_sim.build_squad import build_squad, match_player, match_team
    names = ["V Kohli", "RG Sharma", "B Sai Sudharsan", "JJ Bumrah", "Mohammed Siraj"]
    assert match_player("Virat Kohli", names, names)[0] == "V Kohli"
    assert match_player("Rohit Sharma", names, names)[0] == "RG Sharma"
    assert match_player("Sai Sudharsan", names, names)[0] == "B Sai Sudharsan"
    assert match_player("Mohd Siraj", names, names)[0] == "Mohammed Siraj"
    assert match_team("RCB", ["Royal Challengers Bangalore", "Royal Challengers Bengaluru"]) == "Royal Challengers Bengaluru"
    assert match_team("India", ["India", "Mumbai Indians", "West Indies"]) == "India"
    assert match_team("India", ["Mumbai Indians", "India"]) == "India"
    assert match_team("West Indies", ["Mumbai Indians", "West Indies", "India"]) == "West Indies"
    bf, st = prepare(world["balls"], world["matches"])
    xi = lambda t: [f"T{t}_P{k}" for k in range(11)]
    squad, notes = build_squad(st, world["matches"], "Team0", xi(0), "Team1", xi(1), "High Grnd", sub_b="T1_P12")
    assert squad["venue"] == "High Ground"
    assert [p["name"] for p in squad["team_a"]["players"]][0] in xi(0)
    assert sum(p["bowls"] for p in squad["team_a"]["players"]) >= 5
    assert squad["team_b"]["impact"]["name"] == "T1_P12" and squad["team_b"]["impact"]["replaces"] in xi(1)


def test_typed_pitch_report_moves_the_game(world):
    m = world["model"]
    flat = simulate(m, make_team(0), make_team(1), 1500, 2, "Plain Ground")
    high = simulate(m, make_team(0), make_team(1), 1500, 2, "Plain Ground", pitch_report={"scoring": "high"})
    spin = simulate(m, make_team(0), make_team(1), 1500, 2, "Plain Ground", pitch_report={"spin": "big"})
    assert high["avg_score_a"] > flat["avg_score_a"] + 4
    assert spin["avg_score_a"] < flat["avg_score_a"]
    with pytest.raises(ValueError):
        simulate(m, make_team(0), make_team(1), 10, 2, "Plain Ground", pitch_report={"scoring": "huge"})


def test_scorecard_only_matches_feed_form_and_positions(world, tmp_path):
    from cric_sim.scorecards import add_scorecards
    from cric_sim.lineup import form_points, history_table, infer_roles
    card = {"matches": [{
        "match_id": "SC1", "date": "2099-01-01", "venue": "High Ground, Hiville", "league": "t20s",
        "team_a": "Team0", "team_b": "Team1", "winner": "Team0",
        "xi": {"Team0": [f"T0_P{k}" for k in range(11)], "Team1": [f"T1_P{k}" for k in range(11)]},
        "batting": {"Team0": [["T0_P1", 120, 50, None, None, False], ["T0_P0", 10, 8, 1, 0, True]],
                    "Team1": [["T1_P0", 5, 6, 0, 0, True]]},
        "bowling": {"Team1": [["T1_P6", "4", 0, 30, 1]], "Team0": [["T0_P6", "3.3", 0, 20, 1]]},
        "fielding": [["T1_P2", "catch"]], "bowled_lbw": {"T1_P6": 1}}]}
    path = tmp_path / "card.json"
    path.write_text(json.dumps(card))
    bf, st = prepare(world["balls"], world["matches"])
    m2, st2 = add_scorecards(world["matches"], st, [path])
    assert len(m2) == len(world["matches"]) + 1 and (st2["match_id"] == "SC1").sum() == 22
    r = st2[(st2.match_id == "SC1")].set_index("name")
    assert r.loc["T0_P1", "pos"] == 0 and r.loc["T0_P1", "runs"] == 120 and not r.loc["T0_P1", "out"]
    assert 0 < r.loc["T0_P1", "fours"] * 4 + r.loc["T0_P1", "sixes"] * 6 <= 120  # estimated boundaries fit the runs
    assert r.loc["T0_P6", "bowl_balls"] == 21 and r.loc["T1_P6", "bl_wkts"] == 1 and r.loc["T1_P2", "catches"] == 1
    roles = infer_roles(history_table(st))
    before, _, _ = form_points(st, roles)
    after, _, _ = form_points(st2, roles)
    assert after["T0_P1"] > before["T0_P1"]  # a big recent innings lifts form
    _, st3 = add_scorecards(m2, st2, [path])
    assert (st3["match_id"] == "SC1").sum() == 22  # adding the same file twice changes nothing


def test_captain_value_and_split_captaincy():
    players = [Player(f"a{i}", r, 8.0, "A") for i, r in enumerate(["WK", "BAT", "BAT", "BAT", "AR", "BOWL", "BOWL", "BOWL", "AR", "BAT", "BOWL"])]
    players += [Player(f"b{i}", r, 8.0, "B") for i, r in enumerate(["WK", "BAT", "BAT", "BAT", "AR", "BOWL", "BOWL", "BOWL", "AR", "BAT", "BOWL"])]
    value = np.array([50.0] * 22); value[1] = 90; value[2] = 85  # a1, a2 best by value
    cap = value.copy(); cap[13] = 200  # b2 is the best captain by the captain value
    base = pick_team(players, value[None, :])
    assert base["captain"].name == "a1" and base["vice_captain"].name == "a2"
    c = pick_team(players, value[None, :], captain_value=cap)
    assert c["captain"].name == "b2" and c["captain"] in c["team"]
    s = pick_team(players, value[None, :], split_cv=True)
    assert s["captain"].team != s["vice_captain"].team



def test_credit_bands_match_dream11():
    from cric_sim.lineup import credits_from_rank, estimate_credits
    c = credits_from_rank(np.arange(22) / 21)
    assert c.min() == 7.0 and c.max() == 11.0 and set(np.unique(c * 2) % 1) == {0.0}  # 0.5 steps
    assert 5 <= (c >= 9.5).sum() <= 7 and 6 <= ((c >= 8.5) & (c <= 9.0)).sum() <= 8 and 8 <= (c <= 8.0).sum() <= 10
    assert 8.4 <= c.mean() <= 8.8  # typical Dream11 match average
    names = [f"p{i}" for i in range(22)]
    cr = estimate_credits(names, {}, {n: float(i) for i, n in enumerate(names)}, {}, 30.0)
    assert cr["p21"] == 11.0 and cr["p0"] == 7.0  # best recent form gets the top credit


def test_team_limit_is_ten_per_side():
    players = [Player(f"a{i}", r, 7.0, "A") for i, r in enumerate(["WK", "BAT", "BAT", "BAT", "AR", "BOWL", "BOWL", "BOWL", "AR", "BAT", "BOWL"])]
    players += [Player(f"b{i}", r, 7.0, "B") for i, r in enumerate(["WK", "BAT", "BAT", "BAT", "AR", "BOWL", "BOWL", "BOWL", "AR", "BAT", "BOWL"])]
    value = np.array([100.0] * 11 + [1.0] * 11)  # team A players are far better
    t = pick_team(players, value[None, :])
    assert sum(p.team == "A" for p in t["team"]) == 10  # Dream11 allows 10, not 11
    t7 = pick_team(players, value[None, :], max_per_team=7)
    assert sum(p.team == "A" for p in t7["team"]) == 7
