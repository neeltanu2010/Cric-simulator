"""Walk-forward backtest.

For every test match, in date order:
  1. Train the model ONLY on matches played before that match's date.
  2. Work out the lineup the way you would before the game (batting order, bowlers, roles,
     credits) from past matches only. The playing XI, venue and who bats first are taken from
     the real match, because you know them after the toss.
  3. Simulate the match, pick the best XI with the optimiser.
  4. Score that XI with the REAL Dream11 points from what happened.
  5. Compare with simple baselines:
        form    pick by each player's recent average points
        random  a random valid team (average of several draws)
        oracle  the best possible team in hindsight (an upper limit, not a real strategy)

Known weaknesses (the report prints these too):
  * Cricsheet does not say who the impact sub was, only that 12 names are listed. The backtest
    guesses the sub and who he replaces from past matches. Wrong guesses cost accuracy.
  * Credits are estimated from form, not the real Dream11 credits.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .actual import actual_stats, points_of, points_parts
from .features import add_features
from .lineup import (estimate_credits, form_points, history_table, infer_roles, lineup_history,
                     order_by_position, phase_overs_table, pick_bowlers)
from .model import ALL_FEATURES, RateModel
from .optimizer import pick_team
from .players import bowl_types, load_player_csv
from .simulator import Impact, Player, Team, simulate

BLENDS = (0.1, 0.25, 0.5, 0.75, 0.9)  # share of the simulator in a simulator/form mix

STAGES = [  # name, features switched on (cumulative)
    ("players only", ()),
    ("+ lineup uncertainty", ("lineup",)),
    ("+ day-to-day form", ("lineup", "form")),
    ("+ ground", ("lineup", "form", "venue")),
    ("+ pace/spin", ("lineup", "form", "venue", "type")),
    ("+ innings/toss", ("lineup", "form", "venue", "type", "innings")),
    ("+ game situation", ("lineup", "form", "venue", "type", "innings", "state")),
    ("+ extras", ("lineup", "form", "venue", "type", "innings", "state", "extras")),
    ("+ impact (full)", ALL_FEATURES),
]


@dataclass
class BTConfig:
    features: tuple = ALL_FEATURES
    n_test: int = 100
    min_history: int = 60
    sims: int = 1000
    refit_every: int = 10
    seed: int = 0
    types_csv: str | None = None
    types_override: dict | None = None
    random_draws: int = 20
    label: str = "full"
    test_league: str | None = None  # only test matches from this league (training uses all)
    test_before: str | None = None  # only test matches before this date (for tuning on older seasons)
    test_from: str | None = None  # only test matches on or after this date
    sigmas: tuple = (0.25, 0.15, 0.12)  # day-to-day variation in batter form, bowler form, conditions
    half_life_days: float = 730.0
    prior_scale: float = 1.0
    mean_preserving: bool = False  # cancel the score drift caused by day-to-day swings
    empirical_lineup: bool = False  # draw batting positions and bowling overs from each player's real history
    real_subs: bool = True  # use the recorded impact sub (who came on); the starting XI is then exact
    real_sub_out: bool = False  # also use who went off (decided during the match, so off by default)
    recent_pitch: bool = False  # adjust for how the ground has played in the last ``recent_days``
    recent_days: int = 60
    recent_S: float = 1500.0
    mix_form_decay: float = 0.9  # decay of the form estimate used INSIDE the simulator/form mix (the reported
    # "form baseline" always uses the fixed 0.9 rule so results stay comparable)


def _team_xi(names, sub, hist):
    """Return (starting XI names, impact (in, out) or None). ``out`` may be None (to be guessed)."""
    if len(names) == 11:
        return list(names), None
    if len(names) == 12 and sub:
        return [n for n in names if n != sub[0]], sub
    return None, None


def _guess_out(xi, first, on_field, hist, roles):
    """Who the impact sub most likely replaced. Batting-first team: a batter who has batted (not the keeper).
    Chasing team: a bowler who has bowled."""
    h = lambda n, k: hist.get(n, {}).get(k, 0.0)
    if first:
        pool = [n for n in xi if n in on_field and roles.get(n) != "WK"] or xi
        return min(pool, key=lambda n: (h(n, "bowl_balls"), -h(n, "bat_balls")))
    pool = [n for n in xi if n in on_field] or xi
    return max(pool, key=lambda n: h(n, "bowl_balls") - 0.1 * h(n, "bat_balls"))


def _impact_rate(matches: pd.DataFrame, before: pd.Timestamp, league: str | None = None) -> float:
    m = matches[matches["date"] < before]
    if league:
        m = m[m["league"] == league]
    m = m.tail(80)
    if m.empty:
        return 0.0
    used = np.concatenate([(m["players_a"].map(len) > 11).to_numpy(), (m["players_b"].map(len) > 11).to_numpy()])
    return float(np.clip(used.mean(), 0.0, 0.95))


def _score_xi(res, idx_of, actual):
    xi = [idx_of[p.name] for p in res["team"]]
    c, v = idx_of[res["captain"].name], idx_of[res["vice_captain"].name]
    return float(sum(actual[i] for i in xi) + actual[c] + 0.5 * actual[v])


def run_backtest(balls_f: pd.DataFrame, matches: pd.DataFrame, stats: pd.DataFrame, cfg: BTConfig,
                 progress: bool = False, collect: list | None = None, fit_cache: dict | None = None) -> pd.DataFrame:
    """Returns one row per test match. If ``collect`` is a list, one dict per player per match is added
    to it (predicted vs real points, role, expected batting position), for diagnosing the model."""
    rng = np.random.default_rng(cfg.seed)
    ms = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    pool = ms.iloc[cfg.min_history:]
    if cfg.test_league:
        pool = pool[pool["league"] == cfg.test_league]
    if cfg.test_before:
        pool = pool[pool["date"] < pd.Timestamp(cfg.test_before)]
    if cfg.test_from:
        pool = pool[pool["date"] >= pd.Timestamp(cfg.test_from)]
    test = pool.tail(cfg.n_test)
    csv_over = load_player_csv(cfg.types_csv)
    inn_tot = balls_f.groupby(["match_id", "innings"])["runs_total"].sum().unstack()
    inn_tot = inn_tot.join(ms.set_index("match_id")[["venue_key", "date"]])

    if cfg.empirical_lineup:
        bph = phase_overs_table(balls_f)
    rows, model, model_n = [], None, 10 ** 9
    by_match: dict = {}
    t0 = time.time()
    for k, r in enumerate(test.itertuples()):
        past_b = None
        if model is None or model_n >= cfg.refit_every:
            # settings that do not change the fit (day-to-day variation, blending) can share a fitted model
            key = (tuple(sorted(cfg.features)), cfg.half_life_days, cfg.prior_scale, str(r.date), cfg.types_csv)
            model = fit_cache.get(key) if fit_cache is not None else None
            if model is None:
                past_b = balls_f[balls_f["date"] < r.date]
                types = dict(bowl_types(past_b, cfg.types_csv))
                if cfg.types_override:
                    types.update(cfg.types_override)
                model = RateModel(features=cfg.features, sigmas=cfg.sigmas, half_life_days=cfg.half_life_days,
                                  prior_scale=cfg.prior_scale, mean_preserving=cfg.mean_preserving).fit(past_b, types)
                if fit_cache is not None:
                    fit_cache[key] = model
            model_n = 0
        model.sigmas = cfg.sigmas
        model.mean_preserving = cfg.mean_preserving
        model_n += 1

        past_s = stats[stats["date"] < r.date]
        hist = history_table(past_s)
        roles = infer_roles(hist, csv_over)
        form, role_mean, glob = form_points(past_s, roles)
        mix_form = form if cfg.mix_form_decay == 0.9 else form_points(past_s, roles, decay=cfg.mix_form_decay)[0]

        teams = []
        ok = True
        p_imp = _impact_rate(ms, r.date, r.league) if model.has("impact") else 0.0
        for team_name, names, first in ((r.team_a, r.players_a, True), (r.team_b, r.players_b, False)):
            sub = None
            if cfg.real_subs:
                for x in r.replacements:
                    if (x["team"] == team_name and x.get("reason") in ("impact_player", "concussion_substitute", None)
                            and x["in"] in names and x["out"] in names):
                        sub = (x["in"], x["out"] if cfg.real_sub_out else None)
                        break
            xi, imp = _team_xi(names, sub, hist)
            if xi is not None and imp is not None and imp[1] is None:
                mb = by_match.get(r.match_id)
                if mb is None:
                    mb = by_match[r.match_id] = balls_f[balls_f["match_id"] == r.match_id]
                on_field = (set(mb.loc[mb["innings"] == 1, "batter"]) | set(mb.loc[mb["innings"] == 1, "non_striker"])) if first \
                    else set(mb.loc[mb["innings"] == 1, "bowler"])
                imp = (imp[0], _guess_out(xi, first, on_field, hist, roles))
            if xi is None:
                if len(names) < 11:
                    ok = False
                    break
                # Cricsheet lists 12 names but not who the impact sub was. At the toss you would know the
                # XI, so recover it from who was on the field: the sub cannot have batted in the first
                # innings (team batting first) or bowled in the first innings (chasing team). Among the
                # remaining names, the one used least by this team recently is the sub.
                mb = by_match.get(r.match_id)
                if mb is None:
                    mb = by_match[r.match_id] = balls_f[balls_f["match_id"] == r.match_id]
                if first:
                    on_field = set(mb.loc[(mb["innings"] == 1), "batter"]) | set(mb.loc[(mb["innings"] == 1), "non_striker"])
                else:
                    on_field = set(mb.loc[(mb["innings"] == 1), "bowler"])
                tm = past_s[past_s["team"] == team_name].drop_duplicates("match_id").tail(15)["match_id"]
                recent = past_s[past_s["match_id"].isin(tm) & (past_s["team"] == team_name)].groupby("name").size()
                cands = [n for n in names if n not in on_field] or list(names)
                sub_name = min(cands, key=lambda n: recent.get(n, 99))  # newcomers are assumed to start
                xi = [n for n in names if n != sub_name][:11]
                imp = (sub_name, _guess_out(xi, first, on_field, hist, roles))
            teams.append((team_name, xi, imp))
        if not ok:
            continue

        all_names = [n for _, xi, imp in teams for n in xi + ([imp[0]] if imp else [])]
        cred = estimate_credits(all_names, roles, form, role_mean, glob)
        built = []
        for team_name, xi, imp in teams:
            order = order_by_position(xi, hist, roles)
            players = {n: Player(n, roles.get(n, "BAT"), cred[n], team_name) for n in order}
            bowl_names = pick_bowlers(order, hist, roles, k=8)
            bowl_w = [hist[n]["bowl_balls"] / 6 if n in hist else 1.0 for n in bowl_names]
            impact = None
            if imp and p_imp > 0:
                # The data only lists a sub who played, so score him as playing (p_used = 1) to match the
                # truth. In live use, set p_used < 1 in the squad file. His job follows the innings order:
                # a bowling sub for the team batting first, a batting sub for the chasing team.
                sub_role = roles.get(imp[0], "BAT")
                impact = Impact(Player(imp[0], sub_role, cred[imp[0]], team_name),
                                "bowl" if team_name == r.team_a else "bat", imp[1], 1.0)
            pos_hist = phase_ov = None
            if cfg.empirical_lineup:
                pos_hist, phase_ov = lineup_history(past_s, bph, order + ([imp[0]] if imp else []))
            built.append(Team(team_name, [players[n] for n in order], [players[n] for n in bowl_names], impact,
                              bowl_weights=bowl_w, pos_hist=pos_hist, phase_overs=phase_ov))
        ta, tb = built

        adj = None
        if cfg.recent_pitch:
            lo = r.date - pd.Timedelta(days=cfg.recent_days)
            rec = balls_f[(balls_f["venue_key"] == r.venue_key) & (balls_f["date"] < r.date) & (balls_f["date"] >= lo)]
            adj = model.recent_pitch(rec, cfg.recent_S) if len(rec) else None
        sim = simulate(model, ta, tb, n_sims=cfg.sims, seed=int(rng.integers(1 << 31)), venue=r.venue_key, league=r.league,
                       afternoon=bool(getattr(r, "afternoon", False)), pitch_adj=adj)
        plist = sim["players"]
        names = [p.name for p in plist]
        idx_of = {n: i for i, n in enumerate(names)}

        # the truth for these players
        ms_stats = stats[stats["match_id"] == r.match_id].set_index("name")
        truth_pts = pd.Series(points_of(ms_stats.reset_index(), roles), index=ms_stats.index)
        actual = np.array([truth_pts.get(n, 0.0) for n in names])
        played = np.array([n in truth_pts.index for n in names])
        # (a sub who was in the model's roster but did not play scores 0 in both worlds)

        mean = sim["points"].mean(axis=0)
        form_vec = np.array([form.get(n, role_mean.get(roles.get(n, "BAT"), glob)) for n in names])
        sp = spearmanr(mean[played], actual[played])[0] if played.sum() > 3 else np.nan
        sp_form = spearmanr(form_vec[played], actual[played])[0] if played.sum() > 3 else np.nan
        p10_, p90_ = np.percentile(sim["points"], [10, 90], axis=0)
        p10, p90 = p10_, p90_
        if collect is not None:
            real_pos = ms_stats["pos"] if "pos" in ms_stats.columns else None
            ap = points_parts(ms_stats.reset_index(), roles).set_index(ms_stats.index)
            sd = sim["points"].std(axis=0)
            order_idx = {p_.name: k_ for k_, p_ in enumerate(list(ta.batting_order) + list(tb.batting_order))}
            for i, p_ in enumerate(plist):
                tm = ta if p_.team == ta.name else tb
                bw = dict(zip([b_.name for b_ in tm.bowlers], tm.bowl_weights or [np.nan] * len(tm.bowlers)))
                collect.append({
                    "match_id": r.match_id, "date": r.date, "league": r.league, "venue": r.venue_key, "name": p_.name,
                    "team": p_.team, "bats_first": p_.team == ta.name, "role": p_.role, "credits": p_.credits,
                    "pred": float(mean[i]), "pred_std": float(sd[i]), "pred_p10": float(p10_[i]), "pred_p90": float(p90_[i]),
                    "pred_bat": float(sim["parts"]["bat"][:, i].mean()), "pred_bowl": float(sim["parts"]["bowl"][:, i].mean()),
                    "pred_field": float(sim["parts"]["field"][:, i].mean()),
                    "form": float(form_vec[i]), "actual": float(actual[i]), "played": bool(played[i]),
                    "act_bat": float(ap["bat"].get(p_.name, np.nan)), "act_bowl": float(ap["bowl"].get(p_.name, np.nan)),
                    "act_field": float(ap["field"].get(p_.name, np.nan)),
                    "plan_pos": order_idx.get(p_.name, np.nan), "bowl_weight": bw.get(p_.name, np.nan),
                    "is_sub": bool(tm.impact and tm.impact.player.name == p_.name),
                    "real_pos": float(real_pos.get(p_.name, np.nan)) if real_pos is not None else np.nan,
                    "real_balls": float(ms_stats["balls"].get(p_.name, 0)), "real_runs": float(ms_stats["runs"].get(p_.name, 0)),
                    "real_bowl_balls": float(ms_stats["bowl_balls"].get(p_.name, 0)), "real_wkts": float(ms_stats["wkts"].get(p_.name, 0)),
                    "hist_matches": float(hist.get(p_.name, {}).get("matches", 0)),
                    "hist_bowl_balls": float(hist.get(p_.name, {}).get("bowl_balls", 0)),
                    "hist_bat_balls": float(hist.get(p_.name, {}).get("bat_balls", 0)),
                    "hist_pos": float(hist.get(p_.name, {}).get("pos", np.nan)),
                })
        top = lambda v: set(np.argsort(-v)[:5])

        best = pick_team(plist, sim["points"])
        form_pick = pick_team(plist, form_vec[None, :])
        oracle = pick_team(plist, actual[None, :])
        mix_vec = np.array([mix_form.get(n, role_mean.get(roles.get(n, "BAT"), glob)) for n in names])
        blends = {f"pts_blend{int(a * 100)}": _score_xi(pick_team(plist, (a * mean + (1 - a) * mix_vec)[None, :]), idx_of, actual)
                  for a in BLENDS}
        sp_blend = spearmanr((0.5 * mean + 0.5 * form_vec)[played], actual[played])[0] if played.sum() > 3 else np.nan
        rand = np.mean([_score_xi(pick_team(plist, rng.random((1, len(plist)))), idx_of, actual)
                        for _ in range(cfg.random_draws)])

        a_tot = inn_tot.loc[r.match_id, 1]
        b_tot = inn_tot.loc[r.match_id, 2]
        past_tot = inn_tot[inn_tot["date"] < r.date]
        at_venue = past_tot[past_tot["venue_key"] == r.venue_key][1]
        base_venue = float(at_venue.mean()) if len(at_venue) >= 3 else float(past_tot[1].mean())
        wa = sim["p_a_wins"] / max(sim["p_a_wins"] + sim["p_b_wins"], 1e-9)
        a_won = (r.winner == r.team_a) if r.winner in (r.team_a, r.team_b) else np.nan
        base_wr = float((past_tot[1] > past_tot[2]).mean())

        rows.append({
            "match_id": r.match_id, "date": r.date, "venue": r.venue_key, "n": int(played.sum()),
            "spearman": sp, "spearman_form": sp_form,
            "mae": float(np.abs(mean - actual)[played].mean()), "mae_form": float(np.abs(form_vec - actual)[played].mean()),
            "top5": len(top(np.where(played, mean, -1)) & top(np.where(played, actual, -1))),
            "top5_form": len(top(np.where(played, form_vec, -1)) & top(np.where(played, actual, -1))),
            "cover": float(((actual >= p10) & (actual <= p90))[played].mean()),
            "pts_model": _score_xi(best, idx_of, actual), "pts_form": _score_xi(form_pick, idx_of, actual),
            "pts_random": float(rand), "pts_oracle": _score_xi(oracle, idx_of, actual), **blends,
            "spearman_blend50": sp_blend,
            "pred_a": sim["avg_score_a"], "pred_b": sim["avg_score_b"], "act_a": a_tot, "act_b": b_tot,
            "base_a": base_venue, "p_a_win": wa, "a_won": a_won, "base_p": base_wr,
        })
        if progress and (k + 1) % 10 == 0:
            print(f"  [{cfg.label}] {k + 1}/{len(test)} matches, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def _boot(diff: np.ndarray, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(diff, len(diff)).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["pts_model"])
    out = {"matches": len(d)}
    for k in ("pts_model", "pts_form", "pts_random", "pts_oracle"):
        out[k] = float(d[k].mean())
    out["pct_of_oracle_model"] = out["pts_model"] / out["pts_oracle"]
    out["pct_of_oracle_form"] = out["pts_form"] / out["pts_oracle"]
    for other in ("form", "random"):
        diff = (d["pts_model"] - d[f"pts_{other}"]).to_numpy()
        lo, hi = _boot(diff)
        out[f"model_minus_{other}"] = {"mean": float(diff.mean()), "ci95": [lo, hi],
                                       "win_rate": float((diff > 0).mean())}
    for a in BLENDS:
        k = f"pts_blend{int(a * 100)}"
        diff = (d[k] - d["pts_form"]).to_numpy()
        out[k] = {"pts": float(d[k].mean()), "minus_form": float(diff.mean()), "ci95": list(_boot(diff))}
    out["spearman_blend50"] = float(d["spearman_blend50"].mean())
    out["spearman"] = float(d["spearman"].mean())
    out["spearman_form"] = float(d["spearman_form"].mean())
    out["mae"], out["mae_form"] = float(d["mae"].mean()), float(d["mae_form"].mean())
    out["top5"], out["top5_form"] = float(d["top5"].mean()), float(d["top5_form"].mean())
    out["coverage_p10_p90"] = float(d["cover"].mean())
    out["score_mae_model"] = float(np.abs(d["pred_a"] - d["act_a"]).mean())
    out["score_mae_venue_avg"] = float(np.abs(d["base_a"] - d["act_a"]).mean())
    w = d.dropna(subset=["a_won"])
    if len(w):
        y = w["a_won"].astype(float).to_numpy()
        out["brier_model"] = float(((w["p_a_win"] - y) ** 2).mean())
        out["brier_base_rate"] = float(((w["base_p"] - y) ** 2).mean())
        out["accuracy_model"] = float(((w["p_a_win"] > 0.5) == (y == 1)).mean())
    return out


def run_ablation(balls_f, matches, stats, base_cfg: BTConfig, progress=True) -> dict:
    res = {}
    for name, feats in STAGES:
        cfg = BTConfig(**{**base_cfg.__dict__, "features": feats, "label": name})
        df = run_backtest(balls_f, matches, stats, cfg, progress=progress)
        res[name] = summarize(df)
    return res


def prepare(balls: pd.DataFrame, matches: pd.DataFrame):
    balls_f = add_features(balls)
    return balls_f, actual_stats(balls_f, matches)


def format_report(summary: dict, title="Backtest") -> str:
    s = summary
    L = [f"# {title}", "", f"Matches tested: {s['matches']}", "",
         "## Fantasy team quality (average Dream11 points of the picked XI, captain 2x, vice 1.5x)", "",
         "| Method | Avg points | % of best-possible |", "|---|---:|---:|",
         f"| Model | {s['pts_model']:.1f} | {s['pct_of_oracle_model']:.1%} |",
         f"| Form baseline (recent average) | {s['pts_form']:.1f} | {s['pct_of_oracle_form']:.1%} |",
         f"| Random valid team | {s['pts_random']:.1f} | {s['pts_random'] / s['pts_oracle']:.1%} |",
         f"| Best possible (hindsight) | {s['pts_oracle']:.1f} | 100% |", ""]
    for other in ("form", "random"):
        x = s[f"model_minus_{other}"]
        L.append(f"- Model minus {other}: **{x['mean']:+.1f} pts per match** "
                 f"(95% range {x['ci95'][0]:+.1f} to {x['ci95'][1]:+.1f}); model won {x['win_rate']:.0%} of matches")
    L += ["", "## Mixing the simulator with the form baseline", "",
          "| Share of simulator | Avg points | vs form baseline (95% range) |", "|---|---:|---|"]
    for a in BLENDS:
        b = s[f"pts_blend{int(a * 100)}"]
        L.append(f"| {a:.0%} | {b['pts']:.1f} | {b['minus_form']:+.1f} ({b['ci95'][0]:+.1f} to {b['ci95'][1]:+.1f}) |")
    L += ["", "## Player predictions", "",
          f"- Rank correlation with real points: model {s['spearman']:.3f}, form baseline {s['spearman_form']:.3f}, 50/50 mix {s['spearman_blend50']:.3f}",
          f"- Average error per player: model {s['mae']:.1f}, form baseline {s['mae_form']:.1f}",
          f"- Predicted top-5 players that really were top-5: model {s['top5']:.2f}, form {s['top5_form']:.2f} (of 5)",
          f"- Real points inside the model's 10-90% range: {s['coverage_p10_p90']:.0%} (80% would be perfectly calibrated)",
          "", "## Match predictions", "",
          f"- First-innings score error: model {s['score_mae_model']:.1f} runs, ground average {s['score_mae_venue_avg']:.1f} runs"]
    if "brier_model" in s:
        L.append(f"- Win prediction: accuracy {s['accuracy_model']:.1%}; Brier score model {s['brier_model']:.4f} "
                 f"vs base rate {s['brier_base_rate']:.4f} (lower is better)")
    return "\n".join(L)


def format_ablation(res: dict) -> str:
    L = ["| Step | Avg points of XI | vs form baseline | Rank corr. | Score error | Brier |", "|---|---:|---:|---:|---:|---:|"]
    for name, s in res.items():
        L.append(f"| {name} | {s['pts_model']:.1f} | {s['model_minus_form']['mean']:+.1f} | {s['spearman']:.3f} | "
                 f"{s['score_mae_model']:.1f} | {s.get('brier_model', float('nan')):.4f} |")
    return "\n".join(L)
