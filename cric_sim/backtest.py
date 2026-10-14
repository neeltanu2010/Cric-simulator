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
  * Impact player identity is taken from the real match, so the "impact" feature looks a bit
    better here than it can in real life.
  * Credits are estimated from form, not the real Dream11 credits.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .actual import actual_stats, points_of
from .features import add_features
from .lineup import (estimate_credits, form_points, history_table, infer_roles, order_by_position,
                     pick_bowlers)
from .model import ALL_FEATURES, RateModel
from .optimizer import pick_team
from .players import bowl_types, load_player_csv
from .simulator import Impact, Player, Team, simulate

STAGES = [  # name, features switched on (cumulative)
    ("players only", ()),
    ("+ ground", ("venue",)),
    ("+ pace/spin", ("venue", "type")),
    ("+ innings/toss", ("venue", "type", "innings")),
    ("+ game situation", ("venue", "type", "innings", "state")),
    ("+ extras", ("venue", "type", "innings", "state", "extras")),
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


def _team_xi(names, sub, hist):
    """Return (starting XI names, impact (in, out) or None)."""
    if len(names) == 11:
        return list(names), None
    if len(names) == 12 and sub:
        return [n for n in names if n != sub[0]], sub
    return None, None


def _impact_rate(matches: pd.DataFrame, before: pd.Timestamp) -> float:
    m = matches[matches["date"] < before].tail(80)
    if m.empty:
        return 0.0
    used = np.concatenate([(m["players_a"].map(len) > 11).to_numpy(), (m["players_b"].map(len) > 11).to_numpy()])
    return float(np.clip(used.mean(), 0.0, 0.95))


def _score_xi(res, idx_of, actual):
    xi = [idx_of[p.name] for p in res["team"]]
    c, v = idx_of[res["captain"].name], idx_of[res["vice_captain"].name]
    return float(sum(actual[i] for i in xi) + actual[c] + 0.5 * actual[v])


def run_backtest(balls_f: pd.DataFrame, matches: pd.DataFrame, stats: pd.DataFrame, cfg: BTConfig,
                 progress: bool = False) -> pd.DataFrame:
    """Returns one row per test match."""
    rng = np.random.default_rng(cfg.seed)
    ms = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
    start = max(cfg.min_history, len(ms) - cfg.n_test)
    test = ms.iloc[start:]
    csv_over = load_player_csv(cfg.types_csv)
    inn_tot = balls_f.groupby(["match_id", "innings"])["runs_total"].sum().unstack()
    inn_tot = inn_tot.join(ms.set_index("match_id")[["venue_key", "date"]])

    rows, model, model_n = [], None, 10 ** 9
    t0 = time.time()
    for k, r in enumerate(test.itertuples()):
        past_b = None
        if model is None or model_n >= cfg.refit_every:
            past_b = balls_f[balls_f["date"] < r.date]
            types = dict(bowl_types(past_b, cfg.types_csv))
            if cfg.types_override:
                types.update(cfg.types_override)
            model = RateModel(features=cfg.features).fit(past_b, types)
            model_n = 0
        model_n += 1

        past_s = stats[stats["date"] < r.date]
        hist = history_table(past_s)
        roles = infer_roles(hist, csv_over)
        form, role_mean, glob = form_points(past_s, roles)

        teams = []
        ok = True
        p_imp = _impact_rate(ms, r.date) if model.has("impact") else 0.0
        for team_name, names, first in ((r.team_a, r.players_a, True), (r.team_b, r.players_b, False)):
            sub = None
            for x in r.replacements:
                if x["team"] == team_name and x["in"] in names and x["out"] in names:
                    sub = (x["in"], x["out"])
                    break
            xi, imp = _team_xi(names, sub, hist)
            if xi is None:
                # 12 players listed but no substitution info: use the 11 with most matches played
                if len(names) > 11:
                    xi = sorted(names, key=lambda n: -hist.get(n, {}).get("matches", 0))[:11]
                else:
                    ok = False
                    break
                imp = None
            teams.append((team_name, xi, imp))
        if not ok:
            continue

        all_names = [n for _, xi, imp in teams for n in xi + ([imp[0]] if imp else [])]
        cred = estimate_credits(all_names, roles, form, role_mean, glob)
        built = []
        for team_name, xi, imp in teams:
            order = order_by_position(xi, hist, roles)
            players = {n: Player(n, roles.get(n, "BAT"), cred[n], team_name) for n in order}
            bowl_names = pick_bowlers(order, hist, roles)
            impact = None
            if imp and p_imp > 0:
                sub_role = roles.get(imp[0], "BAT")
                is_bowler = sub_role in ("BOWL", "AR") and hist.get(imp[0], {}).get("bowl_balls", 0) >= 6
                impact = Impact(Player(imp[0], sub_role, cred[imp[0]], team_name),
                                "bowl" if is_bowler else "bat", imp[1], p_imp)
            built.append(Team(team_name, [players[n] for n in order], [players[n] for n in bowl_names], impact))
        ta, tb = built

        sim = simulate(model, ta, tb, n_sims=cfg.sims, seed=int(rng.integers(1 << 31)), venue=r.venue_key)
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
        top = lambda v: set(np.argsort(-v)[:5])
        p10, p90 = np.percentile(sim["points"], [10, 90], axis=0)

        best = pick_team(plist, sim["points"])
        form_pick = pick_team(plist, form_vec[None, :])
        oracle = pick_team(plist, actual[None, :])
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
            "pts_random": float(rand), "pts_oracle": _score_xi(oracle, idx_of, actual),
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
    L += ["", "## Player predictions", "",
          f"- Rank correlation with real points: model {s['spearman']:.3f}, form baseline {s['spearman_form']:.3f}",
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
