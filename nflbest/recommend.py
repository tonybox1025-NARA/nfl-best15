"""한 주차 전체 파이프라인: 피처 → 날씨 반영 → 예측 → 부상 반영 → 배당 → 베스트 15 → JSON."""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, features, live, model, picks

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
BUNDLE = data.CACHE / "model.pkl"
FIRST_SEASON = 2020


def get_bundle(feat: pd.DataFrame, season: int, retrain_hours: float = 20) -> model.Bundle:
    if BUNDLE.exists() and time.time() - BUNDLE.stat().st_mtime < retrain_hours * 3600:
        b = model.Bundle.load(BUNDLE)
        if getattr(b, "version", 0) == model.VERSION:
            return b
    b = model.fit(feat, list(range(FIRST_SEASON, season)))
    b.save(BUNDLE)
    return b


def run(season: int | None = None, week: int | None = None, refresh_only: bool = False,
        force_odds: bool = False) -> dict:
    t0 = time.time()
    today = date.today().isoformat()
    season = season or (date.today().year if date.today().month >= 8 else date.today().year - 1)
    seasons = list(range(FIRST_SEASON, season + 1))
    sched = data.schedules(seasons, max_age_hours=1 if refresh_only else 6)
    week = week or data.current_week(sched, season, today)
    games = sched[(sched.season == season) & (sched.week == week) & sched.home_score.isna()].copy()

    feat = features.build(seasons, season, week)
    bundle = get_bundle(feat, season, retrain_hours=1e9 if refresh_only else 20)

    # ---- 날씨: 다가오는 경기의 바람 피처를 예보값으로 교체
    wx, wx_msg = live.weather(games)
    up = feat[feat.upcoming].copy()
    for gid, w in wx.items():
        if "wind_mph" in w:
            up.loc[up.game_id == gid, "wind"] = w["wind_mph"]

    pred = model.predict(bundle, up)
    from . import role
    role.attach(pred, role.tables(feat, season))

    # ---- 부상
    inj, inj_msg = live.injuries(season, week)
    pred["status"] = pred.player_display_name.map(lambda n: inj.get(picks.name_key(n), {}).get("status", ""))
    pred["injury"] = pred.player_display_name.map(lambda n: inj.get(picks.name_key(n), {}).get("injury", ""))
    pred = _starting_qbs(pred)     # 팀당 주전 QB 1명만 (백업·장기 결장 QB 제외)
    pred0 = pred.copy()            # 부상 반영 전 (시나리오 계산용)
    notes = _redistribute(pred)
    pred = pred[~pred.status.map(live.is_out)]

    # ---- 배당
    od, od_msg = live.odds(games, force=force_odds)

    cand = picks.candidates(bundle, pred, od)
    status_by_pid = pred.set_index("player_id").status.to_dict()
    cand["status"] = cand.player_id.map(status_by_pid).fillna("")
    has_odds = "edge" in cand and cand.edge.notna().any()
    b_ref[0] = bundle
    if len(cand):
        cand["detail"] = _details(cand, feat, pred)
    scen = _scenarios(bundle, pred0, cand, od, feat)

    def pack(df):
        if df is None or df.empty:
            return []
        cols = ["rank", "player_id", "name", "pos", "team", "opp", "market", "line", "label", "prob", "status", "game_id"]
        cols += [c for c in ("odds", "implied", "edge", "model_prob", "market_prob", "check", "detail") if c in df]
        recs = df[cols].replace({np.nan: None}).to_dict("records")
        for r in recs:
            if r.get("check") is not None:
                r["check"] = bool(r["check"])
            for k in ("prob", "implied", "edge", "model_prob", "market_prob"):
                if r.get(k) is not None:
                    r[k] = round(float(r[k]) * 100, 1)
        return recs

    out_games = []
    for g in games.sort_values(["gameday", "gametime"]).itertuples():
        gc = cand[cand.game_id == g.game_id]
        home_imp = g.total_line / 2 + g.spread_line / 2
        out_games.append({
            "game_id": g.game_id, "away": g.away_team, "home": g.home_team,
            "kickoff": f"{g.gameday} {g.gametime} ET", "stadium": g.stadium, "roof": g.roof,
            "spread_home": g.spread_line, "total": g.total_line,
            "proj": {"away": round(g.total_line - home_imp, 1), "home": round(home_imp, 1)},
            "weather": wx.get(g.game_id), "notes": [n for n in notes if n["game_id"] == g.game_id],
            "scenarios": scen.get(g.game_id, []),
            "injuries": pred0[(pred0.game_id == g.game_id) & (pred0.status != "")][
                ["player_display_name", "team", "position", "status", "injury"]].to_dict("records"),
            "best15_prob": pack(picks.best15(gc, "prob")),
            "best15_edge": pack(picks.best15(gc, "edge")) if has_odds else [],
        })

    result = {
        "season": season, "week": week, "generated_at": live.now_et(),
        "sources": {"injuries": inj_msg, "weather": wx_msg, "odds": od_msg},
        "has_odds": bool(has_odds),
        "overall": {
            "prob": pack(picks.best15(cand, "prob")),
            "edge": pack(picks.best15(cand, "edge")) if has_odds else [],
            "td": pack(picks.best15(cand[cand.market == "td"], "prob")),
            "check": pack(_checks(cand)),
        },
        "games": out_games,
        "out_players": sorted({n["name"] for n in notes}),
        "seconds": round(time.time() - t0, 1),
    }
    result["changes"] = _changes(result)
    result = _clean(result)
    path = OUT / f"week{week}.json"
    if path.exists():
        (OUT / f"week{week}.prev.json").write_text(path.read_text())
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str, allow_nan=False))
    (OUT / "latest.json").write_text(path.read_text())
    from . import history
    history.update(result)
    return result


STAT_COL = {"td": "td_any", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards",
            "receptions": "receptions", "pass_yds": "passing_yards", "pass_tds": "passing_tds"}
DEF_COL = {"td": "def_a_td_l8", "rush_yds": "def_a_rush_l8", "rec_yds": "def_a_rec_l8",
           "receptions": "def_a_recs_l8", "pass_yds": "def_a_pass_l8", "pass_tds": "def_a_ptd_l8"}


def _f(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


b_ref = [None]   # 확률 범위 계산용 모델 (run에서 채움)


def _role_detail(market: str, u) -> dict | None:
    """역할 기반 계산 내역 (앱 설명용)."""
    if market == "rush_yds":
        return {"kind": "rush", "team_car": _f(u.get("team_car_exp")), "share": _f(u.get("car_share"), 3),
                "exp_car": _f(u.get("exp_car")), "ypc": _f(u.get("ypc"), 2), "def": _f(u.get("d_rush"), 2),
                "role_mu": _f(u.get("mu_rush_role")), "model_mu": _f(u.get("mu_rush_yds_model"))}
    if market in ("rec_yds", "receptions"):
        return {"kind": "rec" if market == "rec_yds" else "recs", "team_pass": _f(u.get("team_pass_exp")),
                "share": _f(u.get("tgt_share"), 3), "exp_tgt": _f(u.get("exp_tgt")), "ypt": _f(u.get("ypt"), 2),
                "catch": _f(u.get("catch"), 2), "def": _f(u.get("d_rec"), 2),
                "role_mu": _f(u.get("mu_rec_role" if market == "rec_yds" else "mu_recs_role")),
                "model_mu": _f(u.get("mu_rec_yds_model" if market == "rec_yds" else "mu_receptions_model"))}
    return None


def _range(b, market: str, mu) -> list | None:
    """예상 범위 (25%~75% 구간): 비슷한 예상값 선수들의 실제 결과 분포에서."""
    if b is None or market == "td" or mu is None or (isinstance(mu, float) and np.isnan(mu)):
        return None
    name = {"rush_yds": "rush_yds", "rec_yds": "rec_yds", "receptions": "receptions",
            "pass_yds": "pass_yds", "pass_tds": "pass_tds"}[market]
    r = b.resid[name]
    i = int(np.clip(np.searchsorted(r["edges"], mu, side="right") - 1, 0, len(r["ratios"]) - 1))
    q = np.quantile(r["ratios"][i] * mu, [0.25, 0.75])
    return [round(float(q[0]), 1 if name in ("receptions", "pass_tds") else 0), round(float(q[1]), 1 if name in ("receptions", "pass_tds") else 0)]


def _details(cand: pd.DataFrame, feat: pd.DataFrame, pred: pd.DataFrame) -> list[dict]:
    """픽 상세 화면용 근거: 최근 5경기 기록, 상대 수비 허용 순위, 사용량, 팀 예상 득점."""
    hist = feat[~feat.upcoming].sort_values("gameday")
    hist = hist[hist.player_id.isin(set(cand.player_id))]
    last5 = {pid: g.tail(5) for pid, g in hist.groupby("player_id")}
    up = pred.set_index("player_id")
    ranks = {}
    for m, col in DEF_COL.items():
        for pos, g in pred.groupby("position"):
            d = g.drop_duplicates("opponent_team").set_index("opponent_team")[col]
            ranks[(m, pos)] = d.rank(ascending=False, method="min").to_dict()
    out = []
    for r in cand.itertuples():
        u = up.loc[r.player_id]
        u = u.iloc[0] if isinstance(u, pd.DataFrame) else u
        h = last5.get(r.player_id)
        col = STAT_COL[r.market]
        recent = [] if h is None else [
            {"wk": int(x.week), "opp": x.opponent_team, "val": _f(getattr(x, col), 0)} for x in h.itertuples()]
        line = None if r.market == "td" else (r.line + 0.5 if r.line is not None else None)
        hits = None
        if recent:
            vals = [x["val"] for x in recent if x["val"] is not None]
            hits = sum(v >= 1 for v in vals) if r.market == "td" else sum(v >= line for v in vals)
        out.append({
            "recent": recent, "need": line, "recent_hits": hits,
            "mu": _f(getattr(r, "mu", None)),
            "def_rank": ranks.get((r.market, r.pos), {}).get(r.opp),
            "def_allowed": _f(u.get(DEF_COL[r.market])),
            "carry_share": _f(u.get("carry_share_ewm"), 3), "tgt_share": _f(u.get("tgt_share_ewm"), 3),
            "rz_share": _f(u.get("rz_share_ewm"), 3), "implied_total": _f(u.get("implied_total")),
            "spread": _f(u.get("team_spread")), "wind": _f(u.get("wind"), 0),
            "td_rate": _f(u.get("td_rate_l16"), 3),
            "role": _role_detail(r.market, u),
            "range": _range(b_ref[0], r.market, getattr(r, "mu", None)),
            "new_role": bool(u.get("role_changed")) if not pd.isna(u.get("role_changed", np.nan)) else False,
        })
    return out


def _clean(o):
    """JSON에 NaN/Infinity가 들어가면 브라우저가 못 읽어요 → null로 바꿔요."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _starting_qbs(pred: pd.DataFrame) -> pd.DataFrame:
    """팀마다 이번 주 선발 QB 한 명만 남겨요.
    기준: 결장이 아닌 QB 중 팀 직전 경기에서 패스를 가장 많이 던진 선수 (없으면 최근 3경기 평균 시도 수).
    → 몇 주째 결장 중인 QB나 백업 QB가 예측·시나리오에 끼지 않아요."""
    qb = pred[pred.position == "QB"].copy()
    if qb.empty:
        return pred
    qb["_out"] = qb.status.map(live.is_out)
    qb["_k1"] = qb.get("qb_last_att", pd.Series(0, index=qb.index)).fillna(0)
    qb["_k2"] = qb.attempts_l3.fillna(0)
    keep = (qb[~qb._out].sort_values(["_k1", "_k2"], ascending=False).groupby("team").head(1).index)
    starters_out = qb[qb._out & (qb._k1 > 0)]          # 지난주 선발했는데 이번 주 결장 → 시나리오용으로 유지
    drop = qb.index.difference(keep).difference(starters_out.index)
    return pred.drop(index=drop)


SCEN_STATUSES = {"questionable", "doubtful", "out", "훈련 불참"}


def _apply(pred0: pd.DataFrame, force: dict) -> pd.DataFrame:
    """force={player_id: 'Out'|''} 로 상태를 바꾼 뒤 결장 재분배까지 적용한 팀 예측."""
    t = pred0.copy()
    for pid, st in force.items():
        t.loc[t.player_id == pid, "status"] = st
        if st == "Out" and pid not in set(t[t.in_last_game.fillna(False).astype(bool)].player_id):
            t.loc[t.player_id == pid, "in_last_game"] = True   # 불확실 선수가 빠지는 경우: 몫을 동료에게 나눠줘요
    _redistribute(t)
    return t[~t.status.map(live.is_out)]


def _profile(feat: pd.DataFrame, pid: str, row) -> dict:
    """대체 선수 소개: 최근 3경기, 이번 시즌 누적, 팀 내 사용량."""
    h = feat[(feat.player_id == pid) & ~feat.upcoming].sort_values("gameday")
    season = int(row["season"])
    cur = h[h.season == season]
    last3 = [{"wk": int(x.week), "opp": x.opponent_team, "car": _f(x.carries, 0), "rush": _f(x.rushing_yards, 0),
              "tgt": _f(x.targets, 0), "rec": _f(x.receptions, 0), "recyd": _f(x.receiving_yards, 0),
              "td": _f(x.td_any, 0)} for x in h.tail(3).itertuples()]
    return {"name": row["player_display_name"], "pos": row["position"], "team": row["team"],
            "games": int(len(cur)), "rush_yds": _f(cur.rushing_yards.sum(), 0), "rec_yds": _f(cur.receiving_yards.sum(), 0),
            "rec": _f(cur.receptions.sum(), 0), "tds": _f(cur.td_any.sum(), 0),
            "carry_share": _f(row.get("carry_share_ewm"), 3), "tgt_share": _f(row.get("tgt_share_ewm"), 3),
            "last3": last3}


def _scenarios(b, pred0: pd.DataFrame, cand: pd.DataFrame, od, feat: pd.DataFrame) -> dict:
    """출전이 불확실한(또는 결장한) 주요 선수마다: 뛸 때 vs 안 뛸 때 동료 확률, 대체 선수와 그 선수의 픽."""
    out = {}
    rel = pred0[pred0.status.str.lower().isin(SCEN_STATUSES)]
    # 이미 몇 주째 결장 중인 선수는 제외 (팀 데이터에 이미 그 선수 없이 반영돼 있어요)
    rel = rel[~(rel.status.map(live.is_out) & ~rel.in_last_game.fillna(False).astype(bool))]
    rel = rel[(rel.position == "QB") | (rel.carry_share_ewm.fillna(0) >= 0.15) | (rel.tgt_share_ewm.fillna(0) >= 0.10)]
    for r in rel.itertuples():
        team = pred0[pred0.team == r.team]
        play = _apply(team, {r.player_id: ""}).set_index("player_id")
        miss = _apply(team, {r.player_id: "Out"}).set_index("player_id")
        # 동료 픽: 현재 후보 중 같은 팀 상위 픽 + 결장 시 새로 생기는 픽
        base = cand[(cand.team == r.team) & (cand.player_id != r.player_id)].sort_values("prob", ascending=False).head(10)
        effects = []
        for c in base.itertuples():
            if c.player_id not in play.index or c.player_id not in miss.index:
                continue
            bd = c._asdict()
            pp = picks.prob_for(b, play.loc[c.player_id], c.market, c.line, bd)
            pm = picks.prob_for(b, miss.loc[c.player_id], c.market, c.line, bd)
            if np.isnan(pp) or np.isnan(pm) or abs(pm - pp) < 0.01:
                continue
            effects.append({"player_id": c.player_id, "name": c.name, "label": c.label, "market": c.market,
                            "line": c.line, "prob_play": round(pp * 100, 1), "prob_out": round(pm * 100, 1)})
        # 대체 선수: 같은 포지션에서 결장 시 예상값이 가장 많이 오르는 선수
        key = {"RB": "mu_rush_yds", "QB": "mu_pass_yds"}.get(r.position, "mu_rec_yds")
        same = [pid for pid in miss.index if pid != r.player_id and miss.loc[pid, "position"] == r.position and pid in play.index]
        repl = None
        if same:
            gain = {pid: (miss.loc[pid, key] if not pd.isna(miss.loc[pid, key]) else 0) -
                         (play.loc[pid, key] if not pd.isna(play.loc[pid, key]) else 0) for pid in same}
            rid = max(gain, key=gain.get)
            if gain[rid] > 0.5:
                rrow = miss.loc[rid].copy(); rrow["player_display_name"] = miss.loc[rid, "player_display_name"]
                prof = _profile(feat, rid, rrow)
                rc = picks.candidates(b, miss.loc[[rid]].reset_index(), od)
                rp = []
                if len(rc):
                    for c in rc.sort_values("prob", ascending=False).head(4).itertuples():
                        bd = c._asdict()
                        pp = picks.prob_for(b, play.loc[rid], c.market, c.line, bd)
                        rp.append({"label": c.label, "market": c.market, "prob_out": round(c.prob * 100, 1),
                                   "prob_play": None if np.isnan(pp) else round(pp * 100, 1)})
                prof["picks"] = rp
                prof["mu_play"] = _f(play.loc[rid, key]); prof["mu_out"] = _f(miss.loc[rid, key]); prof["stat"] = key[3:]
                repl = prof
        own = cand[cand.player_id == r.player_id].sort_values("prob", ascending=False).head(3)
        out.setdefault(r.game_id, []).append({
            "player_id": r.player_id, "name": r.player_display_name, "pos": r.position, "team": r.team,
            "status": r.status, "injury": r.injury,
            "carry_share": _f(r.carry_share_ewm, 3), "tgt_share": _f(r.tgt_share_ewm, 3),
            "own_picks": [{"label": c.label, "prob": round(c.prob * 100, 1)} for c in own.itertuples()],
            "effects": sorted(effects, key=lambda e: -abs(e["prob_out"] - e["prob_play"]))[:6],
            "replacement": repl,
        })
    return out


def _checks(cand: pd.DataFrame) -> pd.DataFrame:
    """모델과 시장 차이가 큰 픽 (모델이 모르는 역할 변화·부상 가능성) — 순위에서 빠진 것들."""
    if "check" not in cand:
        return cand.iloc[0:0]
    c = cand[cand.check.fillna(False).astype(bool)].copy()
    c["gap"] = (c.model_prob - c.market_prob).abs()
    c = c.sort_values("gap", ascending=False).head(15)
    c.insert(0, "rank", range(1, len(c) + 1))
    return c


ROLE_W = 0.5   # 역할이 바뀐 선수: 최종 예상 = 모델(과거 기록) 50% + 역할 기반 계산 50%


def _redistribute(pred: pd.DataFrame) -> list[dict]:
    """결장 선수의 타깃/캐리 70%를 같은 팀 동료에게 점유율 비례로 나눠줘요.
    몫이 늘어난 선수는 '새 역할'로 역할 기반 예상(사용량 × 효율 × 상대 수비)을 다시 계산해서 섞어요."""
    from . import role
    notes = []
    for team, t in pred.groupby("team"):
        out = t[t.status.map(live.is_out) & t.in_last_game.fillna(False).astype(bool)]
        if out.empty:
            continue
        act = t[~t.status.map(live.is_out)]
        catch = act[act.position.isin(["RB", "WR", "TE"])]
        rbs = act[act.position == "RB"]
        vac_t = out[out.position.isin(["RB", "WR", "TE"])].tgt_share.fillna(0).sum()
        vac_c = out[out.position == "RB"].car_share.fillna(0).sum()
        st = max(catch.tgt_share.sum(), 0.05)
        sc = max(rbs.car_share.sum(), 0.05)
        old_t = pred.loc[catch.index, "tgt_share"].copy()
        old_c = pred.loc[rbs.index, "car_share"].copy()
        pred.loc[catch.index, "tgt_share"] = old_t + 0.7 * vac_t * old_t / st
        pred.loc[rbs.index, "car_share"] = (old_c + 0.7 * vac_c * old_c / sc).clip(upper=0.8)
        idx = catch.index
        role.compute(pred, idx)
        ft = (pred.loc[idx, "tgt_share"] / old_t.reindex(idx).replace(0, np.nan)).fillna(1).clip(upper=2.0)
        fc = (pred.loc[idx, "car_share"] / pred.loc[idx, "car_share"].where(~idx.isin(rbs.index), old_c.reindex(idx)).replace(0, np.nan)).fillna(1).clip(upper=2.0)
        fc = fc.where(idx.isin(rbs.index), 1.0)
        for i in idx:
            if ft[i] > 1.001:
                pred.loc[i, "mu_rec_yds"] = (1 - ROLE_W) * pred.loc[i, "mu_rec_yds"] * ft[i] + ROLE_W * pred.loc[i, "mu_rec_role"]
                pred.loc[i, "mu_receptions"] = (1 - ROLE_W) * pred.loc[i, "mu_receptions"] * ft[i] + ROLE_W * pred.loc[i, "mu_recs_role"]
            if fc[i] > 1.001 and not pd.isna(pred.loc[i, "mu_rush_yds"]):
                pred.loc[i, "mu_rush_yds"] = (1 - ROLE_W) * pred.loc[i, "mu_rush_yds"] * fc[i] + ROLE_W * pred.loc[i, "mu_rush_role"]
        cs, ts = pred.loc[idx, "car_share"], pred.loc[idx, "tgt_share"]
        w = (cs / (cs + ts).replace(0, np.nan)).fillna(0)
        f_td = w * fc + (1 - w) * ft
        pred.loc[idx, "p_td"] = 1 - (1 - pred.loc[idx, "p_td"]) ** f_td
        pred.loc[idx, "role_changed"] = (ft > 1.001) | (fc > 1.001)
        if (out.position == "QB").any():  # 주전 QB 결장: 패스 지표 하향
            pred.loc[idx, ["mu_rec_yds", "mu_receptions"]] *= 0.9
        for r in out.itertuples():
            notes.append({"game_id": r.game_id, "team": team, "name": r.player_display_name,
                          "pos": r.position, "status": r.status,
                          "effect": f"동료 타깃 x{float(ft.max()):.2f}, RB 캐리 x{float(fc.max()):.2f}"})
    return notes


def _changes(result: dict) -> list[str]:
    """직전 실행 대비 바뀐 점 (업데이트 버튼 누른 뒤 보여줄 요약)."""
    prev_path = OUT / f"week{result['week']}.json"
    if not prev_path.exists():
        return []
    prev = json.loads(prev_path.read_text())
    msgs = []
    old = {(p["name"], p["label"]): p for p in prev["overall"]["prob"]}
    new = {(p["name"], p["label"]): p for p in result["overall"]["prob"]}
    for k, p in new.items():
        if k not in old:
            msgs.append(f"▲ {p['name']} {p['label']} 새로 진입 ({p['rank']}위, {p['prob']}%)")
        elif abs(p["prob"] - old[k]["prob"]) >= 2:
            msgs.append(f"• {p['name']} {p['label']} {old[k]['prob']}% → {p['prob']}%")
    for k, p in old.items():
        if k not in new:
            msgs.append(f"▼ {p['name']} {p['label']} 베스트 15에서 빠짐")
    newly_out = set(result["out_players"]) - set(prev.get("out_players", []))
    msgs += [f"✕ {n} 결장 반영" for n in sorted(newly_out)]
    return msgs
