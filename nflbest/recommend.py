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

    # ---- 부상
    inj, inj_msg = live.injuries(season, week)
    pred["status"] = pred.player_display_name.map(lambda n: inj.get(picks.name_key(n), {}).get("status", ""))
    pred["injury"] = pred.player_display_name.map(lambda n: inj.get(picks.name_key(n), {}).get("injury", ""))
    notes = _redistribute(pred)
    pred = pred[~pred.status.map(live.is_out)]

    # ---- 배당
    od, od_msg = live.odds(games, force=force_odds)

    cand = picks.candidates(bundle, pred, od)
    status_by_pid = pred.set_index("player_id").status.to_dict()
    cand["status"] = cand.player_id.map(status_by_pid).fillna("")
    has_odds = "edge" in cand and cand.edge.notna().any()

    def pack(df):
        cols = ["rank", "name", "pos", "team", "opp", "label", "prob", "status", "game_id"]
        cols += [c for c in ("odds", "implied", "edge", "model_prob", "market_prob", "check") if c in df]
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
            "injuries": pred[(pred.game_id == g.game_id) & (pred.status != "")][
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
        },
        "games": out_games,
        "out_players": sorted({n["name"] for n in notes}),
        "seconds": round(time.time() - t0, 1),
    }
    result["changes"] = _changes(result)
    path = OUT / f"week{week}.json"
    if path.exists():
        (OUT / f"week{week}.prev.json").write_text(path.read_text())
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    (OUT / "latest.json").write_text(path.read_text())
    return result


def _redistribute(pred: pd.DataFrame) -> list[dict]:
    """결장 선수의 타깃/캐리를 같은 팀 동료에게 나눠줘요 (70%만 — 나머지는 다른 곳으로 샘)."""
    notes = []
    for team, t in pred.groupby("team"):
        out = t[t.status.map(live.is_out) & t.in_last_game.fillna(False).astype(bool)]
        if out.empty:
            continue
        act = t[~t.status.map(live.is_out)]
        catch = act[act.position.isin(["RB", "WR", "TE"])]
        rbs = act[act.position == "RB"]
        vac_t = out[out.position.isin(["RB", "WR", "TE"])].tgt_share_ewm.fillna(0).sum()
        vac_c = out[out.position == "RB"].carry_share_ewm.fillna(0).sum()
        ft = min(1 + 0.7 * vac_t / max(catch.tgt_share_ewm.fillna(0).sum(), 0.05), 1.5)
        fc = min(1 + 0.7 * vac_c / max(rbs.carry_share_ewm.fillna(0).sum(), 0.05), 1.6)
        idx_c, idx_r = catch.index, rbs.index
        pred.loc[idx_c, "mu_rec_yds"] *= ft
        pred.loc[idx_c, "mu_receptions"] *= ft
        pred.loc[idx_r, "mu_rush_yds"] *= fc
        cs, ts = pred.loc[idx_c, "carry_share_ewm"].fillna(0), pred.loc[idx_c, "tgt_share_ewm"].fillna(0)
        w = (cs / (cs + ts).replace(0, np.nan)).fillna(0)
        f_td = w * fc + (1 - w) * ft
        pred.loc[idx_c, "p_td"] = 1 - (1 - pred.loc[idx_c, "p_td"]) ** f_td
        if (out.position == "QB").any():  # 주전 QB 결장: 패스 지표 하향
            pred.loc[idx_c, ["mu_rec_yds", "mu_receptions"]] *= 0.9
        for r in out.itertuples():
            notes.append({"game_id": r.game_id, "team": team, "name": r.player_display_name,
                          "pos": r.position, "status": r.status,
                          "effect": f"동료 타깃 x{ft:.2f}, RB 캐리 x{fc:.2f}"})
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
