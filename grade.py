"""경기 후 자동 채점: output/history의 '경기 직전 픽'을 nflverse 실제 기록과 비교 → output/record.json

결과: hit(적중) / miss(실패) / void(출전 안 함 → FanDuel처럼 무효) / pending(경기 전·기록 대기)
  python grade.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nflbest import data

HIST = Path("output/history")
OUT = Path("output/record.json")
STAT = {"td": None, "rush_yds": "rushing_yards", "rec_yds": "receiving_yards",
        "receptions": "receptions", "pass_yds": "passing_yards", "pass_tds": "passing_tds"}
BUCKETS = [(0, .4), (.4, .5), (.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, 1.01)]


def judge(p: dict, stats: dict, done: set) -> tuple[str, float | None]:
    if p["game_id"] not in done:
        return "pending", None
    row = stats.get((p["player_id"], p["game_id"]))
    if row is None:
        return "void", None
    if p["market"] == "td":
        v = (row["rushing_tds"] or 0) + (row["receiving_tds"] or 0)
        return ("hit" if v >= 1 else "miss"), v
    v = row[STAT[p["market"]]]
    v = 0 if pd.isna(v) else float(v)
    return ("hit" if v >= p["line"] + 0.5 else "miss"), v


def summarize(picks: list[dict]) -> dict:
    g = [p for p in picks if p["result"] in ("hit", "miss")]
    h = sum(p["result"] == "hit" for p in g)
    return {"graded": len(g), "hits": h, "rate": round(100 * h / len(g), 1) if g else None,
            "avg_prob": round(sum(p["prob"] for p in g) / len(g), 1) if g else None,
            "void": sum(p["result"] == "void" for p in picks),
            "pending": sum(p["result"] == "pending" for p in picks)}


def main():
    files = sorted(HIST.glob("*_w*.json"))
    if not files:
        print("보관된 픽 없음")
        return
    weeks_out, pool = [], {}
    seasons = sorted({int(f.name[:4]) for f in files})
    sched = data.schedules(list(range(min(seasons), max(seasons) + 1)), max_age_hours=1)
    done = set(sched[sched.home_score.notna()].game_id)
    for season in seasons:
        ps = data.player_stats([season], current=season, max_age_hours=3)
        stats = {(r.player_id, r.game_id): r._asdict() for r in ps.itertuples()}
        for f in [f for f in files if f.name.startswith(str(season))]:
            h = json.loads(f.read_text())
            wk = {"season": season, "week": h["week"], "lists": {}, "games": {}}
            for key in ("prob", "td", "edge"):
                lst = []
                for p in h["overall"].get(key, []):
                    res, val = judge(p, stats, done)
                    lst.append({**p, "result": res, "actual": val})
                    pool[(p["player_id"], p["game_id"], p["label"])] = lst[-1]
                wk["lists"][key] = {"summary": summarize(lst), "picks": lst}
            all_game = []
            for gid, g in h["games"].items():
                lst = []
                for p in g["best15_prob"]:
                    res, val = judge(p, stats, done)
                    lst.append({**p, "result": res, "actual": val})
                    pool[(p["player_id"], p["game_id"], p["label"])] = lst[-1]
                all_game += lst
                wk["games"][gid] = {"away": g["away"], "home": g["home"], "kickoff": g["kickoff"],
                                    "summary": summarize(lst), "picks": lst}
            wk["games_summary"] = summarize(all_game)
            weeks_out.append(wk)

    graded = [p for p in pool.values() if p["result"] in ("hit", "miss")]
    calib = []
    for lo, hi in BUCKETS:
        b = [p for p in graded if lo <= p["prob"] / 100 < hi]
        if b:
            calib.append({"label": f"{int(lo*100)}–{min(int(hi*100), 100)}%", "n": len(b),
                          "pred": round(sum(p["prob"] for p in b) / len(b), 1),
                          "actual": round(100 * sum(p["result"] == "hit" for p in b) / len(b), 1)})
    by_market = {}
    for m in sorted({p["market"] for p in graded}):
        by_market[m] = summarize([p for p in graded if p["market"] == m])
    out = {
        "updated_at": datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET"),
        "overall_best15": summarize([p for w in weeks_out for p in w["lists"]["prob"]["picks"]]),
        "td_top15": summarize([p for w in weeks_out for p in w["lists"]["td"]["picks"]]),
        "edge_list": summarize([p for w in weeks_out for p in w["lists"]["edge"]["picks"]]),
        "game_best15": summarize([p for w in weeks_out for g in w["games"].values() for p in g["picks"]]),
        "calibration": calib, "by_market": by_market,
        "weeks": sorted(weeks_out, key=lambda w: (w["season"], w["week"]), reverse=True),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    s = out["overall_best15"]
    print(f"전체 베스트 15: {s['hits']}/{s['graded']} 적중 (대기 {s['pending']}, 무효 {s['void']})")


if __name__ == "__main__":
    main()
