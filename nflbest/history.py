"""경기 전 마지막 픽 보관.

실행할 때마다 output/history/{season}_w{week}.json을 갱신하는데,
이미 킥오프한 경기의 픽은 더 이상 바꾸지 않아요 → '경기 직전에 추천했던 픽'이 그대로 남아서 채점에 쓰여요.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HIST = Path(__file__).resolve().parent.parent / "output" / "history"

KEEP = ("rank", "player_id", "name", "pos", "team", "opp", "game_id", "market", "line", "label",
        "prob", "odds", "implied", "edge", "status")


def _kick(s: str) -> datetime:
    return datetime.strptime(s[:16], "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def _slim(p: dict) -> dict:
    return {k: p.get(k) for k in KEEP}


def update(result: dict, now: datetime | None = None) -> Path:
    now = now or datetime.now(ET)
    HIST.mkdir(parents=True, exist_ok=True)
    path = HIST / f"{result['season']}_w{result['week']:02d}.json"
    old = json.loads(path.read_text()) if path.exists() else {"games": {}, "overall": {}}
    kick = {g["game_id"]: g["kickoff"] for g in result["games"]}
    kick.update({gid: g["kickoff"] for gid, g in old["games"].items()})
    started = {gid for gid, k in kick.items() if now >= _kick(k)}

    games = dict(old["games"])
    for g in result["games"]:
        if g["game_id"] in started and g["game_id"] in games:
            continue  # 킥오프 이후: 경기 직전 픽 유지
        games[g["game_id"]] = {
            "away": g["away"], "home": g["home"], "kickoff": g["kickoff"],
            "best15_prob": [_slim(p) for p in g["best15_prob"]],
            "best15_edge": [_slim(p) for p in g.get("best15_edge", [])],
        }

    overall = {}
    for key, sort_by in (("prob", "prob"), ("td", "prob"), ("edge", "edge")):
        kept = [p for p in old["overall"].get(key, []) if p["game_id"] in started]
        fresh = [_slim(p) for p in result["overall"].get(key, []) if p["game_id"] not in started]
        merged = sorted(kept + fresh, key=lambda p: (p.get(sort_by) or -999), reverse=True)[:15]
        for i, p in enumerate(merged, 1):
            p["rank"] = i
        overall[key] = merged

    out = {"season": result["season"], "week": result["week"], "updated_at": result["generated_at"],
           "games": games, "overall": overall}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return path
