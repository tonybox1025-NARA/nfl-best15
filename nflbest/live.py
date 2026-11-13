"""실시간 데이터: 부상(nflverse + ESPN), 날씨(Open-Meteo), 배당(The Odds API).

각 함수는 실패해도 앱 전체가 멈추지 않게 (데이터, 상태메시지)를 돌려줘요.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from . import data
from .picks import name_key

ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "nfl-best15/0.1"}
OVERRIDES = Path(__file__).resolve().parent.parent / "overrides.json"

OUT_STATUSES = {"out", "doubtful", "injured reserve", "ir", "inactive", "suspended", "pup",
                "physically unable to perform", "reserve/covid-19", "non football injury"}


def now_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")


# =================================================================== 부상
def injuries(season: int, week: int) -> tuple[dict, str]:
    """{name_key: {'status': 'Out'|'Doubtful'|'Questionable'|..., 'injury': str, 'source': str}}"""
    out: dict = {}
    msgs = []
    inj = data.injuries(season)
    if len(inj):
        w = inj[(inj.week == week) & (inj.season_type == "REG")]
        for r in w.itertuples():
            injury = r.report_primary_injury or r.practice_primary_injury or ""
            if isinstance(r.report_status, str) and r.report_status:
                st = r.report_status
            elif isinstance(r.practice_status, str) and "did not" in r.practice_status.lower():
                st = "훈련 불참"   # 최종 리포트 전 (결장 처리하지 않고 표시만)
            elif isinstance(r.practice_status, str) and "limited" in r.practice_status.lower():
                st = "제한 훈련"
            else:
                continue
            out[name_key(r.full_name)] = {"status": st, "team": r.team, "injury": injury, "source": "NFL 리포트"}
        msgs.append(f"NFL 공식 리포트 {len(w)}건")
    try:
        js = requests.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries",
                          headers=UA, timeout=15).json()
        n = 0
        for team in js.get("injuries", []):
            for it in team.get("injuries", []):
                nm = (it.get("athlete") or {}).get("displayName")
                st = it.get("status") or (it.get("type") or {}).get("description")
                if nm and st:
                    out[name_key(nm)] = {"status": st, "team": team.get("displayName", ""),
                                         "injury": (it.get("details") or {}).get("type", ""), "source": "ESPN"}
                    n += 1
        msgs.append(f"ESPN 실시간 {n}건")
    except Exception as e:  # noqa: BLE001
        msgs.append(f"ESPN 연결 실패({type(e).__name__})")
    # 수동 입력 (경기 직전 비출전 명단을 직접 넣을 때)
    if OVERRIDES.exists():
        ov = json.loads(OVERRIDES.read_text())
        for nm in ov.get("out", []):
            out[name_key(nm)] = {"status": "Out", "injury": "", "source": "수동"}
        for nm in ov.get("active", []):
            out.pop(name_key(nm), None)
        msgs.append(f"수동 입력 {len(ov.get('out', []))}건")
    return out, " · ".join(msgs)


def is_out(status: str) -> bool:
    return str(status).lower() in OUT_STATUSES


# =================================================================== 날씨
# 홈 경기장 좌표 (lat, lon). 돔은 날씨 영향이 없어서 계산에서 빠져요.
STADIUMS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4008), "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870), "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995), "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839), "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389), "MIN": (44.9737, -93.2577),
    "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9077, -76.8645),
}


def weather(games: pd.DataFrame) -> tuple[dict, str]:
    """{game_id: {'temp_f', 'wind_mph', 'precip_pct'}} — 킥오프 시각 예보.
    야외 경기장 전부를 한 번의 요청으로 받아요 (Open-Meteo 다중 좌표), 실패하면 2번 더 시도."""
    out = {}
    outdoor = []
    for g in games.itertuples():
        if g.roof in ("dome", "closed"):
            out[g.game_id] = {"dome": True}
        elif g.home_team in STADIUMS:
            outdoor.append(g)
    if not outdoor:
        return out, "날씨: 야외 경기 없음"
    days = sorted({g.gameday for g in outdoor})
    params = {
        "latitude": ",".join(str(STADIUMS[g.home_team][0]) for g in outdoor),
        "longitude": ",".join(str(STADIUMS[g.home_team][1]) for g in outdoor),
        "timezone": "America/New_York", "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "start_date": days[0], "end_date": days[-1]}
    err = ""
    for attempt in range(3):
        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast", headers=UA, timeout=40, params=params)
            js = resp.json()
            js = js if isinstance(js, list) else [js]
            if "hourly" not in js[0]:
                raise ValueError(f"HTTP {resp.status_code} {js[0].get('reason', '')}")
            for g, loc in zip(outdoor, js):
                h = loc["hourly"]
                i = h["time"].index(f"{g.gameday}T{g.gametime[:2]}:00")
                out[g.game_id] = {"temp_f": round(h["temperature_2m"][i]), "wind_mph": round(h["wind_speed_10m"][i]),
                                  "precip_pct": h["precipitation_probability"][i]}
            return out, f"날씨 {len(outdoor)}경기 (야외)"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:100]}"
            time.sleep(3 * (attempt + 1))
    return out, f"날씨 실패 ({err})"


# =================================================================== 배당
TEAM_FULL = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Los Angeles Rams": "LA", "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA", "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
# 무료 플랜(월 500 크레딧) 절약: 1경기 1회 조회 = 마켓 수만큼 크레딧. 기본은 3개 마켓.
DEFAULT_MARKETS = os.environ.get("ODDS_MARKETS", "player_anytime_td,player_rush_yds,player_reception_yds")
# 무료 크레딧 보호: 배당은 기본 12시간에 한 번만 새로 받아요 (--force-odds로 강제 가능)
ODDS_CACHE_MIN = int(os.environ.get("ODDS_CACHE_MIN", "720"))


def odds(games: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame | None, str]:
    """FanDuel 선수 프롭 (The Odds API). 키가 없으면 None."""
    from .markets import ODDS_MARKETS
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return None, "배당 미사용 (ODDS_API_KEY 없음)"
    cache = data.CACHE / "odds.parquet"
    if not force and cache.exists() and time.time() - cache.stat().st_mtime < ODDS_CACHE_MIN * 60:
        age = int((time.time() - cache.stat().st_mtime) / 60)
        return pd.read_parquet(cache), f"배당 {age}분 전 데이터 사용 (크레딧 절약)"
    base = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
    try:
        r = requests.get(f"{base}/events", params={"apiKey": key}, timeout=20)
        events = r.json()
        want = {(g.away_team, g.home_team): g.game_id for g in games.itertuples()}
        rows, left = [], r.headers.get("x-requests-remaining")
        for ev in events:
            gid = want.get((TEAM_FULL.get(ev["away_team"]), TEAM_FULL.get(ev["home_team"])))
            if not gid:
                continue
            rr = requests.get(f"{base}/events/{ev['id']}/odds", timeout=20, params={
                "apiKey": key, "regions": "us", "bookmakers": "fanduel",
                "markets": DEFAULT_MARKETS, "oddsFormat": "american"})
            left = rr.headers.get("x-requests-remaining", left)
            for bk in rr.json().get("bookmakers", []):
                for mk in bk.get("markets", []):
                    ours = ODDS_MARKETS.get(mk["key"])
                    for o in mk.get("outcomes", []):
                        if o.get("name") not in ("Over", "Yes") or not ours:
                            continue
                        rows.append({"game_id": gid, "player_key": name_key(o.get("description", "")),
                                     "market": ours, "line": o.get("point"), "price": o["price"]})
        df = pd.DataFrame(rows)
        df.to_parquet(cache, index=False)
        return df, f"FanDuel 배당 {len(df)}개 (남은 크레딧 {left})"
    except Exception as e:  # noqa: BLE001
        if cache.exists():
            return pd.read_parquet(cache), f"배당 갱신 실패({type(e).__name__}) → 이전 배당 사용"
        return None, f"배당 연결 실패({type(e).__name__})"
