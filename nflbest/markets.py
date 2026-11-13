"""FanDuel에서 흔히 파는 선수 프롭 마켓과 기준선.

배당 데이터가 없을 때는 각 선수·마켓마다 '확률이 MIN_P 이상인 가장 높은 기준선'을 후보로 골라요.
(예: 러싱 50/60/70/80야드 중 확률 55% 넘는 가장 높은 것 → "러싱 70야드 이상 58%")
배당 데이터가 있으면 FanDuel의 실제 기준선(O/U x.5)을 그대로 써요.
"""
from __future__ import annotations

MIN_P = 0.55       # 야드/리셉션 마일스톤 후보 최소 확률
MIN_P_TD = 0.30    # TD 후보 최소 확률 (TD 전용 순위용)
MIN_P_EDGE = 0.45  # 엣지순 목록 최소 확률 (파레이에 쓰기 어려운 롱샷 제외)
MAX_PER_MARKET = 4  # 베스트 15 안에서 같은 종류 마켓 최대 개수 (다양성)

# key: (모델 이름, 라벨, {포지션: 기준선 목록})
MARKETS = {
    "rush_yds":   ("rush_yds", "러싱 {k}야드 이상",
                   {"RB": [40, 50, 60, 70, 80, 90, 100], "QB": [20, 30, 40, 50]}),
    "rec_yds":    ("rec_yds", "리시빙 {k}야드 이상",
                   {"RB": [20, 30, 40, 50], "WR": [40, 50, 60, 70, 80, 90, 100],
                    "TE": [30, 40, 50, 60, 70, 80]}),
    "receptions": ("receptions", "리셉션 {k}개 이상",
                   {"RB": [3, 4, 5], "WR": [4, 5, 6, 7, 8], "TE": [3, 4, 5, 6, 7]}),
    "pass_yds":   ("pass_yds", "패싱 {k}야드 이상", {"QB": [200, 225, 250, 275, 300]}),
    "pass_tds":   ("pass_tds", "패싱 TD {k}개 이상", {"QB": [2, 3]}),
}
TD_LABEL = "애니타임 터치다운"

# The Odds API 마켓 이름 → 우리 마켓
ODDS_MARKETS = {
    "player_anytime_td": "td",
    "player_rush_yds": "rush_yds",
    "player_reception_yds": "rec_yds",
    "player_receptions": "receptions",
    "player_pass_yds": "pass_yds",
    "player_pass_tds": "pass_tds",
}
