"""nflverse 데이터 로딩 + 로컬 캐시 (parquet).

모든 데이터는 nflreadpy(= nflverse GitHub releases)에서 받아요. API 키가 필요 없어요.
"""
from __future__ import annotations

import time
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "cache"
CACHE.mkdir(exist_ok=True)

SKILL_POS = ["QB", "RB", "WR", "TE"]


def _cached(name: str, loader, max_age_hours: float) -> pd.DataFrame:
    path = CACHE / f"{name}.parquet"
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return pd.read_parquet(path)
    df = loader().to_pandas()
    df.to_parquet(path, index=False)
    return df


def schedules(seasons: list[int], max_age_hours: float = 6) -> pd.DataFrame:
    key = f"schedules_{min(seasons)}_{max(seasons)}"
    return _cached(key, lambda: nfl.load_schedules(seasons), max_age_hours)


def player_stats(seasons: list[int], current: int, max_age_hours: float = 6) -> pd.DataFrame:
    """지난 시즌은 오래 캐시하고, 현재 시즌만 자주 새로 받아요."""
    frames = []
    for s in seasons:
        age = max_age_hours if s == current else 24 * 365
        frames.append(_cached(f"player_stats_{s}", lambda s=s: nfl.load_player_stats([s]), age))
    return pd.concat(frames, ignore_index=True)


def injuries(season: int, max_age_hours: float = 1) -> pd.DataFrame:
    try:
        return _cached(f"injuries_{season}", lambda: nfl.load_injuries([season]), max_age_hours)
    except Exception:  # 시즌 초 등 파일이 없을 때
        return pd.DataFrame()


def current_week(sched: pd.DataFrame, season: int, today: str) -> int:
    """오늘 이후(오늘 포함) 첫 경기가 있는 주차. 이미 끝난 목요일 경기가 있어도 같은 주차로 봐요."""
    s = sched[(sched.season == season) & (sched.game_type == "REG")]
    upcoming = s[s.gameday >= today]
    if upcoming.empty:
        return int(s.week.max())
    return int(upcoming.week.min())
