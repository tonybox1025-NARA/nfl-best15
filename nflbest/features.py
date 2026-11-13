"""선수-경기 단위 피처 테이블 만들기.

핵심 원칙: 모든 피처는 '그 경기 전에 알 수 있었던 정보'만 써요 (shift 후 rolling).
다가오는 경기는 기록이 빈 행을 미리 붙여서 같은 계산으로 피처를 얻어요.
"""
from __future__ import annotations

import numpy as np
import nflreadpy as nfl
import pandas as pd

from . import data

STATS = [
    "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
    "receiving_yards", "receiving_tds", "attempts", "completions",
    "passing_yards", "passing_tds", "td_any", "carry_share", "tgt_share",
    "rz10_carries", "rz10_targets", "rz_share", "fantasy_points_ppr",
]


# ---------------------------------------------------------------- red zone (pbp)
def redzone(season: int, current: int) -> pd.DataFrame:
    """10야드 안쪽 캐리/타깃 수 (TD 예측에 가장 중요한 지표)."""
    path = data.CACHE / f"rz_{season}.parquet"
    import time
    max_age = 6 * 3600 if season == current else 365 * 86400
    if path.exists() and time.time() - path.stat().st_mtime < max_age:
        return pd.read_parquet(path)
    pbp = nfl.load_pbp([season]).select(
        ["game_id", "posteam", "yardline_100", "rush_attempt", "pass_attempt",
         "rusher_player_id", "receiver_player_id", "two_point_attempt"]
    ).to_pandas()
    pbp = pbp[(pbp.yardline_100 <= 10) & (pbp.two_point_attempt != 1)]
    r = (pbp[pbp.rush_attempt == 1].groupby(["game_id", "rusher_player_id"]).size()
         .rename("rz10_carries").reset_index().rename(columns={"rusher_player_id": "player_id"}))
    t = (pbp[pbp.pass_attempt == 1].dropna(subset=["receiver_player_id"])
         .groupby(["game_id", "receiver_player_id"]).size()
         .rename("rz10_targets").reset_index().rename(columns={"receiver_player_id": "player_id"}))
    team = (pbp[(pbp.rush_attempt == 1) | (pbp.pass_attempt == 1)]
            .groupby(["game_id", "posteam"]).size().rename("team_rz10_plays").reset_index()
            .rename(columns={"posteam": "team"}))
    out = r.merge(t, on=["game_id", "player_id"], how="outer").fillna(0)
    out.to_parquet(path, index=False)
    team.to_parquet(data.CACHE / f"rzteam_{season}.parquet", index=False)
    return out


def redzone_team(season: int) -> pd.DataFrame:
    p = data.CACHE / f"rzteam_{season}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["game_id", "team", "team_rz10_plays"])


# ---------------------------------------------------------------- main table
def build(seasons: list[int], current_season: int, target_week: int | None) -> pd.DataFrame:
    sched = data.schedules(seasons)
    ps = data.player_stats(seasons, current_season)
    ps = ps[ps.season_type.isin(["REG", "POST"])]

    # 팀 경기별 합계 (점유율 계산용) — 모든 포지션 기준
    team_tot = ps.groupby(["game_id", "team"]).agg(
        team_carries=("carries", "sum"), team_targets=("targets", "sum"),
        team_pass_att=("attempts", "sum")).reset_index()

    ps = ps[ps.position.isin(data.SKILL_POS)].copy()
    rz = pd.concat([redzone(s, current_season) for s in seasons], ignore_index=True)
    rzt = pd.concat([redzone_team(s) for s in seasons], ignore_index=True)
    ps = ps.merge(rz, on=["game_id", "player_id"], how="left")
    ps[["rz10_carries", "rz10_targets"]] = ps[["rz10_carries", "rz10_targets"]].fillna(0)
    ps = ps.merge(team_tot, on=["game_id", "team"], how="left").merge(rzt, on=["game_id", "team"], how="left")

    # ---- 다가오는 경기 행 추가
    if target_week is not None:
        ps = pd.concat([ps, _upcoming_rows(ps, sched, current_season, target_week)], ignore_index=True)
    ps["upcoming"] = ps["upcoming"].fillna(False).astype(bool) if "upcoming" in ps else False

    # ---- 경기 정보
    g = sched[["game_id", "gameday", "gametime", "home_team", "away_team", "spread_line",
               "total_line", "roof", "wind", "temp", "stadium"]]
    ps = ps.drop(columns=[c for c in ["gameday"] if c in ps.columns]).merge(g, on="game_id", how="left")
    ps["is_home"] = (ps.team == ps.home_team).astype(int)
    ps["team_spread"] = np.where(ps.is_home == 1, ps.spread_line, -ps.spread_line)  # +면 우리 팀이 유리
    ps["implied_total"] = ps.total_line / 2 + ps.team_spread / 2
    ps["is_dome"] = ps.roof.isin(["dome", "closed"]).astype(int)
    ps["wind"] = np.where(ps.is_dome == 1, 0, ps.wind)

    # ---- 경기 단위 파생 지표
    ps["td_any"] = ps.rushing_tds + ps.receiving_tds
    ps["carry_share"] = ps.carries / ps.team_carries.replace(0, np.nan)
    ps["tgt_share"] = ps.targets / ps.team_targets.replace(0, np.nan)
    ps["rz_share"] = (ps.rz10_carries + ps.rz10_targets) / ps.team_rz10_plays.replace(0, np.nan)

    ps = ps.sort_values(["player_id", "gameday", "game_id"]).reset_index(drop=True)
    grp = ps.groupby("player_id", sort=False)
    for c in STATS:
        prev = grp[c].shift(1)
        ps[f"{c}_l3"] = prev.groupby(ps.player_id).transform(lambda x: x.rolling(3, min_periods=1).mean())
        ps[f"{c}_l8"] = prev.groupby(ps.player_id).transform(lambda x: x.rolling(8, min_periods=1).mean())
        ps[f"{c}_ewm"] = prev.groupby(ps.player_id).transform(lambda x: x.ewm(halflife=3, ignore_na=True).mean())
    ps["games_prior"] = grp.cumcount()
    ps["td_rate_l16"] = grp["td_any"].shift(1).groupby(ps.player_id).transform(
        lambda x: (x > 0).rolling(16, min_periods=1).mean())

    ps = ps.merge(_defense(ps, sched), on=["game_id", "opponent_team", "position"], how="left")
    ps = ps.merge(_team_form(ps, sched), on=["game_id", "team"], how="left")
    return ps


def _upcoming_rows(ps: pd.DataFrame, sched: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    games = sched[(sched.season == season) & (sched.week == week) & sched.home_score.isna()]
    played = ps.merge(sched[["game_id", "gameday"]], on="game_id")
    rows = []
    # 각 선수의 '가장 최근 소속팀'이 이 팀일 때만 (트레이드 반영)
    latest_team = played.sort_values("gameday").groupby("player_id").team.last()
    for _, gm in games.iterrows():
        for team, opp in [(gm.home_team, gm.away_team), (gm.away_team, gm.home_team)]:
            tg = played[(played.team == team)]
            last3 = tg.drop_duplicates("game_id").sort_values("gameday").game_id.tail(3)
            recent = tg[tg.game_id.isin(last3)]
            for pid, r in recent.groupby("player_id").tail(1).set_index("player_id").iterrows():
                if latest_team.get(pid) != team:
                    continue
                rows.append({"player_id": pid, "player_display_name": r.player_display_name,
                             "position": r.position, "headshot_url": r.headshot_url,
                             "season": season, "week": week, "season_type": "REG",
                             "game_id": gm.game_id, "team": team, "opponent_team": opp,
                             "upcoming": True, "in_last_game": r.game_id == last3.iloc[-1]})
    return pd.DataFrame(rows)


def _defense(ps: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """상대 수비가 포지션별로 최근 8경기에 허용한 평균 (경기 전 기준)."""
    played = ps[~ps.upcoming]
    allowed = played.groupby(["game_id", "opponent_team", "position"]).agg(
        a_rush=("rushing_yards", "sum"), a_rec=("receiving_yards", "sum"),
        a_recs=("receptions", "sum"), a_pass=("passing_yards", "sum"),
        a_ptd=("passing_tds", "sum"), a_td=("td_any", "sum")).reset_index()
    # 모든 경기 × 수비팀 × 포지션 뼈대 (다가오는 경기 포함)
    base = pd.concat([
        sched[["game_id", "gameday", "home_team"]].rename(columns={"home_team": "opponent_team"}),
        sched[["game_id", "gameday", "away_team"]].rename(columns={"away_team": "opponent_team"}),
    ])
    base = base.merge(pd.DataFrame({"position": data.SKILL_POS}), how="cross")
    d = base.merge(allowed, on=["game_id", "opponent_team", "position"], how="left")
    # 아직 안 한 경기는 NaN, 이미 한 경기에서 기록이 없으면 0
    done = sched.set_index("game_id").home_score.notna()
    d["done"] = d.game_id.map(done)
    cols = ["a_rush", "a_rec", "a_recs", "a_pass", "a_ptd", "a_td"]
    d.loc[d.done == True, cols] = d.loc[d.done == True, cols].fillna(0)  # noqa: E712
    d = d.sort_values(["opponent_team", "position", "gameday"])
    g = d.groupby(["opponent_team", "position"])
    for c in cols:
        d[f"def_{c}_l8"] = g[c].transform(lambda x: x.shift(1).rolling(8, min_periods=2).mean())
    return d[["game_id", "opponent_team", "position"] + [f"def_{c}_l8" for c in cols]]


def _team_form(ps: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """팀의 최근 공격 성향: 경기당 패스 시도, 러시 시도."""
    t = ps.drop_duplicates(["game_id", "team"])[["game_id", "team", "gameday", "team_carries", "team_pass_att"]]
    t = t.sort_values(["team", "gameday"])
    g = t.groupby("team")
    t["team_carries_l6"] = g.team_carries.transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    t["team_pass_l6"] = g.team_pass_att.transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    t["team_pass_rate_l6"] = t.team_pass_l6 / (t.team_pass_l6 + t.team_carries_l6)
    return t[["game_id", "team", "team_carries_l6", "team_pass_l6", "team_pass_rate_l6"]]
