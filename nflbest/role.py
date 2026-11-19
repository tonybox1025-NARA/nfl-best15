"""역할 기반 예상: 이번 경기 예상 사용량 × 선수 효율 × 상대 수비 보정.

과거 기록만 보는 모델은 '역할이 바뀐 선수'(주전 결장으로 올라온 백업, 부상 복귀 선수)를 잘 모르기 때문에,
이 계산으로 이번 경기에 맞는 예상값을 따로 만들어서 섞어요.

  예상 러싱 야드  = 팀 예상 러싱 시도 × 캐리 점유율 × 캐리당 야드(리그 평균으로 보정) × 상대 런 수비 보정
  예상 리시빙 야드 = 팀 예상 패스 시도 × 타깃 점유율 × 타깃당 야드 × 상대 패스 수비(포지션별) 보정
  예상 리셉션     = 예상 타깃 × 캐치율
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR_CAR, PRIOR_TGT = 40, 30          # 표본이 적은 선수는 리그 평균 쪽으로 당겨요
DEF_PRIOR_CAR, DEF_PRIOR_TGT = 60, 40


def tables(feat: pd.DataFrame, season: int) -> dict:
    hist = feat[~feat.upcoming & feat.season.isin([season - 1, season])].sort_values("gameday")
    rb = hist[hist.position == "RB"]
    lg = {"ypc": rb.rushing_yards.sum() / max(rb.carries.sum(), 1)}
    for pos in ("RB", "WR", "TE"):
        h = hist[hist.position == pos]
        lg[f"ypt_{pos}"] = h.receiving_yards.sum() / max(h.targets.sum(), 1)
        lg[f"catch_{pos}"] = h.receptions.sum() / max(h.targets.sum(), 1)

    # 선수 효율 (최근 16경기)
    allh = feat[~feat.upcoming].sort_values("gameday").groupby("player_id").tail(16)
    g = allh.groupby("player_id").agg(car=("carries", "sum"), ry=("rushing_yards", "sum"),
                                      tgt=("targets", "sum"), rec=("receptions", "sum"),
                                      recy=("receiving_yards", "sum"), pos=("position", "last"))
    g["ypc"] = (g.ry + lg["ypc"] * PRIOR_CAR) / (g.car + PRIOR_CAR)
    g["ypt"] = [(r.recy + lg.get(f"ypt_{r.pos}", 7.5) * PRIOR_TGT) / (r.tgt + PRIOR_TGT) for r in g.itertuples()]
    g["catch"] = [(r.rec + lg.get(f"catch_{r.pos}", 0.65) * PRIOR_TGT) / (r.tgt + PRIOR_TGT) for r in g.itertuples()]

    # 상대 수비 보정 (최근 8경기, 리그 평균 대비)
    games = hist.drop_duplicates(["game_id", "opponent_team"])[["game_id", "opponent_team", "gameday"]]
    last8 = games.sort_values("gameday").groupby("opponent_team").tail(8)
    h8 = hist.merge(last8[["game_id", "opponent_team"]], on=["game_id", "opponent_team"])
    d = {}
    r = h8[h8.position == "RB"].groupby("opponent_team").agg(car=("carries", "sum"), y=("rushing_yards", "sum"))
    for team, x in r.iterrows():
        d[(team, "rush")] = ((x.y + lg["ypc"] * DEF_PRIOR_CAR) / (x.car + DEF_PRIOR_CAR)) / lg["ypc"]
    for pos in ("RB", "WR", "TE"):
        r = h8[h8.position == pos].groupby("opponent_team").agg(t=("targets", "sum"), y=("receiving_yards", "sum"))
        for team, x in r.iterrows():
            base = lg[f"ypt_{pos}"]
            d[(team, f"rec_{pos}")] = ((x.y + base * DEF_PRIOR_TGT) / (x.t + DEF_PRIOR_TGT)) / base
    return {"lg": lg, "eff": g[["ypc", "ypt", "catch"]], "def": d}


def attach(pred: pd.DataFrame, T: dict) -> None:
    """예측 표에 효율·수비 보정·팀 예상 볼륨을 붙이고 역할 기반 예상값을 계산해요 (제자리 수정)."""
    eff = T["eff"]
    pred["ypc"] = pred.player_id.map(eff.ypc).fillna(T["lg"]["ypc"])
    pred["ypt"] = [eff.ypt.get(p, T["lg"].get(f"ypt_{pos}", 7.5)) for p, pos in zip(pred.player_id, pred.position)]
    pred["catch"] = [eff["catch"].get(p, T["lg"].get(f"catch_{pos}", 0.65)) for p, pos in zip(pred.player_id, pred.position)]
    pred["d_rush"] = [T["def"].get((o, "rush"), 1.0) for o in pred.opponent_team]
    pred["d_rec"] = [T["def"].get((o, f"rec_{pos}"), 1.0) for o, pos in zip(pred.opponent_team, pred.position)]
    sp = pred.team_spread.fillna(0).clip(-14, 14)
    pred["team_car_exp"] = pred.team_carries_l6.fillna(25) * (1 + 0.015 * sp)     # 이기는 팀은 더 달려요
    pred["team_pass_exp"] = pred.team_pass_l6.fillna(34) * (1 - 0.010 * sp)       # 지는 팀은 더 던져요
    pred["car_share"] = pred.carry_share_ewm.fillna(0)
    pred["tgt_share"] = pred.tgt_share_ewm.fillna(0)
    for c in ("mu_rush_yds", "mu_rec_yds", "mu_receptions"):
        pred[c + "_model"] = pred[c]
    compute(pred)


def compute(pred: pd.DataFrame, idx=None) -> None:
    idx = pred.index if idx is None else idx
    p = pred.loc[idx]
    pred.loc[idx, "exp_car"] = p.team_car_exp * p.car_share
    pred.loc[idx, "exp_tgt"] = p.team_pass_exp * 0.95 * p.tgt_share
    pred.loc[idx, "mu_rush_role"] = pred.loc[idx, "exp_car"] * p.ypc * p.d_rush
    pred.loc[idx, "mu_rec_role"] = pred.loc[idx, "exp_tgt"] * p.ypt * p.d_rec
    pred.loc[idx, "mu_recs_role"] = pred.loc[idx, "exp_tgt"] * p["catch"]
