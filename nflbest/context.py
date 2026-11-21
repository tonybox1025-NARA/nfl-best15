"""팀 맥락: 현재 선발 QB 기준 데이터, 팀 공격·수비 스타일, 픽별 신뢰도.

1) 현재 QB 기준 데이터
   팀의 최근 8경기마다 선발 QB(패스 시도 최다)를 찾고, 이번 주 선발 QB와 다른 QB가 던진 경기가 섞여 있으면
   리시버들의 타깃 점유율과 팀 패스량을 '현재 QB와 뛴 경기' 쪽으로 다시 계산해요.
   (예: BAL은 Huntley → Fagnano → Brissett, 지금은 Brissett 기준)

2) 팀 스타일: 최근 6경기 패스 비율·경기당 플레이, 수비가 내준 캐리당/타깃당 야드 순위

3) 신뢰도: 데이터가 불확실한 상황(역할 변화, QB 교체 직후, 공백 후 복귀, 이적, 출전 불확실)을 표시
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import role


def game_starters(feat: pd.DataFrame) -> pd.DataFrame:
    """경기·팀별 선발 QB (패스 시도 최다)."""
    q = feat[~feat.upcoming & (feat.position == "QB")][["game_id", "team", "gameday", "player_id",
                                                           "player_display_name", "attempts"]]
    q = q.sort_values("attempts", ascending=False).drop_duplicates(["game_id", "team"])
    return q.rename(columns={"player_id": "qb_id", "player_display_name": "qb_name"})


def apply_qb_context(pred: pd.DataFrame, feat: pd.DataFrame) -> dict:
    """QB가 바뀐 팀의 리시버 점유율·팀 패스량을 현재 QB 기준으로 다시 계산 (pred 제자리 수정).
    반환: {team: {qb, qb_games(현재 QB 선발 경기 수, 최근 8경기 중), changed(bool), prev_qbs}}"""
    st = game_starters(feat)
    hist = feat[~feat.upcoming]
    info = {}
    from .live import is_out
    cur_qb = pred[(pred.position == "QB") & ~pred.status.map(is_out)].set_index("team")
    for team in pred.team.unique():
        tg = st[st.team == team].sort_values("gameday").tail(8)
        if tg.empty or team not in cur_qb.index:
            continue
        q = cur_qb.loc[team]
        q = q.iloc[0] if isinstance(q, pd.DataFrame) else q
        cur_games = tg[tg.qb_id == q.player_id].game_id.tolist()
        last6 = tg.tail(6)
        changed = (last6.qb_id != q.player_id).sum() >= 2
        info[team] = {"qb": q.player_display_name, "qb_games": len(cur_games), "changed": bool(changed),
                      "prev_qbs": sorted(set(tg[tg.qb_id != q.player_id].qb_name))}
        if not changed or not cur_games:
            continue
        # 현재 QB 경기의 팀 패스량
        th = hist[(hist.team == team) & hist.game_id.isin(cur_games)]
        team_pass_cur = th[th.position == "QB"].groupby("game_id").attempts.sum().mean()
        n = len(cur_games)
        w = min(n, 3) / 3 * 0.8          # 현재 QB 경기가 많을수록 더 믿어요 (최대 80%)
        idx = pred.index[(pred.team == team) & pred.position.isin(["RB", "WR", "TE"])]
        for i in idx:
            pid = pred.at[i, "player_id"]
            ph = th[th.player_id == pid]
            share_cur = (ph.targets / ph.team_targets.replace(0, np.nan)).fillna(0)
            # 현재 QB 경기에 기록이 없으면(그 경기에 안 뛰었으면) 점유율 0으로 보지 않고 기존 값 유지
            if share_cur.empty:
                continue
            old = pred.at[i, "tgt_share"]
            pred.at[i, "tgt_share"] = w * share_cur.mean() + (1 - w) * old
        if not np.isnan(team_pass_cur):
            pred.loc[idx, "team_pass_exp"] = (w * team_pass_cur + (1 - w) * pred.loc[idx, "team_pass_l6"].fillna(team_pass_cur)) \
                * (1 - 0.010 * pred.loc[idx, "team_spread"].fillna(0).clip(-14, 14))
        role.compute(pred, idx)
        for i in idx:
            for mu, r in (("mu_rec_yds", "mu_rec_role"), ("mu_receptions", "mu_recs_role")):
                if not pd.isna(pred.at[i, mu]) and not pd.isna(pred.at[i, r]):
                    pred.at[i, mu] = 0.5 * pred.at[i, mu] + 0.5 * pred.at[i, r]
        pred.loc[idx, "qb_ctx"] = True
    return info


def team_styles(feat: pd.DataFrame, pred: pd.DataFrame, qbinfo: dict) -> dict:
    """팀별 공격 스타일(최근 6경기)과 수비 허용(최근 8경기) + 이번 주 팀들 사이 순위."""
    hist = feat[~feat.upcoming]
    teams = sorted(pred.team.unique())
    off, dfn = {}, {}
    for t in teams:
        g = hist[hist.team == t].drop_duplicates("game_id").sort_values("gameday").tail(6).game_id
        h = hist[(hist.team == t) & hist.game_id.isin(g)]
        pa = h[h.position == "QB"].groupby("game_id").attempts.sum()
        ca = h.groupby("game_id").carries.sum()
        n = max(len(g), 1)
        off[t] = {"pass_att": round(pa.sum() / n, 1), "rush_att": round(ca.sum() / n, 1),
                  "pass_rate": round(pa.sum() / max(pa.sum() + ca.sum(), 1), 3)}
        g8 = hist[hist.opponent_team == t].drop_duplicates("game_id").sort_values("gameday").tail(8).game_id
        d = hist[(hist.opponent_team == t) & hist.game_id.isin(g8)]
        rb = d[d.position == "RB"]
        dfn[t] = {"ypc": round(rb.rushing_yards.sum() / max(rb.carries.sum(), 1), 2),
                  "ypt": round(d.receiving_yards.sum() / max(d.targets.sum(), 1), 2)}
    out = {}
    pr = pd.Series({t: off[t]["pass_rate"] for t in teams}).rank(ascending=False, method="min")
    ry = pd.Series({t: dfn[t]["ypc"] for t in teams}).rank(ascending=False, method="min")
    py = pd.Series({t: dfn[t]["ypt"] for t in teams}).rank(ascending=False, method="min")
    for t in teams:
        out[t] = {**off[t], "pass_rate_rank": int(pr[t]), "def_ypc": dfn[t]["ypc"], "def_ypc_rank": int(ry[t]),
                  "def_ypt": dfn[t]["ypt"], "def_ypt_rank": int(py[t]), "n": len(teams), **qbinfo.get(t, {})}
    return out


def confidence(c, u, qbinfo: dict, week: int, last_wk: int | None) -> tuple[str, list[str]]:
    """픽 신뢰도와 이유 (이유는 코드로 저장 → 앱에서 한국어/영어로 보여줘요)."""
    why = []
    if bool(u.get("role_changed")) if not pd.isna(u.get("role_changed", np.nan)) else False:
        why.append("role")              # 동료 결장으로 역할이 바뀜
    qi = qbinfo.get(c.team, {})
    if qi.get("changed") and qi.get("qb_games", 9) <= 1 and c.pos != "QB":
        why.append("qb_new")            # QB 교체 직후 (현재 QB 선발 1경기 이하)
    if c.pos == "QB" and qi.get("qb_games", 9) <= 1:
        why.append("qb_few")            # 선발 경험이 적은 QB
    if last_wk is not None and last_wk < week - 2:
        why.append("layoff")            # 공백 후 복귀
    if bool(u.get("new_to_team")) if not pd.isna(u.get("new_to_team", np.nan)) else False:
        why.append("new_team")          # 이적 후 첫 경기들
    st = str(getattr(c, "status", "") or "").lower()
    if st in ("questionable", "doubtful", "훈련 불참"):
        why.append("status")            # 본인 출전 불확실
    if bool(getattr(c, "check", False)) if not pd.isna(getattr(c, "check", np.nan)) else False:
        why.append("market")            # 시장과 큰 차이
    low = {"role", "qb_new", "qb_few", "layoff", "new_team", "market"}
    level = "low" if any(w in low for w in why) else ("mid" if why or (qi.get("changed") and c.pos != "QB") else "high")
    if level == "mid" and qi.get("changed") and "qb_mid" not in why and c.pos != "QB":
        why.append("qb_mid")            # QB가 바뀐 지 2~3경기
    return level, why
