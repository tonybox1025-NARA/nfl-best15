"""예측값 → 후보 픽 목록 → 베스트 15."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import markets as M
from .model import Bundle, prob_at_least


def american_to_prob(odds: float) -> float:
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


MODEL_WEIGHT = 0.65   # 배당이 있을 때 최종 확률 = 모델 65% + 시장(수수료 뺀 확률) 35%
GAP_FLAG = 0.15       # 모델과 시장 차이가 15%p 넘으면 '확인 필요' (역할 변화·부상 등 모델이 모르는 정보 가능성)


def _with_market(row: dict) -> dict:
    """모델 확률을 시장 확률과 섞고, 차이가 크면 표시해요."""
    vig = 1.12 if row["market"] == "td" else 1.045   # 한쪽 배당만 있을 때 대략적인 수수료
    novig = min(row["implied"] / vig, 0.97)
    row["model_prob"] = row["prob"]
    row["market_prob"] = novig
    row["prob"] = MODEL_WEIGHT * row["prob"] + (1 - MODEL_WEIGHT) * novig
    row["check"] = abs(row["model_prob"] - novig) > GAP_FLAG
    return row


def candidates(b: Bundle, pred: pd.DataFrame, odds: pd.DataFrame | None = None) -> pd.DataFrame:
    """선수별 후보 픽. odds가 있으면 FanDuel 기준선과 엣지를 함께 계산해요.

    odds 열: player_key, market, line(없으면 TD), price(미국식), side('Over'|'Yes')
    """
    rows = []
    odds_idx = {}
    if odds is not None and len(odds):
        for r in odds.itertuples():
            odds_idx.setdefault((r.player_key, r.market), []).append(r)

    for r in pred.itertuples():
        base = {"player_id": r.player_id, "name": r.player_display_name, "pos": r.position,
                "team": r.team, "opp": r.opponent_team, "game_id": r.game_id}
        key = name_key(r.player_display_name)

        # ---- 애니타임 TD
        if not np.isnan(r.p_td):
            o = odds_idx.get((key, "td"))
            if o:
                price = o[0].price
                rows.append(_with_market({**base, "market": "td", "label": M.TD_LABEL, "line": None,
                                          "prob": r.p_td, "odds": price, "implied": american_to_prob(price)}))
            elif r.p_td >= M.MIN_P_TD:
                rows.append({**base, "market": "td", "label": M.TD_LABEL, "line": None,
                             "prob": _sel(b, "td", r.p_td)})

        # ---- 야드/리셉션/패싱
        for mkey, (mname, label, by_pos) in M.MARKETS.items():
            mu = getattr(r, f"mu_{mname}", np.nan)
            if mu is None or np.isnan(mu) or r.position not in by_pos:
                continue
            o = odds_idx.get((key, mkey))
            if o:  # FanDuel 실제 기준선 (Over 쪽만)
                for oo in o:
                    k = np.floor(oo.line) + 1  # 60.5 이상 = 61 이상
                    p = prob_at_least(b, mname, mu, k)
                    rows.append(_with_market({**base, "market": mkey, "label": label.format(k=f"{oo.line}"),
                                              "line": oo.line, "prob": p, "mu": mu, "odds": oo.price,
                                              "implied": american_to_prob(oo.price)}))
                continue
            best = None
            for k in by_pos[r.position]:
                p = prob_at_least(b, mname, mu, k)
                if p >= M.MIN_P:
                    best = (k, p)
            if best:
                rows.append({**base, "market": mkey, "label": label.format(k=best[0]),
                             "line": best[0] - 0.5, "prob": _sel(b, mname, best[1]), "mu": mu})
    c = pd.DataFrame(rows)
    if len(c) and "implied" in c:
        c["edge"] = c.prob - c.implied
    return c


def _sel(b: Bundle, name: str, p: float) -> float:
    sc = getattr(b, "sel_calib", {}).get(name)
    return float(sc.predict([p])[0]) if sc is not None else p


def best15(c: pd.DataFrame, by: str = "prob", n: int = 15, per_player: int = 2,
           per_market: int | None = None) -> pd.DataFrame:
    """확률순(또는 엣지순) 상위 n개.
    한 선수(최대 per_player개)나 한 종류 마켓(최대 per_market개)이 목록을 독차지하지 않게 해요."""
    if c.empty:
        return c
    per_market = per_market or M.MAX_PER_MARKET
    key = by if by in c and c[by].notna().any() else "prob"
    if key == "edge":   # 엣지순: 배당 있는 픽만, '확인 필요'는 제외
        c = c[c.edge.notna() & ~c.get("check", pd.Series(False, index=c.index)).fillna(False).astype(bool)]
    c = c.sort_values([key, "prob"], ascending=False)
    c = c.groupby("player_id", sort=False).head(per_player)
    if c.market.nunique() > 1:
        c = c.groupby("market", sort=False).head(per_market)
    out = c.head(n).copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def name_key(name: str) -> str:
    import re
    import unicodedata
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", n)
    return re.sub(r"[^a-z]", "", n)
