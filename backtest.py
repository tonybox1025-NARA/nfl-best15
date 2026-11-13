"""백테스트: 2020~2024로 학습 → 2025 시즌 전체를 '경기 전 시점'으로 예측해서 실제 결과와 비교.

python backtest.py  → reports/backtest_2025.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nflbest import features, model, picks
from nflbest import markets as M

import os
TEST = int(os.environ.get("TEST", 2025))
TRAIN = list(range(2020, TEST))
OUT = Path("reports"); OUT.mkdir(exist_ok=True)

ACTUAL = {"td": "td_any", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards",
          "receptions": "receptions", "pass_yds": "passing_yards", "pass_tds": "passing_tds"}


def hit(row, actual_df):
    a = actual_df.loc[(row.player_id, row.game_id)]
    v = a[ACTUAL[row.market]]
    return float(v >= 1) if row.market == "td" else float(v >= row.line + 0.5)


def main():
    cache = Path("cache/features_hist.parquet")
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = features.build(list(range(2020, 2026)), 2025, None)
        df.to_parquet(cache)
    hist = df[df.season.isin(TRAIN)]
    test = df[(df.season == TEST) & (df.season_type == "REG")]

    b = model.fit(hist, TRAIN)
    pred = model.predict(b, test)
    lines = [f"# 백테스트: {TRAIN[0]}–{TRAIN[-1]} 학습 → {TEST} 정규시즌 예측\n"]

    # ---- 1) TD 확률 품질
    d = pred.dropna(subset=["p_td"])
    y = (d.td_any > 0).astype(float)
    base = d.td_rate_l16.fillna(y.mean()).clip(0.01, 0.99)
    brier_m = np.mean((d.p_td - y) ** 2); brier_b = np.mean((base - y) ** 2)
    lines.append("## 1. 애니타임 TD 확률\n")
    lines.append(f"- 대상 {len(d):,}명-경기, 실제 TD 비율 {y.mean():.1%}")
    lines.append(f"- Brier 점수 (낮을수록 좋음): 모델 **{brier_m:.4f}** vs 단순 '최근 TD 비율' {brier_b:.4f}"
                 f" → {100*(1-brier_m/brier_b):.1f}% 개선\n")
    lines.append("| 예측 구간 | 픽 수 | 평균 예측 | 실제 적중 |\n|---|---|---|---|")
    for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1)]:
        m = (d.p_td >= lo) & (d.p_td < hi)
        if m.sum():
            lines.append(f"| {lo:.0%}–{hi:.0%} | {m.sum()} | {d.p_td[m].mean():.1%} | {y[m].mean():.1%} |")

    # ---- 2) 야드/리셉션 마일스톤 보정
    rows = []
    for r in pred.itertuples():
        for mkey, (mname, _, by_pos) in M.MARKETS.items():
            mu = getattr(r, f"mu_{mname}")
            if np.isnan(mu) or r.position not in by_pos:
                continue
            for k in by_pos[r.position]:
                rows.append((mkey, model.prob_at_least(b, mname, mu, k),
                             float(getattr(r, ACTUAL[mkey]) >= k)))
    mk = pd.DataFrame(rows, columns=["market", "p", "y"])
    lines.append("\n## 2. 야드·리셉션·패싱 '○ 이상' 확률 보정\n")
    lines.append("예측 확률과 실제 적중률이 비슷할수록 믿을 수 있는 확률이에요.\n")
    lines.append("| 예측 구간 | 예측 수 | 평균 예측 | 실제 적중 |\n|---|---|---|---|")
    for lo, hi in [(.4, .5), (.5, .55), (.55, .6), (.6, .65), (.65, .7), (.7, .8), (.8, .9)]:
        m = (mk.p >= lo) & (mk.p < hi)
        if m.sum():
            lines.append(f"| {lo:.0%}–{hi:.0%} | {m.sum():,} | {mk.p[m].mean():.1%} | {mk.y[m].mean():.1%} |")
    lines.append("\n| 마켓 | Brier 모델 | Brier 기준(전체 평균) |\n|---|---|---|")
    for mkey, g in mk.groupby("market"):
        lines.append(f"| {mkey} | {np.mean((g.p-g.y)**2):.4f} | {np.mean((g.y.mean()-g.y)**2):.4f} |")

    # ---- 3) 주차별 베스트 15 (배당 없이 확률순)
    actual = test.set_index(["player_id", "game_id"])
    lines.append("\n## 3. 매주 '전체 베스트 15' 실제 성적 (확률순, 선수당 최대 2개)\n")
    lines.append("| 주차 | 평균 예측 확률 | 적중 | 적중률 | TOP 5 적중 |\n|---|---|---|---|---|")
    tot_h = tot_n = tot_p = top5 = 0; game_h = game_n = 0
    for wk, wpred in pred.groupby("week"):
        c = picks.candidates(b, wpred)
        top = picks.best15(c)
        h = top.apply(lambda r: hit(r, actual), axis=1)
        tot_h += h.sum(); tot_n += len(h); tot_p += top.prob.sum(); top5 += h.head(5).sum()
        lines.append(f"| {wk} | {top.prob.mean():.1%} | {int(h.sum())}/{len(h)} | {h.mean():.1%} | {int(h.head(5).sum())}/5 |")
        for _, gc in c.groupby("game_id"):
            gt = picks.best15(gc)
            gh = gt.apply(lambda r: hit(r, actual), axis=1)
            game_h += gh.sum(); game_n += len(gh)
    lines.append(f"| **합계** | **{tot_p/tot_n:.1%}** | **{int(tot_h)}/{tot_n}** | **{tot_h/tot_n:.1%}** | **{int(top5)}/{5*pred.week.nunique()}** |")
    lines.append(f"\n경기별 베스트 15 전체: {int(game_h)}/{game_n} 적중 = **{game_h/game_n:.1%}**")

    # ---- 4) TD 전용 순위
    lines.append("\n## 4. 매주 'TD 확률 상위 15명' 실제 성적\n")
    th = tn = tp = 0
    for wk, wpred in pred.groupby("week"):
        t = wpred.dropna(subset=["p_td"]).nlargest(15, "p_td")
        th += (t.td_any > 0).sum(); tn += len(t); tp += t.p_td.sum()
    lines.append(f"- 평균 예측 {tp/tn:.1%} → 실제 {th}/{tn} = **{th/tn:.1%}** "
                 f"(같은 시즌 전체 선수 평균 TD 비율 {y.mean():.1%})")

    imp = pd.Series(b.models["td"].feature_importances_, index=b.feats).nlargest(10)
    lines.append("\n## 5. TD 모델이 가장 많이 본 지표 TOP 10\n")
    lines += [f"- {k}: {v}" for k, v in imp.items()]

    (OUT / f"backtest_{TEST}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
