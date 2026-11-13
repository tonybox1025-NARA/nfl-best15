"""예측 모델.

- 애니타임 TD: LightGBM 분류 → 확률
- 야드/리셉션/패싱TD: LightGBM 회귀로 '예상값(평균)'을 구하고,
  비슷한 예상값을 받았던 과거 선수들의 실제 결과 분포(잔차 비율)로 'N 이상일 확률'을 계산해요.
  → 정규분포 가정 없이 실제 NFL 결과의 들쭉날쭉함(부상 퇴장, 대박 경기)을 그대로 반영해요.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import STATS

VERSION = 2   # 모델 구조가 바뀌면 올려요 → 자동 재학습

# 확률 보정에 쓰는 기준선 범위 (FanDuel이 낮은/높은 기준선을 걸어도 정확하게)
CALIB_GRID = {
    "rush_yds": list(range(5, 131, 5)), "rec_yds": list(range(5, 131, 5)),
    "receptions": list(range(1, 11)), "pass_yds": list(range(150, 376, 25)), "pass_tds": [1, 2, 3, 4],
}

POS_CODE = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}

TARGETS = {
    # name: (kind, positions, target column, row filter on PRE-GAME info)
    "td":         ("clf", ["QB", "RB", "WR", "TE"], "td_any", None),
    "rush_yds":   ("reg", ["RB", "QB"], "rushing_yards", None),
    "rec_yds":    ("reg", ["RB", "WR", "TE"], "receiving_yards", None),
    "receptions": ("reg", ["RB", "WR", "TE"], "receptions", None),
    "pass_yds":   ("reg", ["QB"], "passing_yards", "starter"),
    "pass_tds":   ("reg", ["QB"], "passing_tds", "starter"),
}


def feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [f"{s}_{w}" for s in STATS for w in ("l3", "l8", "ewm")]
    cols += [c for c in df.columns if c.startswith("def_") or c.startswith("team_") and c.endswith("_l6")]
    cols += ["games_prior", "td_rate_l16", "is_home", "team_spread", "implied_total",
             "total_line", "is_dome", "wind", "pos_code"]
    return [c for c in dict.fromkeys(cols) if c in df.columns]


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos_code"] = df.position.map(POS_CODE)
    return df


def rows_for(df: pd.DataFrame, name: str) -> pd.DataFrame:
    kind, pos, ycol, filt = TARGETS[name]
    d = df[df.position.isin(pos) & (df.games_prior >= 1)]
    if filt == "starter":
        d = d[d.attempts_l3 >= 15]
    return d


PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def _fit_one(name: str, d: pd.DataFrame, feats: list[str]):
    kind, _, ycol, _ = TARGETS[name]
    y = d[ycol].astype(float)
    if kind == "clf":
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(d[feats], (y > 0).astype(int))
    else:
        obj = "poisson" if name in ("receptions", "pass_tds") else "regression"
        m = lgb.LGBMRegressor(objective=obj, **PARAMS)
        m.fit(d[feats], y.clip(lower=0) if obj == "poisson" else y)
    return m


def _pred_one(m, name, X):
    return m.predict_proba(X)[:, 1] if TARGETS[name][0] == "clf" else np.clip(m.predict(X), 0.05, None)


class LogitCalib:
    """확률 보정: logit(p_보정) = a + b·logit(p_원래). 단조롭고 부드러워서 같은 값이 겹치지 않아요."""

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        x = _logit(p).reshape(-1, 1)
        self.m = LogisticRegression(C=100).fit(x, y)
        return self

    def predict(self, p):
        return self.m.predict_proba(_logit(np.asarray(p, float)).reshape(-1, 1))[:, 1]


def _logit(p):
    p = np.clip(np.asarray(p, float), 0.002, 0.998)
    return np.log(p / (1 - p))


@dataclass
class Bundle:
    feats: list[str]
    models: dict = field(default_factory=dict)
    # 회귀 모델별: 예측값 구간 경계 + 구간별 (실제/예측) 비율 샘플
    resid: dict = field(default_factory=dict)
    calib: dict = field(default_factory=dict)       # 1단계 확률 보정
    sel_calib: dict = field(default_factory=dict)   # 2단계: 후보로 '선택된' 픽의 확률 보정
    version: int = 0

    def save(self, path: Path):
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "Bundle":
        return pickle.loads(path.read_bytes())


def _oof(df: pd.DataFrame, seasons: list[int], feats: list[str]) -> dict:
    """시즌 단위 교차검증 예측 (각 시즌을 빼고 학습 → 그 시즌 예측)."""
    out = {}
    for name, (kind, _p, ycol, _f) in TARGETS.items():
        d = rows_for(df, name)
        d = d[d.season.isin(seasons) & d[ycol].notna()]
        preds, acts, poss = [], [], []
        for s in seasons:
            tr, te = d[d.season != s], d[d.season == s]
            m = _fit_one(name, tr, feats)
            preds.append(_pred_one(m, name, te[feats]))
            acts.append(te[ycol].to_numpy(float))
            poss.append(te.position.to_numpy())
        out[name] = (np.concatenate(preds), np.concatenate(acts), np.concatenate(poss))
    return out


def _residuals(p: np.ndarray, a: np.ndarray) -> dict:
    """'모델이 X를 예상했을 때 실제로는 어땠나' — 예상값 구간별 (실제/예상) 비율, 정렬해서 저장."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 13)))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return {"edges": edges, "ratios": [np.sort(a[bins == b] / p[bins == b]) for b in range(len(edges) - 1)]}


def fit(df: pd.DataFrame, resid_seasons: list[int]) -> Bundle:
    from . import markets as M
    df = prep(df)
    feats = feature_cols(df)
    train = df[~df.upcoming.astype(bool)]
    b = Bundle(feats=feats, version=VERSION)
    for name in TARGETS:
        d = rows_for(train, name)
        d = d[d[TARGETS[name][2]].notna()]
        b.models[name] = _fit_one(name, d, feats)

    oof = _oof(train, resid_seasons, feats)
    for name, (p, a, pos) in oof.items():
        if TARGETS[name][0] == "clf":
            b.calib[name] = LogitCalib().fit(p, (a > 0).astype(int))
            pc = b.calib[name].predict(p)
            sel = pc >= M.MIN_P_TD   # 2단계: TD 후보로 뽑힌 선수만 다시 보정
            b.sel_calib[name] = LogitCalib().fit(pc[sel], (a[sel] > 0).astype(int))
            continue
        b.resid[name] = _residuals(p, a)
        # 마켓별 확률 보정: OOF 예측으로 '○ 이상' 확률을 계산해 실제 적중과 맞춤
        ks = CALIB_GRID[name]
        raw = np.concatenate([_raw_prob(b.resid[name], name, p, k) for k in ks])
        y = np.concatenate([(a >= k).astype(float) for k in ks])
        b.calib[name] = LogitCalib().fit(raw, y.astype(int))

        # 2단계: 실제 후보 선택 규칙(확률 MIN_P 넘는 가장 높은 기준선)을 그대로 흉내 내서,
        # '선택된 픽'의 확률을 다시 보정 → 선택 편향(운 좋게 높게 나온 숫자) 제거
        by_pos = next(m[2] for m in M.MARKETS.values() if m[0] == name)
        sp, sy = [], []
        for ps_, kl in by_pos.items():
            mask = pos == ps_
            if not mask.any():
                continue
            P = np.column_stack([b.calib[name].predict(_raw_prob(b.resid[name], name, p[mask], k)) for k in kl])
            ok = P >= M.MIN_P
            has = ok.any(axis=1)
            j = np.where(ok, np.arange(len(kl)), -1).max(axis=1)
            ks = np.array(kl)[j[has]]
            sp.append(P[has, j[has]]); sy.append(a[mask][has] >= ks)
        b.sel_calib[name] = LogitCalib().fit(np.concatenate(sp), np.concatenate(sy).astype(int))
    return b


def predict(b: Bundle, df: pd.DataFrame) -> pd.DataFrame:
    """각 선수 행에 예상값(mu_*)과 TD 확률(p_td)을 붙여요. 해당 없는 포지션은 NaN."""
    df = prep(df)
    out = df.copy()
    for name in TARGETS:
        d = rows_for(df, name)
        col = "p_td" if name == "td" else f"mu_{name}"
        out[col] = np.nan
        if len(d):
            v = _pred_one(b.models[name], name, d[b.feats])
            if name in b.calib and TARGETS[name][0] == "clf":
                v = b.calib[name].predict(v)
            out.loc[d.index, col] = v
    return out


def _raw_prob(r: dict, name: str, mu, k: float):
    """벡터화: 예상값 mu(배열 가능)인 선수가 k 이상 기록할 (보정 전) 확률."""
    mu = np.atleast_1d(np.asarray(mu, float))
    idx = np.clip(np.searchsorted(r["edges"], mu, side="right") - 1, 0, len(r["ratios"]) - 1)
    # 정수 기록은 반올림 기준 (k-0.5 이상), 연속값은 k 이상
    thr = (k - 0.5 if name in ("receptions", "pass_tds") else k) / mu
    out = np.empty(len(mu))
    for i, (j, t) in enumerate(zip(idx, thr)):
        rs = r["ratios"][j]
        out[i] = 1 - np.searchsorted(rs, t - 1e-12, side="left") / len(rs)
    return out


def prob_at_least(b: Bundle, name: str, mu: float, k: float) -> float:
    """예상값 mu인 선수가 실제로 k 이상 기록할 확률 (보정 후)."""
    raw = _raw_prob(b.resid[name], name, mu, k)
    return float(b.calib[name].predict(raw)[0]) if name in b.calib else float(raw[0])
