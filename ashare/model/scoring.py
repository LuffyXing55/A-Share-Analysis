"""打分模型：线性合成（等权 / 滚动 ICIR 加权）与 LightGBM 走步训练。

数据形态：features 为 dict[name -> 宽表(日期×股票)]，取值为股票池内截面百分位（0~1）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def stack_features(feats: dict[str, pd.DataFrame], names: list[str], dates) -> np.ndarray:
    """→ [n_dates, n_stocks, n_feat]"""
    arr = np.stack([feats[n].reindex(dates).values.astype("float32") for n in names], axis=-1)
    return arr


def linear_score(feats: dict[str, pd.DataFrame], weights: dict[str, float], dates) -> pd.DataFrame:
    tot, wsum = None, None
    for n, w in weights.items():
        f = feats[n].reindex(dates)
        v = (f - 0.5).fillna(0.0) * w
        tot = v if tot is None else tot + v
        m = f.notna().astype("float32") * abs(w)
        wsum = m if wsum is None else wsum + m
    s = tot / wsum.where(wsum > 0)
    return s


def rolling_icir_weights(ic_table: pd.DataFrame, asof: pd.Timestamp, lookback: int = 24,
                         min_obs: int = 12, shrink: float = 0.5) -> dict[str, float]:
    """ic_table：index=信号日期，列=因子，值=该日 IC（标签期结束于下一调仓日附近）。
    只使用标签已经完全实现的历史 IC（日期 < asof - 1 个月），避免未来函数。"""
    hist = ic_table[ic_table.index < asof - pd.DateOffset(days=31)].tail(lookback)
    w = {}
    for c in hist.columns:
        x = hist[c].dropna()
        if len(x) < min_obs:
            continue
        icir = x.mean() / (x.std() + 1e-9)
        w[c] = float(np.clip(icir, -2, 2))
    if not w:
        return {}
    # 向等权收缩，提高稳健性
    ew = {k: np.sign(v) * 1.0 for k, v in w.items()}
    mean_abs = np.mean([abs(v) for v in w.values()]) or 1.0
    return {k: shrink * w[k] / mean_abs + (1 - shrink) * ew[k] for k in w if w[k] > 0}


@dataclass
class GBMConfig:
    num_leaves: int = 31
    learning_rate: float = 0.04
    n_estimators: int = 350
    min_child_samples: int = 300
    feature_fraction: float = 0.7
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    lambda_l2: float = 5.0
    max_depth: int = 6
    seed: int = 7
    extra: dict = field(default_factory=dict)


def _rank_gauss(y: np.ndarray) -> np.ndarray:
    from scipy.special import erfinv

    out = np.full_like(y, np.nan, dtype="float32")
    m = np.isfinite(y)
    if m.sum() < 10:
        return out
    r = pd.Series(y[m]).rank(pct=True).values
    r = np.clip(r, 0.001, 0.999)
    out[m] = (np.sqrt(2) * erfinv(2 * r - 1)).astype("float32")
    return out


def make_xy(X3: np.ndarray, Y: np.ndarray, U: np.ndarray, rows: np.ndarray):
    """把若干日期的截面拼成训练样本。X3:[T,N,F] Y:[T,N] U:[T,N] rows: 日期下标"""
    xs, ys = [], []
    for t in rows:
        m = U[t] & np.isfinite(Y[t])
        if m.sum() < 100:
            continue
        xs.append(X3[t][m])
        ys.append(_rank_gauss(Y[t][m]))
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


def train_gbm(X: np.ndarray, y: np.ndarray, cfg: GBMConfig):
    import lightgbm as lgb

    params = dict(objective="regression", num_leaves=cfg.num_leaves, learning_rate=cfg.learning_rate,
                  min_child_samples=cfg.min_child_samples, feature_fraction=cfg.feature_fraction,
                  bagging_fraction=cfg.bagging_fraction, bagging_freq=cfg.bagging_freq,
                  lambda_l2=cfg.lambda_l2, max_depth=cfg.max_depth, seed=cfg.seed, verbose=-1,
                  num_threads=4, **cfg.extra)
    ds = lgb.Dataset(X, y, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=cfg.n_estimators)


def walk_forward_gbm(X3: np.ndarray, Y: np.ndarray, U: np.ndarray, dates: pd.DatetimeIndex,
                     pred_rows: np.ndarray, horizon: int, cfg: GBMConfig, retrain_months: int = 6,
                     train_years: float | None = None, sample_every: int = 5,
                     min_train_rows: int = 120, log=print):
    """走步训练：每 retrain_months 个月重训一次，只用标签已完全实现的样本（样本日 + horizon + 1 < 训练日）。
    返回 pred[T,N]（只在 pred_rows 上有值）和每次训练的特征重要性。"""
    T, N, F = X3.shape
    pred = np.full((T, N), np.nan, dtype="float32")
    model, last_train = None, None
    importances = []
    for t in pred_rows:
        d = dates[t]
        need = last_train is None or (d.year * 12 + d.month) - (last_train.year * 12 + last_train.month) >= retrain_months
        if need:
            end = t - horizon - 2
            if end < min_train_rows:
                continue
            start = 0
            if train_years:
                start = max(0, end - int(train_years * 244))
            rows = np.arange(start, end, sample_every)
            Xtr, ytr = make_xy(X3, Y, U, rows)
            if Xtr is None:
                continue
            model = train_gbm(Xtr, ytr, cfg)
            last_train = d
            importances.append((d, model.feature_importance("gain")))
            log(f"  retrain @ {d.date()} rows={len(ytr)}")
        if model is None:
            continue
        m = U[t]
        if m.sum() == 0:
            continue
        pred[t, m] = model.predict(X3[t][m])
    return pred, importances
