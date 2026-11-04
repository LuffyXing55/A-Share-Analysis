"""盘中实时重算（本地 Mac 模式）。

思路：收盘后流水线保存了「全市场因子分布分位点」和「模型输出分布」（state/factor_dist.json）。
盘中只需要为 30 只入选股拉取腾讯日 K（每天一次）+ 实时行情（每 1 分钟），把今天的未完成 K 线
（成交量按已交易时长折算成全天）拼进去，用同一份因子代码计算原始因子，再对照全市场分布换算成分位，
就能用同一个模型给出「盘中实时」的价值分与短线分。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..factors.library import compute_factors
from . import sources as S

BJ = timezone(timedelta(hours=8))


def session_fraction(now: datetime) -> float:
    """已经交易的时间占全天 240 分钟的比例（用于把盘中成交量折算成全天）。"""
    t = now.hour * 60 + now.minute
    am = max(0, min(t, 11 * 60 + 30) - (9 * 60 + 30))
    pm = max(0, min(t, 15 * 60) - 13 * 60)
    return max(0.02, min(1.0, (am + pm) / 240))


def build_mini_panels(hist: dict[str, pd.DataFrame], quotes: dict[str, dict], now: datetime) -> dict:
    """hist：code → 腾讯前复权日 K（date, open, close, high, low, volume[手]）。"""
    frames = {}
    today = pd.Timestamp(now.date())
    frac = session_fraction(now)
    # 9:25–9:30 只有集合竞价成交，不做全天折算（否则成交量会被放大 50 倍）
    in_session = now.weekday() < 5 and (9 * 60 + 30) <= now.hour * 60 + now.minute <= 15 * 60 + 5
    for code, df in hist.items():
        df = df.copy().set_index("date").sort_index()
        q = quotes.get(code)
        if q and q.get("price") and q.get("volume_lots"):
            qt = str(q.get("time", ""))[:8]
            if qt == today.strftime("%Y%m%d"):
                scale = 1.0 / frac if in_session else 1.0
                df.loc[today] = {"open": q["open"], "close": q["price"], "high": q["high"], "low": q["low"],
                                 "volume": q["volume_lots"] * scale}
        frames[code] = df
    idx = sorted(set().union(*[f.index for f in frames.values()])) if frames else []
    idx = pd.DatetimeIndex(idx)
    p = {}
    for fld in ["open", "close", "high", "low", "volume"]:
        p[fld] = pd.DataFrame({c: f[fld] for c, f in frames.items()}).reindex(idx).astype("float64")
    p["factor"] = pd.DataFrame(1.0, index=idx, columns=p["close"].columns)
    p["vwap"] = (p["open"] + p["high"] + p["low"] + p["close"]) / 4
    p["amount"] = p["volume"] * 100 * p["close"] / 1000.0             # 千元，与回测口径一致
    # 基本面：最新一天来自实时行情
    float_shares = pd.Series({c: (q.get("float_mv") or np.nan) / q["price"] for c, q in quotes.items()
                              if q.get("price")})
    total_shares = pd.Series({c: (q.get("total_mv") or np.nan) / q["price"] for c, q in quotes.items()
                              if q.get("price")})
    cols = p["close"].columns
    p["circ_mv"] = p["close"].mul(float_shares.reindex(cols), axis=1) / 1e4
    p["total_mv"] = p["close"].mul(total_shares.reindex(cols), axis=1) / 1e4
    pe = pd.DataFrame(np.nan, index=idx, columns=cols)
    pb = pe.copy()
    pe.iloc[-1] = [quotes.get(c, {}).get("pe_ttm") or np.nan for c in cols]
    pb.iloc[-1] = [quotes.get(c, {}).get("pb") or np.nan for c in cols]
    p["pe_ttm"], p["pb"] = pe, pb
    return p


def attach_market(p: dict, mkt_hist: dict | None, today_mkt: float | None) -> dict:
    """特质波动需要「全市场等权日收益」：历史部分来自收盘流水线保存的序列，今天用中证全指涨跌幅近似。"""
    if not mkt_hist:
        return p
    s = pd.Series({pd.Timestamp(k): v for k, v in mkt_hist.items()}, dtype="float64")
    idx = p["close"].index
    m = s.reindex(idx)
    if today_mkt is not None and pd.isna(m.iloc[-1]):
        m.iloc[-1] = today_mkt
    p["mkt_ret"] = m
    return p


def pct_from_dist(x: float, dist: list[float]) -> float:
    """按全市场分位点把原始值换算成分位（0~1）。遇到并列值（离散因子，如近 5 日涨停次数）取并列区间的中点，
    与收盘流水线 rank(pct=True) 的平均排名口径一致。"""
    if x is None or not np.isfinite(x) or not dist:
        return np.nan
    q = np.array([v if v is not None else np.nan for v in dist], dtype=float)
    if np.isnan(q).any():
        return np.nan
    n = len(q) - 1
    lo, hi = np.searchsorted(q, x, "left"), np.searchsorted(q, x, "right")
    if hi - lo >= 2:
        return float((lo + hi - 1) / 2 / n)
    return float(np.interp(x, q, np.linspace(0, 1, len(q))))


def live_scores(models: dict, fdist: dict, hist: dict, quotes: dict, now: datetime | None = None,
                fallback_ranks: dict | None = None, today_mkt: float | None = None,
                pred_dist_key: str = "pred_dist") -> dict:
    now = now or datetime.now(BJ)
    p = build_mini_panels(hist, quotes, now)
    p = attach_market(p, fdist.get("mkt_ret"), today_mkt)
    fac = compute_factors(p)
    last = {k: v.iloc[-1] for k, v in fac.items()}
    dist = fdist.get("dist", {})
    codes = list(p["close"].columns)
    ranks = {k: pd.Series({c: pct_from_dist(last[k].get(c), dist.get(k)) for c in codes}) for k in last}
    # 盘中算不出来的因子（行业动量等）沿用昨日收盘时的分位
    for m in models.values():
        for f in m.features:
            if f not in ranks or ranks[f].isna().all():
                ranks[f] = pd.Series({c: ((fallback_ranks or {}).get(c) or {}).get(f, np.nan) for c in codes},
                                     dtype="float64")
    out = {c: {} for c in codes}
    pdist = fdist.get(pred_dist_key) or {}
    tail_w = fdist.get("tail_weights") or {}
    try:   # 与收盘流水线一致：剔除 config/settings.json 里 exclude_factors 列出的因子（例如方案 A 的 size）
        import json as _json
        from pathlib import Path as _P

        _ex = set(_json.loads((_P(__file__).resolve().parents[2] / "config" / "settings.json").read_text())
                  .get("exclude_factors", []))
        tail_w = {k: v for k, v in tail_w.items() if k not in _ex}
    except Exception:  # noqa: BLE001
        pass
    if tail_w:
        from ..model.tail import score as tail_score

        v = tail_score(ranks, tail_w, codes)
        for c in codes:
            out[c]["value"] = round(pct_from_dist(v[c], pdist.get("value")) * 100, 1) if pdist.get("value") else None
    for key, m in models.items():
        X = np.column_stack([ranks[f].reindex(codes).values if f in ranks else np.full(len(codes), np.nan)
                             for f in m.features])
        pred = m.predict(X)
        for c, v in zip(codes, pred):
            out[c][key] = round(pct_from_dist(v, pdist.get(key)) * 100, 1) if pdist.get(key) else None
    return out


def fetch_histories(codes: list[str], days: int = 320) -> dict[str, pd.DataFrame]:
    out = {}
    for c in codes:
        try:
            df = S.tencent_kline(c, days, "qfq")
            if len(df) > 60:
                out[c] = df
        except Exception:  # noqa: BLE001
            continue
    return out
