"""研究 / 回测公共工具：加载面板、构建股票池、计算标签、IC 统计。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.library import limit_table

PANEL_DIR = Path(os.environ.get("ASHARE_PANELS", "data/panels"))
PRICE = ["open", "high", "low", "close", "vwap", "volume", "amount", "factor"]
BASIC = ["turnover_rate_f", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv"]


def load_panels(start: str = "2012-01-01", basic: bool = True, panel_dir: Path | None = None) -> dict:
    d = Path(panel_dir or PANEL_DIR)
    p = {}
    for f in PRICE:
        df = pd.read_parquet(d / f"{f}.parquet")
        p[f] = df.loc[df.index >= pd.Timestamp(start)]
    idx, cols = p["close"].index, p["close"].columns
    if basic:
        for f in BASIC:
            fp = d / f"{f}.parquet"
            if fp.exists():
                df = pd.read_parquet(fp)
                p[f] = df.reindex(index=idx, columns=cols).ffill(limit=5)
    return p


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().values)


def st_proxy(p: dict) -> pd.DataFrame:
    """历史 ST 近似：主板 + 亏损（PE 为空）+ 近 120 日最大涨跌幅不超过 5.3%。
    实盘中直接用股票名称是否含 ST 判断（腾讯行情返回名称）。"""
    r = p["close"].pct_change(fill_method=None).abs()
    max120 = r.rolling(120, min_periods=40).max()
    main = np.array([not c[2:].startswith(("300", "301", "688")) for c in p["close"].columns])
    cond = (max120 < 0.053) & (max120 > 0.045)
    if "pe_ttm" in p:
        loss = p["pe_ttm"].isna() | (p["pe_ttm"] <= 0)
        loss.loc[loss.index < pd.Timestamp("2020-02-01")] = True
        cond = cond & loss
    return cond & main[None, :]


def universe(p: dict, min_amount_wan: float = 2000.0, min_age: int = 250,
             drop_bottom_liq: float = 0.15, st: pd.DataFrame | None = None) -> pd.DataFrame:
    """可投资股票池（布尔表）。st：实盘时传入按股票名称判断的 ST 表，回测时用 st_proxy 近似。"""
    close, amount = p["close"], p["amount"]
    traded = close.notna() & (p["volume"] > 0)
    age = traded.cumsum()
    active60 = traded.rolling(60, min_periods=1).mean()
    amt20 = amount.rolling(20, min_periods=10).mean()          # 千元
    liq_ok = amt20 >= min_amount_wan * 10                      # 万元 → 千元
    liq_rank = amt20.rank(axis=1, pct=True)
    raw_price = close / p["factor"]
    u = traded & (age >= min_age) & (active60 >= 0.8) & liq_ok & (liq_rank > drop_bottom_liq)
    u &= raw_price >= 2.0
    u &= ~(st if st is not None else st_proxy(p))
    return u


def forward_return(p: dict, h: int, entry_lag: int = 1) -> pd.DataFrame:
    """以 t+entry_lag 日 vwap 买入、持有 h 个交易日后按 vwap 卖出的收益。停牌期间价格前向填充。"""
    v = p["vwap"].ffill(limit=60)
    return v.shift(-(entry_lag + h)) / v.shift(-entry_lag) - 1


def buy_blocked(p: dict, lim: pd.DataFrame | None = None) -> pd.DataFrame:
    """t+1 一字涨停 / 停牌 → 无法买入（返回在 t 行）。"""
    close, high, low = p["close"], p["high"], p["low"]
    r = close.pct_change(fill_method=None)
    lim = lim if lim is not None else limit_table(close)
    one_word_up = (high == low) & (r >= lim - 0.005)
    susp = close.isna()
    return (one_word_up | susp).shift(-1).fillna(True).astype(bool)


def sell_blocked(p: dict, lim: pd.DataFrame | None = None) -> pd.DataFrame:
    close, high, low = p["close"], p["high"], p["low"]
    r = close.pct_change(fill_method=None)
    lim = lim if lim is not None else limit_table(close)
    one_word_down = (high == low) & (r <= -lim + 0.005)
    susp = close.isna()
    return (one_word_down | susp).shift(-1).fillna(True).astype(bool)


def rank_ic(factor: pd.DataFrame, label: pd.DataFrame, mask: pd.DataFrame,
            dates: pd.DatetimeIndex) -> pd.Series:
    out = {}
    for d in dates:
        if d not in factor.index:
            continue
        m = mask.loc[d].values & np.isfinite(factor.loc[d].values) & np.isfinite(label.loc[d].values)
        if m.sum() < 100:
            continue
        x = pd.Series(factor.loc[d].values[m]).rank()
        y = pd.Series(label.loc[d].values[m]).rank()
        out[d] = float(np.corrcoef(x, y)[0, 1])
    return pd.Series(out)


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if len(ic) < 3:
        return {"n": len(ic)}
    t = ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else np.nan
    return {"n": len(ic), "ic": round(ic.mean(), 4), "icir": round(ic.mean() / ic.std(), 3),
            "t": round(t, 2), "pos": round((ic > 0).mean(), 3)}
