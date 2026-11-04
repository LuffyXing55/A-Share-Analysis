"""全市场快照（腾讯行情，约 90 个请求 / 1–2 分钟）：名称、ST 标记、估值、市值、当日 K 线。

用途：
1. 补齐基本面面板（pe_ttm、pb、total_mv、circ_mv），口径与回测一致；
2. 当 GitHub Release 还没更新到今天时，用快照拼出今天的日线（价格乘以最近复权因子换成复权价）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..live import sources as S


def market_snapshot(codes: list[str]) -> pd.DataFrame:
    rows = S.tencent_quotes(codes)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("code").set_index("code")
    # 真实行情里可能有 None / 空串，统一转成数值（float64）
    for col in ["price", "prev_close", "open", "high", "low", "volume_lots", "pct", "amount", "turnover",
                "pe_ttm", "float_mv", "total_mv", "pb", "limit_up", "limit_down", "vol_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["is_st"] = df["name"].str.contains("ST", na=False)
    df["is_delist"] = df["name"].str.contains("退", na=False)
    return df


def basics_from_snapshot(p: dict, snap: pd.DataFrame) -> dict:
    """生成与回测同口径的基本面面板：只有最新一天有 pe/pb/市值；流通市值按流通股本不变回推历史。"""
    close, factor = p["close"], p["factor"]
    idx, cols = close.index, close.columns
    snap = snap.reindex(cols)
    raw_last = (close / factor).iloc[-1]
    price = pd.to_numeric(snap["price"], errors="coerce").where(lambda x: x > 0, raw_last)
    float_shares = snap["float_mv"] / price            # 股
    total_shares = snap["total_mv"] / price
    raw = close / factor
    circ_mv = raw.mul(float_shares, axis=1) / 1e4      # 万元
    total_mv = raw.mul(total_shares, axis=1) / 1e4
    pe = pd.DataFrame(np.nan, index=idx, columns=cols, dtype="float64")
    pb = pe.copy()
    pe_v = pd.to_numeric(snap["pe_ttm"], errors="coerce").astype("float64")
    pb_v = pd.to_numeric(snap["pb"], errors="coerce").astype("float64")
    pe.iloc[-1] = pe_v.where(pe_v != 0).values
    pb.iloc[-1] = pb_v.where(pb_v != 0).values
    return {"pe_ttm": pe, "pb": pb, "total_mv": total_mv.astype("float32"),
            "circ_mv": circ_mv.astype("float32")}


def append_today_bar(p: dict, snap: pd.DataFrame, today: pd.Timestamp) -> dict:
    """若面板最后一天早于 today，则用快照拼出今天的一行（复权价 = 原始价 × 最近复权因子）。"""
    if p["close"].index[-1] >= today:
        return p
    cols = p["close"].columns
    s = snap.reindex(cols)
    f_last = p["factor"].ffill().iloc[-1]
    traded = (s["volume_lots"] > 0) & (s["price"] > 0)
    row = {
        "open": s["open"] * f_last, "high": s["high"] * f_last, "low": s["low"] * f_last,
        "close": s["price"] * f_last,
        "vwap": (s["amount"] / (s["volume_lots"] * 100)).where(traded) * f_last,
        "volume": s["volume_lots"] / f_last, "amount": s["amount"] / 1000.0, "factor": f_last,
    }
    out = {}
    for k, v in p.items():
        if k in row:
            vals = row[k] if k == "factor" else row[k].where(traded)   # 复权因子即使停牌也沿用
            new = pd.DataFrame([vals.values],
                               index=[today], columns=cols, dtype="float32")
            out[k] = pd.concat([v, new])
        else:
            out[k] = v
    return out
