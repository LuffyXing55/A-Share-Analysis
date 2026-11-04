"""「尾部加权」合成分（中期分 / 价值投资分的核心）：实盘版本。

- config/tail_table.csv：每个月末、每个因子「前 5% 股票」在随后 20 个交易日的平均超额收益（研究期由回测数据计算；
  上线后由流水线自动追加）。这是统计结果，不含任何原始行情。
- 每月末：记录各因子前 5% 的股票名单（state/tail_members_YYYY-MM-DD.json）；
- 20 个交易日之后：用价格面板计算这些名单的已实现超额收益，追加进 tail_table。
- 权重：过去 36 个月的 t 值（只保留 t ≥ 0.5 的因子，上限 4），只使用已完全实现的月份，避免未来函数。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK, MIN_T, MIN_OBS = 36, 0.5, 18


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, parse_dates=True)


def weights(table: pd.DataFrame, asof: pd.Timestamp, lookback: int = LOOKBACK, min_t: float = MIN_T,
            available: set[str] | None = None, realized: pd.Series | None = None,
            min_obs: int = MIN_OBS) -> dict[str, float]:
    """realized：每个月末标签的「实现日」（该月末之后第 21 个交易日）。只使用实现日不晚于 asof 的月份。
    实盘不传 realized：表里的行只有在标签实现后才会被追加，所以直接用 asof 之前的全部行。"""
    if realized is not None:
        ok = realized.reindex(table.index)
        hist = table[(ok <= asof).fillna(False).values]
    else:
        hist = table[table.index < asof]
    hist = hist.tail(lookback)
    w = {}
    for c in hist.columns:
        if available is not None and c not in available:
            continue
        x = hist[c].dropna()
        if len(x) < min_obs:
            continue
        t = x.mean() / (x.std() / np.sqrt(len(x)) + 1e-12)
        if t >= min_t:
            w[c] = float(min(t, 4.0))
    return w


def score(ranks: dict[str, pd.Series], w: dict[str, float], codes) -> pd.Series:
    tot, ws = pd.Series(0.0, index=codes), 0.0
    for f, x in w.items():
        if f not in ranks:
            continue
        tot = tot + ranks[f].reindex(codes).fillna(0.5) * x
        ws += x
    return tot / ws if ws else pd.Series(np.nan, index=codes)


def contributions(ranks: dict[str, pd.Series], w: dict[str, float], code: str) -> list[dict]:
    ws = sum(w.values()) or 1.0
    items = []
    for f, x in w.items():
        r = ranks.get(f, pd.Series(dtype=float)).get(code)
        r = 0.5 if r is None or not np.isfinite(r) else float(r)
        items.append({"factor": f, "contrib": round((r - 0.5) * x / ws, 4), "pct": round(r * 100, 1)})
    return sorted(items, key=lambda z: -abs(z["contrib"]))


def record_members(state: Path, d: pd.Timestamp, ranks: dict[str, pd.Series], mask: pd.Series) -> None:
    members = {f: list(r[(r >= 0.95)].index) for f, r in ranks.items()}
    (state / f"tail_members_{d.date()}.json").write_text(json.dumps(
        {"date": str(d.date()), "universe": list(mask[mask].index), "members": members}, ensure_ascii=False))


def update_table(state: Path, table_path: Path, adj_vwap: pd.DataFrame, horizon: int = 20,
                 last_is_snapshot: bool = False) -> pd.DataFrame:
    """把已满 horizon 个交易日的月末名单的已实现超额收益追加进表。
    口径与研究完全一致：t+1 日复权均价买入、t+1+horizon 日复权均价卖出（停牌前向填充），相对股票池等权。"""
    table = load_table(table_path)
    px = adj_vwap.ffill(limit=60)
    idx = px.index
    # 面板最后一行若是用实时快照拼出的（不是官方日线），不把它当作卖出日，推迟一天再计算
    last_ok = len(idx) - 2 if last_is_snapshot else len(idx) - 1
    changed = False
    for fp in sorted(state.glob("tail_members_*.json")):
        obj = json.loads(fp.read_text())
        d = pd.Timestamp(obj["date"])
        if d in table.index or d not in idx:
            continue
        pos = idx.get_loc(d)
        if pos + 1 + horizon > last_ok:
            continue
        a, b = idx[pos + 1], idx[pos + 1 + horizon]
        ret = px.loc[b] / px.loc[a] - 1
        uni = ret.reindex(obj["universe"]).dropna()
        base = uni.mean()
        row = {f: float(ret.reindex(m).dropna().mean() - base) for f, m in obj["members"].items() if m}
        table = pd.concat([table, pd.DataFrame([row], index=[d])])   # concat 会自动扩列，不丢新因子
        changed = True
    if changed:
        table = table.sort_index()
        table.to_csv(table_path, float_format="%.6f")
    return table


def label_realized_dates(month_ends: pd.DatetimeIndex, calendar: pd.DatetimeIndex, horizon: int = 20) -> pd.Series:
    """月末 k 的标签（t+1 → t+1+horizon 均价收益）在第 pos(k)+1+horizon 个交易日收盘后才可知。"""
    out = {}
    for k in month_ends:
        pos = calendar.searchsorted(k)
        j = pos + 1 + horizon
        out[k] = calendar[j] if j < len(calendar) else pd.Timestamp.max
    return pd.Series(out)


def backfill_members(state: Path, d: pd.Timestamp, ranks: dict[str, pd.Series], mask: pd.Series,
                     partial: list[str] | None = None) -> None:
    """补记某个过去月末的前 5% 名单（例如那天流水线没跑成）。只在文件不存在时写入；
    partial 记录缺失的因子（历史月末没有当时的估值快照）。"""
    fp = state / f"tail_members_{d.date()}.json"
    if not fp.exists():
        record_members(state, d, ranks, mask)
        if partial:
            obj = json.loads(fp.read_text())
            obj["partial"] = partial
            fp.write_text(json.dumps(obj, ensure_ascii=False))
