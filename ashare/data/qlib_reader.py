"""读取 Qlib 二进制格式（chenditc/investment_data 与 xu-duqing/investment_data 的 Release）。

Qlib `.day.bin` 文件格式：little-endian float32 数组，第 0 个元素是该序列在交易日历中的起始下标，
其后为逐日数值。

chenditc 数据包字段口径（2026-11 实测核对）：
- open/high/low/close/vwap：复权归一化价格，原始价格 = close / factor
- volume：手 / factor（原始成交量(手) = volume * factor）
- amount：成交额，单位千元（Tushare 口径）
- factor：复权因子；change：当日涨跌幅（小数）
xu-duqing daily_basic（Tushare daily_basic 原值）：pe_ttm、pb、ps_ttm、dv_ttm（%）、
total_mv/circ_mv（万元）、turnover_rate（%）等，起始日期为 2020-01-02。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_calendar(root: str | Path) -> pd.DatetimeIndex:
    path = Path(root) / "calendars" / "day.txt"
    return pd.DatetimeIndex(pd.to_datetime(path.read_text().split()))


def read_instruments(root: str | Path, name: str = "all") -> pd.DataFrame:
    path = Path(root) / "instruments" / f"{name}.txt"
    df = pd.read_csv(path, sep="\t", header=None, names=["code", "start", "end"])
    df["code"] = df["code"].str.lower()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    return df


def read_series(path: str | Path, calendar: pd.DatetimeIndex) -> pd.Series | None:
    if not os.path.exists(path):
        return None
    arr = np.fromfile(path, dtype="<f4")
    if arr.size < 2:
        return None
    start = int(arr[0])
    values = arr[1:]
    end = min(start + len(values), len(calendar))
    return pd.Series(values[: end - start], index=calendar[start:end])


def build_panel(
    feature_root: str | Path,
    field: str,
    calendar: pd.DatetimeIndex,
    codes: Iterable[str],
    start: str | None = None,
) -> pd.DataFrame:
    """返回 日期 × 股票 的宽表（float32）。"""
    cal = calendar
    if start is not None:
        cal = calendar[calendar >= pd.Timestamp(start)]
    out = {}
    for code in codes:
        s = read_series(Path(feature_root) / code / f"{field}.day.bin", calendar)
        if s is None:
            continue
        s = s[s.index >= cal[0]]
        if len(s):
            out[code] = s
    df = pd.DataFrame(out, index=cal, dtype="float32")
    df.index.name = "date"
    return df


def index_membership(root: str | Path, name: str, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """指数成分历史 → 日期 × 股票 的布尔表。"""
    inst = read_instruments(root, name)
    codes = sorted(inst["code"].unique())
    mat = pd.DataFrame(False, index=calendar, columns=codes)
    for row in inst.itertuples(index=False):
        mat.loc[(calendar >= row.start) & (calendar <= row.end), row.code] = True
    return mat
