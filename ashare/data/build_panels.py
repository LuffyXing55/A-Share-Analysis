"""把 Qlib 二进制历史数据转换成 Parquet 宽表，供回测与训练使用。

用法：
    python -m ashare.data.build_panels --qlib data_raw/qlib --basic data_raw/db/daily_basic \
        --out data/panels --start 2012-01-01

数据下载（只在本地/研究环境使用，不要提交到公开仓库）：
    https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz  (Apache-2.0)
    https://github.com/xu-duqing/investment_data/releases （daily_basic_qlib_features.tar.gz，可选）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .qlib_reader import build_panel, index_membership, read_calendar, read_instruments

PRICE_FIELDS = ["open", "high", "low", "close", "vwap", "volume", "amount", "factor", "change"]
BASIC_FIELDS = ["turnover_rate", "turnover_rate_f", "pe_ttm", "pb", "ps_ttm", "dv_ttm",
                "total_mv", "circ_mv", "free_share", "total_share"]


def is_target_board(code: str) -> bool:
    """沪深主板 + 创业板 + 科创板（剔除北交所与 B 股）。"""
    if code.startswith("sh"):
        return code[2:5] in {"600", "601", "603", "605", "688"}
    if code.startswith("sz"):
        return code[2:5] in {"000", "001", "002", "003", "300", "301"}
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib", required=True)
    ap.add_argument("--basic", default=None)
    ap.add_argument("--out", default="data/panels")
    ap.add_argument("--start", default="2012-01-01")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cal = read_calendar(args.qlib)
    inst = read_instruments(args.qlib, "all")
    codes = sorted(c for c in inst["code"].unique() if is_target_board(c))
    inst[inst["code"].isin(codes)].to_parquet(out / "instruments.parquet")
    print(f"{len(codes)} instruments, calendar {cal[0].date()} .. {cal[-1].date()}")

    for f in PRICE_FIELDS:
        df = build_panel(Path(args.qlib) / "features", f, cal, codes, args.start)
        df.to_parquet(out / f"{f}.parquet")
        print("price", f, df.shape)

    if args.basic:
        for f in BASIC_FIELDS:
            df = build_panel(Path(args.basic) / "features", f, cal, codes, "2020-01-01")
            df.to_parquet(out / f"{f}.parquet")
            print("basic", f, df.shape)

    for idx in ["csi300", "csi500", "csi1000"]:
        m = index_membership(args.qlib, idx, cal[cal >= pd.Timestamp(args.start)])
        m = m[[c for c in m.columns if c in set(codes)]]
        m.to_parquet(out / f"member_{idx}.parquet")
        print("member", idx, m.shape)


if __name__ == "__main__":
    main()
