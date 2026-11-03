"""预计算因子截面排名（股票池内百分位），缓存为 parquet，供模型实验与训练使用。

python -m ashare.research.build_features --out data/features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.library import compute_factors, limit_table
from .common import (buy_blocked, forward_return, load_panels, sell_blocked, st_proxy, universe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--start", default="2014-01-01")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = load_panels("2012-01-01")
    u = universe(p)
    st = st_proxy(p)
    lim = limit_table(p["close"], st)
    keep = p["close"].index >= pd.Timestamp(args.start)
    fac = compute_factors(p)
    for name, f in fac.items():
        r = f.where(u).rank(axis=1, pct=True).astype("float32")
        r.loc[keep].to_parquet(out / f"f_{name}.parquet")
        print("feature", name)
    u.loc[keep].to_parquet(out / "universe.parquet")
    for h in (1, 5, 10, 20):
        forward_return(p, h).loc[keep].astype("float32").to_parquet(out / f"y_{h}.parquet")
    buy_blocked(p, lim).loc[keep].to_parquet(out / "buy_block.parquet")
    sell_blocked(p, lim).loc[keep].to_parquet(out / "sell_block.parquet")
    p["vwap"].loc[keep].to_parquet(out / "vwap.parquet")
    mv = p.get("total_mv")
    if mv is not None:
        np.log(mv.where(mv > 0)).loc[keep].astype("float32").to_parquet(out / "log_mv.parquet")
    print("done")


if __name__ == "__main__":
    main()
