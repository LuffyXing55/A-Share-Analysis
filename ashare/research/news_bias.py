"""新闻情绪的「涨停偏差」检验（没有历史新闻，只能用价格数据检验其后果）。

现有词典把「涨停」(+2)、「上涨」「新高」「放量」「突破」(+1) 这类描述股价本身的词当作利好。
一只股票涨停后，快讯（如「涨停分析」「异动」）会点名它，于是情绪分变高，短线分被加分。
这里检验：涨停之后（以及近期经常涨停的股票）接下来 5 / 20 个交易日相对股票池的超额收益是多少。
如果为负，说明「因为涨停而加分」方向是反的。

python -m ashare.research.news_bias --feat data/features --panels data/panels --out data/feedback
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..factors.library import limit_table
from .common import load_panels, st_proxy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--panels", default="data/panels")
    ap.add_argument("--out", default="data/feedback")
    a = ap.parse_args()
    F, out = Path(a.feat), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(F / "universe.parquet")
    p = load_panels(str(u.index[0].date()), basic=True, panel_dir=Path(a.panels))
    close = p["close"].reindex(index=u.index, columns=u.columns)
    lim = limit_table(close, st_proxy({**p, "close": close}).reindex(index=u.index, columns=u.columns))
    r = close.pct_change(fill_method=None)
    lu = (r >= lim - 0.005) & u
    freq = lu.astype(float).rolling(60, min_periods=20).sum()
    rows = []
    for h in (5, 20):
        y = pd.read_parquet(F / f"y_{h}.parquet").where(u)
        ex = y.sub(y.mean(axis=1), axis=0)
        for lab, m in {"当天涨停": lu, "近60日涨停≥3次": freq.ge(3) & u, "近60日无涨停": freq.eq(0) & u}.items():
            v = ex.where(m)
            s = v.mean(axis=1)                 # 每天这一类股票的平均超额
            n = m.sum(axis=1)
            s = s[n >= 5]
            yr = s.groupby(s.index.year).mean()
            row = {"持有期": h, "类别": lab, "全期日均超额": round(float(s.mean()), 4),
                   "日均只数": round(float(n[n >= 5].mean()), 1), "t值(粗略)": round(float(s.mean() / s.std() * len(s) ** 0.5 / (h ** 0.5)), 2)}
            for y_, x in yr.items():
                if y_ >= 2007:
                    row[str(y_)] = round(float(x), 4)
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "news_bias_limitup.csv", index=False)
    print(tab.iloc[:, :6].to_string())
    print(tab.to_string())


if __name__ == "__main__":
    main()
