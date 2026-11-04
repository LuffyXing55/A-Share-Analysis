"""「尾部加权」走步合成：每个月只用过去 36 个月已实现的数据，挑选「前 5% 组合超额」稳定为正的因子并加权。

与 IC 加权不同，这里的权重依据是因子「做多端」的表现（只做多 Top30 需要的正是这一端），
而不是整个截面的相关性（A 股多数因子的 IC 主要来自空头端）。

python -m ashare.research.tail_composite --feat data/features --out data/research_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..factors.library import PROD_FACTORS
from .common import month_ends

EXTRA = ["ind_mom_20", "ind_mom_60", "rel_ind_20", "rel_ind_60"]


def tail_table(feats: dict, ym: pd.DataFrame, q: float = 0.95) -> pd.DataFrame:
    """每个调仓日、每个因子：前 5% 股票的下期超额收益（等权）。"""
    rows = {}
    for n, f in feats.items():
        top = f.ge(q)
        rows[n] = ym.where(top).mean(axis=1)
    return pd.DataFrame(rows)


def tail_weights(tt: pd.DataFrame, asof: pd.Timestamp, lookback: int = 36, min_obs: int = 18,
                 min_t: float = 1.0, realized: pd.Series | None = None) -> dict:
    """与实盘共用 ashare.model.tail.weights：只用「标签实现日 ≤ asof」的月份（按交易日判断，不用自然日近似）。"""
    from ..model import tail as TAIL

    return TAIL.weights(tt, asof, lookback=lookback, min_t=min_t, realized=realized)


def composite(feats: dict, w: dict, d) -> pd.Series:
    tot, ws = 0.0, 0.0
    for n, x in w.items():
        tot = tot + feats[n].loc[d].fillna(0.5) * x
        ws += x
    return tot / ws if ws else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--out", default="data/research_v2")
    ap.add_argument("--lookback", type=int, default=36)
    ap.add_argument("--min-t", type=float, default=1.0)
    args = ap.parse_args()
    F, out = Path(args.feat), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(F / "universe.parquet")
    y = pd.read_parquet(F / "y_20.parquet")
    me = month_ends(u.index)
    me = me[me >= pd.Timestamp("2014-06-01")][:-1]
    uu = u.loc[me]
    ym = y.loc[me].where(uu)
    ym = ym.sub(ym.mean(axis=1), axis=0)
    names = [n for n in PROD_FACTORS + EXTRA if (F / f"f_{n}.parquet").exists()]
    feats = {n: pd.read_parquet(F / f"f_{n}.parquet").loc[me].where(uu) for n in names}
    tt = tail_table(feats, ym)
    tt.to_csv(out / "tail_table.csv")
    from ..model.tail import label_realized_dates

    realized = label_realized_dates(me, u.index)
    rows, wlog = [], {}
    for d in me:
        w = tail_weights(tt, d, args.lookback, min_t=args.min_t, realized=realized)
        if not w:
            continue
        # 只在股票池内给分（与实盘一致：池外股票没有分数，不参与分位排名）
        s = composite(feats, w, d).where(uu.loc[d])
        rows.append(s.rename(d))
        wlog[str(d.date())] = {k: round(v, 2) for k, v in sorted(w.items(), key=lambda x: -x[1])}
    sc = pd.DataFrame(rows)
    sc.to_parquet(out / f"score_TAIL{args.lookback}_t{args.min_t}.parquet")
    (out / f"tail_weights_{args.lookback}.json").write_text(json.dumps(wlog, ensure_ascii=False, indent=0))
    top = sc.rank(axis=1, ascending=False) <= 30
    ex = ym.reindex(sc.index).where(top).mean(axis=1)
    print({p: round(float(ex.loc[a:b].mean() * 100), 2) for p, (a, b) in
           {"16-19": ("2016", "2019"), "20-22": ("2020", "2022"), "23-26": ("2023", "2026")}.items()})
    print("last weights:", list(wlog.items())[-1])


if __name__ == "__main__":
    main()
