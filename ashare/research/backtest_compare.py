"""用完整回测引擎（成本、涨跌停、停牌、行业上限、持仓缓冲）比较多个打分方案。

python -m ashare.research.backtest_compare --feat data/features --scores data/research_v2 --v1 data/research
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BTConfig, metrics, run_backtest

PERIODS = [("2016-01-01", "2026-11-03", "16-26"), ("2016-01-01", "2019-12-31", "16-19"),
           ("2020-01-01", "2022-12-31", "20-22"), ("2023-01-01", "2026-11-03", "23-26"),
           ("2025-01-01", "2026-11-03", "25-26")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--scores", default="data/research_v2")
    ap.add_argument("--v1", default="data/research")
    ap.add_argument("--industry", default="data/industry.csv")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    F, S, V1 = Path(args.feat), Path(args.scores), Path(args.v1)
    u = pd.read_parquet(F / "universe.parquet")
    vwap = pd.read_parquet(F / "vwap.parquet")
    bb = pd.read_parquet(F / "buy_block.parquet")
    sb = pd.read_parquet(F / "sell_block.parquet")
    ep = pd.read_parquet(F / "f_ep.parquet")
    bench = pd.read_csv(V1 / "bench_全A等权(股票池).csv", index_col=0, parse_dates=True).iloc[:, 0]
    raw = {}
    for fp in list(S.glob("score_*.parquet")) + [V1 / "score_GBM5_P.parquet", V1 / "score_GBM20_P.parquet"]:
        if fp.exists():
            raw[fp.stem.replace("score_", "")] = pd.read_parquet(fp)
    me = raw["TAIL36"].index
    R = {k: v.reindex(me).rank(axis=1, pct=True) for k, v in raw.items()}
    cands = {
        "TAIL36": R["TAIL36"],
        "TAIL36+GBM5(0.7/0.3)": 0.7 * R["TAIL36"] + 0.3 * R["GBM5_P"],
        "TAIL36+GBM5(0.5/0.5)": 0.5 * R["TAIL36"] + 0.5 * R["GBM5_P"],
        "TAIL48": R.get("TAIL48"),
        "GBM5_P": R["GBM5_P"],
    }
    for k in ["RANK20", "CLS20", "REG5I"]:
        if k in R:
            cands[k] = R[k]
            cands[f"TAIL36+{k}(0.5/0.5)"] = 0.5 * R["TAIL36"] + 0.5 * R[k]
    if "RANK20" in R and "REG5I" in R:
        cands["TAIL36+RANK20+REG5I"] = (R["TAIL36"] + R["RANK20"] + R["REG5I"]) / 3
    cands = {k: v for k, v in cands.items() if v is not None}
    if args.only:
        cands = {k: v for k, v in cands.items() if k in args.only.split(",")}
    ind = pd.read_csv(args.industry).set_index("code")["l1"]
    ind_df = pd.DataFrame([ind.reindex(u.columns).values] * len(me), index=me, columns=u.columns)
    elig_base = u.reindex(me).fillna(False)
    profitable = ep.reindex(me).rank(axis=1, pct=True).gt(0.2) & ep.reindex(me).notna()
    # 2020 年前没有估值数据，价值过滤只在有数据时生效
    has_ep = ep.reindex(me).notna().sum(axis=1) > 1000
    elig_value = elig_base & (profitable | ~has_ep.values[:, None])
    rows = []
    for name, sc in cands.items():
        for el_tag, elig in [("全池", elig_base), ("价值过滤", elig_value)]:
            for buf in [0, 120, 250]:
                cfg = BTConfig(top_n=30, industry_cap=5, hold_buffer=buf)
                res = run_backtest(sc.where(elig), elig, vwap, bb, sb, cfg, industry=ind_df, start="2015-06-01")
                key = f"{name}|{el_tag}|buf{buf}"
                res.nav.to_csv(S / f"btnav_{name.replace('/', '_').replace('(', '_').replace(')', '')}_{el_tag}_buf{buf}.csv")
                for s, e, lab in PERIODS:
                    nav = res.nav[(res.nav.index >= s) & (res.nav.index <= e)]
                    nav = nav / nav.iloc[0]
                    tv = res.turnover[(res.turnover.index >= s) & (res.turnover.index <= e)]
                    m = metrics(nav, bench[(bench.index >= s) & (bench.index <= e)], tv)
                    rows.append({"model": key, "period": lab, "ann": m["年化收益"], "excess": m["超额年化"],
                                 "ir": m["信息比"], "mdd": m["最大回撤"], "ex_mdd": m["超额最大回撤"],
                                 "win": m["月胜率(跑赢基准)"], "turn": m["年换手(单边)"]})
                r0 = [r for r in rows if r["model"] == key]
                print(key, " | ".join(f"{r['period']}:ex{r['excess']:+.3f} ir{r['ir']:.2f}" for r in r0),
                      f"mdd{r0[0]['mdd']:.2f} turn{r0[0]['turn']:.1f}", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(S / "compare_table.csv", index=False)
    print(tab.pivot(index="model", columns="period", values="excess").round(3).to_string())


if __name__ == "__main__":
    main()
