"""稳健性检验：30 强的回测结果对「短线模型的具体版本」有多敏感？以及每周重选 vs 每月重选。

发现的问题：同一套规则，只把短线模型换成另一次走步训练的版本（重训的月份不同，整体预测力 IC 基本相同），
2020–2024 年年化超额就从 +16% 掉到 +4%。说明 Top30 对模型的细微差别很敏感，单一版本的回测数字不可靠。

做法：
1. gbm 子命令：用不同的重训月份（1/7 月、3/9 月、5/11 月）× 不同随机种子（7、11）走步训练 6 个短线模型，
   每个版本都同时在「每月最后一个交易日」和「每周最后一个交易日」给出预测；
2. eval 子命令：对每个版本分别跑「每月重选（缓冲 400）」和「每周重选（缓冲 200 / 400 / 600）」，
   其余规则与现行策略一致（价值分 5 组市值中性化、0.6/0.4、盈利且 PE 低于 70 分位、行业 ≤5、Top30），
   报告 2020-02~2024、2025 至今、全期的分布。

python -m ashare.research.robust_eval gbm  --feat data/features --out data/robust
python -m ashare.research.robust_eval eval --feat data/features --final data/final --tech data/tech --research data/research \
       --panels data/panels --industry data/industry.csv --out data/robust
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BTConfig, period_metrics, run_backtest
from ..factors.library import PROD_FACTORS
from ..model.scoring import GBMConfig, walk_forward_gbm
from .common import month_ends
from .neutral_eval import bucket_pct
from .tech_eval import week_ends

EXCLUDE = {"size"}
ANCHORS = [1, 3, 5]
SEEDS = [7, 11]
PERIODS = {"sel": ("2020-02-04", "2024-12-31"), "hold": ("2025-01-01", "2026-12-31"), "full": ("2020-02-04", "2026-12-31")}


def cmd_gbm(a) -> None:
    F, out = Path(a.feat), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(F / "universe.parquet")
    dates = u.index
    names = [n for n in PROD_FACTORS if n not in EXCLUDE]
    X3 = np.stack([pd.read_parquet(F / f"f_{n}.parquet").values.astype("float32") for n in names], axis=-1)
    y5 = pd.read_parquet(F / "y_5.parquet").values
    rows_all = month_ends(dates)[:-1].union(week_ends(dates)[:-1])
    pos = {d: i for i, d in enumerate(dates)}
    for anchor, seed in itertools.product(ANCHORS, SEEDS):
        fp = out / f"gbm_a{anchor}_s{seed}.parquet"
        if fp.exists():
            continue
        start = pd.Timestamp(f"2015-{anchor:02d}-01")
        sel = rows_all[rows_all >= start]
        rows = np.array([pos[d] for d in sel])
        pred, _ = walk_forward_gbm(X3, y5, u.values, dates, rows, 5, GBMConfig(seed=seed),
                                   retrain_months=6, train_years=6, sample_every=5,
                                   log=lambda m: print(f"[a{anchor} s{seed}] {m}", flush=True))
        pd.DataFrame(pred[rows], index=sel, columns=u.columns).to_parquet(fp)
        print("saved", fp, flush=True)


def cmd_eval(a) -> None:
    from .size_eval import tail_scores

    F, out = Path(a.feat), Path(a.out)
    u = pd.read_parquet(F / "universe.parquet")
    vwap = pd.read_parquet(F / "vwap.parquet")
    bb = pd.read_parquet(F / "buy_block.parquet")
    sb = pd.read_parquet(F / "sell_block.parquet")
    bench = pd.read_csv(Path(a.research) / "bench_全A等权(股票池).csv", index_col=0, parse_dates=True).iloc[:, 0]
    mv = pd.read_parquet(Path(a.panels) / "total_mv.parquet").reindex(columns=u.columns) / 1e4
    pe = pd.read_parquet(Path(a.panels) / "pe_ttm.parquet").reindex(columns=u.columns)
    ind = pd.read_csv(a.industry).set_index("code")["l1"].reindex(u.columns)
    tm, _ = tail_scores(F, Path(a.final), EXCLUDE)
    tw = pd.read_parquet(Path(a.tech) / "score_TAIL_weekly.parquet")

    def run(dates, T_src, G_src, buf):
        dates = pd.DatetimeIndex(dates)
        base = u.reindex(dates).fillna(False)
        mvd = mv.reindex(mv.index.union(dates)).ffill().reindex(dates).where(base)
        lm = np.log(mvd.where(mvd > 0))
        T = T_src.reindex(dates).where(base).rank(axis=1, pct=True)
        S = G_src.reindex(dates).where(base).rank(axis=1, pct=True)
        sc = 0.6 * bucket_pct(T, lm, 5) + 0.4 * S
        pem = pe.reindex(dates).where(base)
        prof = pem.gt(0)
        el = base & prof & pem.where(prof).rank(axis=1, pct=True).le(0.7)
        ind_df = pd.DataFrame([ind.values] * len(dates), index=dates, columns=u.columns)
        res = run_backtest(sc.where(el), el, vwap, bb, sb, BTConfig(top_n=30, industry_cap=5, hold_buffer=buf),
                           industry=ind_df, start="2020-01-01")
        r = {}
        for lab, (s, e) in PERIODS.items():
            m = period_metrics(res.nav, bench, s, e, res.turnover)
            r.update({f"{lab}_excess": m["超额年化"], f"{lab}_ir": m["信息比"], f"{lab}_turn": m.get("年换手(单边)")})
        return r, res.nav

    rows = []
    for fp in sorted(out.glob("gbm_a*_s*.parquet")):
        g = pd.read_parquet(fp)
        me = tm.index[(tm.index >= "2020-01-01")].intersection(g.index)
        we = tw.index[(tw.index >= "2020-01-01")].intersection(g.index)
        for freq, dates, T_src, bufs in [("monthly", me, tm, [400]), ("weekly", we, tw, [200, 400, 600])]:
            for buf in bufs:
                r, nav = run(dates, T_src, g, buf)
                row = {"model": fp.stem, "freq": freq, "buffer": buf, **r}
                rows.append(row)
                nav.to_csv(out / f"nav_{fp.stem}_{freq}_buf{buf}.csv")
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                                 ensure_ascii=False), flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "robust_eval.csv", index=False)
    agg = tab.groupby(["freq", "buffer"])[[c for c in tab.columns if c.endswith(("_excess", "_ir", "_turn"))]] \
        .agg(["median", "min", "max"]).round(3)
    print(agg.to_string())
    agg.to_csv(out / "robust_summary.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gbm", "eval"])
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--final", default="data/final")
    ap.add_argument("--tech", default="data/tech")
    ap.add_argument("--research", default="data/research")
    ap.add_argument("--panels", default="data/panels")
    ap.add_argument("--industry", default="data/industry.csv")
    ap.add_argument("--out", default="data/robust")
    a = ap.parse_args()
    {"gbm": cmd_gbm, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
