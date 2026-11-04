"""30 强「每周重选」评估：与现行的每月重选（价值分市值中性化，第 6.6 节）比较。

规则与现行策略一致：价值分（尾部加权，5 组市值中性化）×0.6 + 短线分（5 日 LightGBM，不含小市值）×0.4，
盈利且 PE 低于盈利公司 70 分位，同行业 ≤ 5 只，Top30，次日均价成交，含成本与涨跌停限制。
区别只在调仓频率：每周最后一个交易日收盘后重选；持仓缓冲取 {0, 30, 100, 200, 400, 600}。
市值数据从 2020 年才有，所有方案统一从 2020-01 起选股，区间从 2020-02-04 起算。
缓冲只按 2020-02 ~ 2024-12 的信息比选；2025 年以后单独报告。

需要先运行 tech_eval（生成全市场的每周分数 score_TAIL_weekly / score_GBM5_weekly_exsize）和 neutral_eval（月度对照）。

python -m ashare.research.weekly_eval --feat data/features --tech data/tech --neutral data/neutral --research data/research \
       --panels data/panels --industry data/industry.csv --out data/weekly
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BTConfig, period_metrics, run_backtest
from .neutral_eval import bucket_pct

START = pd.Timestamp("2020-01-01")
PERIODS = {"sel": ("2020-02-04", "2024-12-31"), "hold": ("2025-01-01", "2026-12-31"), "full": ("2020-02-04", "2026-12-31"),
           "y2022": ("2020-02-04", "2022-12-31"), "y2324": ("2023-01-01", "2024-12-31")}
BUFFERS = [0, 30, 100, 200, 400, 600]
G: dict = {}


def _init(args):
    F, tech, research, panels, industry, out = map(Path, args)
    u = pd.read_parquet(F / "universe.parquet")
    G["vwap"] = pd.read_parquet(F / "vwap.parquet")
    G["bb"] = pd.read_parquet(F / "buy_block.parquet")
    G["sb"] = pd.read_parquet(F / "sell_block.parquet")
    G["bench"] = pd.read_csv(research / "bench_全A等权(股票池).csv", index_col=0, parse_dates=True).iloc[:, 0]
    t = pd.read_parquet(tech / "score_TAIL_weekly.parquet")
    g = pd.read_parquet(tech / "score_GBM5_weekly_exsize.parquet")
    we = t.index.intersection(g.index)
    we = we[we >= START]
    base = u.reindex(we).fillna(False)
    mv = pd.read_parquet(panels / "total_mv.parquet").reindex(columns=u.columns) / 1e4
    mv_we = mv.reindex(mv.index.union(we)).ffill().reindex(we).where(base)
    logmv = np.log(mv_we.where(mv_we > 0))
    T = t.reindex(we).where(base).rank(axis=1, pct=True)
    S = g.reindex(we).where(base).rank(axis=1, pct=True)
    G["score"] = 0.6 * bucket_pct(T, logmv, 5) + 0.4 * S
    pe = pd.read_parquet(panels / "pe_ttm.parquet").reindex(columns=u.columns)
    pem = pe.reindex(pe.index.union(we)).reindex(we).where(base)
    prof = pem.gt(0)
    G["el"] = base & prof & pem.where(prof).rank(axis=1, pct=True).le(0.7)
    ind = pd.read_csv(industry).set_index("code")["l1"].reindex(u.columns)
    G["ind"] = pd.DataFrame([ind.values] * len(we), index=we, columns=u.columns)
    G["mv"] = mv
    G["out"] = out


def metrics_row(tag, nav, turnover, holdings, mv):
    row = {"tag": tag}
    for lab, (s, e) in PERIODS.items():
        m = period_metrics(nav, G["bench"], s, e, turnover)
        row.update({f"{lab}_excess": m["超额年化"], f"{lab}_ir": m["信息比"], f"{lab}_ann": m["年化收益"],
                    f"{lab}_mdd": m["最大回撤"], f"{lab}_turn": m.get("年换手(单边)")})
    if holdings is not None:
        st = []
        for dte, codes in holdings.items():
            prev = mv.index[mv.index < dte]
            if len(prev):
                x = mv.loc[prev[-1]].reindex(codes).dropna()
                st.append((float((x >= 100).mean()), float((x < 50).mean())))
        if st:
            row["gt100"], row["lt50"] = float(np.mean([a for a, _ in st])), float(np.mean([b for _, b in st]))
    return row


def run_one(buf: int) -> dict:
    res = run_backtest(G["score"].where(G["el"]), G["el"], G["vwap"], G["bb"], G["sb"],
                       BTConfig(top_n=30, industry_cap=5, hold_buffer=buf), industry=G["ind"], start="2020-01-01")
    tag = f"weekly_buf{buf}"
    res.nav.to_csv(G["out"] / f"nav_{tag}.csv")
    return metrics_row(tag, res.nav, res.turnover, res.holdings, G["mv"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--tech", default="data/tech")
    ap.add_argument("--neutral", default="data/neutral")
    ap.add_argument("--research", default="data/research")
    ap.add_argument("--panels", default="data/panels")
    ap.add_argument("--industry", default="data/industry.csv")
    ap.add_argument("--out", default="data/weekly")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = (a.feat, a.tech, a.research, a.panels, a.industry, a.out)
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(args,)) as ex:
        for r in ex.map(run_one, BUFFERS):
            rows.append(r)
            print(r["tag"], f"sel {r['sel_excess']:+.3f}/{r['sel_ir']:.2f}  hold {r['hold_excess']:+.3f}/{r['hold_ir']:.2f}"
                  f"  full {r['full_excess']:+.3f}/{r['full_ir']:.2f}  turn {r['full_turn']:.1f}", flush=True)
    # 月度对照（neutral_eval 的 V-Q5，即现行方案）
    _init(args)
    nav_m = pd.read_csv(Path(a.neutral) / "nav_V-Q5.csv", index_col=0, parse_dates=True).iloc[:, 0]
    mrow = metrics_row("monthly_V-Q5", nav_m, None, None, G["mv"])
    tab = pd.DataFrame(rows + [mrow])
    tab.to_csv(out / "weekly_eval.csv", index=False)
    weekly = tab[tab["tag"].str.startswith("weekly")]
    best = weekly.sort_values("sel_ir", ascending=False).iloc[0]
    summary = {"chosen_by_sel_ir": best.to_dict(), "monthly": mrow,
               "holdout_rank_of_chosen": int((weekly["hold_excess"] > best["hold_excess"]).sum()) + 1}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
