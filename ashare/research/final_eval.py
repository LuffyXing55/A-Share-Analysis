"""最终评估：修正后的尾部加权 + 参数网格 + 「选参期 / 保留期」分离。

为什么要这样做：网格里的参数（回看期、估值约束、分数权重、持仓缓冲）如果在整个回测区间上挑最优，
得到的数字带有选择偏差，不能叫「样本外」。这里只用 2016-01 ~ 2024-12 选参数，再单独报告 2025-01 以后的表现，
并给出全部参数组的分布。注意：网格本身（例如估值约束这一维）是在看过 2025 年以后的结果后设计的，
所以 2025 年以后只算「部分干净」的检验，见 docs/方法与回测报告.md 第 6.1 节。

python -m ashare.research.final_eval --feat data/features --research data/research \
       --industry data/industry.csv --panels data/panels --out data/final
"""
from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BTConfig, period_metrics, run_backtest
from ..factors.library import PROD_FACTORS
from ..model.tail import label_realized_dates
from .common import month_ends
from .tail_composite import EXTRA, composite, tail_table, tail_weights

# 选参期。策略在 2015-12-31 收盘后选出、2016-01-04 按均价首次成交；区间从 2016-01-05 起算，
# period_metrics 以 2016-01-04 为基点，策略与基准都从首次成交日开始比较（不多算基准当天的 −5.5%）
SEL = ("2016-01-05", "2024-12-31")
HOLD = ("2025-01-01", "2026-12-31")      # 保留期（不参与任何选择）
FULL = ("2016-01-05", "2026-12-31")
G = {}


def build_tail_scores(F: Path, out: Path, lookbacks=(24, 36, 48), min_t=0.5) -> dict[int, pd.DataFrame]:
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
    realized = label_realized_dates(me, u.index)
    res = {}
    for lb in lookbacks:
        rows, wlog = [], {}
        for d in me:
            w = tail_weights(tt, d, lb, min_t=min_t, realized=realized)
            if not w:
                continue
            rows.append(composite(feats, w, d).where(uu.loc[d]).rename(d))
            wlog[str(d.date())] = {k: round(v, 2) for k, v in sorted(w.items(), key=lambda x: -x[1])}
        sc = pd.DataFrame(rows)
        sc.to_parquet(out / f"score_TAIL{lb}.parquet")
        (out / f"tail_weights_{lb}.json").write_text(json.dumps(wlog, ensure_ascii=False, indent=0))
        res[lb] = sc
    return res


def _init(args):
    F, research, industry, panels, out = map(Path, args)
    u = pd.read_parquet(F / "universe.parquet")
    G["u"] = u
    G["vwap"] = pd.read_parquet(F / "vwap.parquet")
    G["bb"] = pd.read_parquet(F / "buy_block.parquet")
    G["sb"] = pd.read_parquet(F / "sell_block.parquet")
    G["pe"] = pd.read_parquet(panels / "pe_ttm.parquet").reindex(columns=u.columns)
    G["bench"] = pd.read_csv(research / "bench_全A等权(股票池).csv", index_col=0, parse_dates=True).iloc[:, 0]
    G["gbm"] = pd.read_parquet(research / "score_GBM5_P.parquet")
    G["tail"] = {lb: pd.read_parquet(out / f"score_TAIL{lb}.parquet") for lb in (24, 36, 48)
                 if (out / f"score_TAIL{lb}.parquet").exists()}
    ind = pd.read_csv(industry).set_index("code")["l1"]
    G["ind"] = ind.reindex(u.columns)
    G["out"] = out


def value_mask(kind: str, me) -> pd.DataFrame:
    u = G["u"]
    base = u.reindex(me).fillna(False)
    pem = G["pe"].reindex(me).where(base)
    has = pem.notna().sum(axis=1) > 1000          # 2020 年前没有估值数据，约束不生效
    prof = pem.gt(0)
    pct = pem.where(prof).rank(axis=1, pct=True)
    cond = {"profit": prof, "pct80": prof & pct.le(0.8), "pct70": prof & pct.le(0.7),
            "pe60": prof & pem.le(60)}[kind]
    return base & (cond | ~has.values[:, None])


def run_one(cfg: tuple) -> dict:
    lb, vk, wv, buf = cfg
    t = G["tail"][lb]
    me = t.index
    base = G["u"].reindex(me).fillna(False)
    # 与实盘一致：两个分数都只在股票池内取分位，再按权重合成
    T = t.where(base).rank(axis=1, pct=True)
    S = G["gbm"].reindex(me).where(base).rank(axis=1, pct=True)
    score = wv * T + (1 - wv) * S
    el = value_mask(vk, me)
    ind_df = pd.DataFrame([G["ind"].values] * len(me), index=me, columns=G["u"].columns)
    res = run_backtest(score.where(el), el, G["vwap"], G["bb"], G["sb"],
                       BTConfig(top_n=30, industry_cap=5, hold_buffer=buf), industry=ind_df, start="2015-06-01")
    tag = f"lb{lb}_{vk}_w{wv}_buf{buf}"
    res.nav.to_csv(G["out"] / f"nav_{tag}.csv")
    row = {"tag": tag, "lookback": lb, "value": vk, "w_value": wv, "buffer": buf}
    for lab, (s, e) in {"sel": SEL, "hold": HOLD, "full": FULL}.items():
        m = period_metrics(res.nav, G["bench"], s, e, res.turnover)
        row.update({f"{lab}_excess": m["超额年化"], f"{lab}_ir": m["信息比"], f"{lab}_ann": m["年化收益"],
                    f"{lab}_mdd": m["最大回撤"], f"{lab}_win": m["月胜率(跑赢基准)"], f"{lab}_turn": m["年换手(单边)"]})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--research", default="data/research")
    ap.add_argument("--industry", default="data/industry.csv")
    ap.add_argument("--panels", default="data/panels")
    ap.add_argument("--out", default="data/final")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-tail", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.skip_tail:
        build_tail_scores(Path(a.feat), out)
        print("tail scores rebuilt", flush=True)
    grid = list(itertools.product([24, 36, 48], ["profit", "pct80", "pct70", "pe60"], [0.6, 0.7, 0.8], [250, 400]))
    init_args = (a.feat, a.research, a.industry, a.panels, a.out)
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(init_args,)) as ex:
        for r in ex.map(run_one, grid):
            rows.append(r)
            print(r["tag"], f"sel {r['sel_excess']:+.3f}/{r['sel_ir']:.2f}  hold {r['hold_excess']:+.3f}/{r['hold_ir']:.2f}",
                  flush=True)
    tab = pd.DataFrame(rows).sort_values("sel_ir", ascending=False)
    tab.to_csv(out / "grid.csv", index=False)
    best = tab.iloc[0].to_dict()
    hold = tab["hold_excess"]
    summary = {
        "selection_period": SEL, "holdout_period": HOLD, "n_configs": len(tab),
        "chosen_by_selection_ir": best,
        "holdout_excess_all_configs": {"median": float(hold.median()), "min": float(hold.min()),
                                       "max": float(hold.max()), "share_positive": float((hold > 0).mean())},
        "chosen_holdout_rank": int((hold > best["hold_excess"]).sum()) + 1,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
