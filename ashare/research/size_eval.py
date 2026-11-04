"""「去掉小市值加分」的影响评估。

方案：
  现状  价值分 = 尾部合成（全部因子）；短线 = 含市值的 LightGBM
  A1    价值分去掉「小市值」；短线不变（只看价值分这一项改动）
  A     价值分去掉「小市值」；短线模型重新走步训练、不用市值
  B     价值分去掉「小市值」和「非流动性」；短线模型两项都不用
其余规则与最终策略一致：0.6/0.4、盈利且 PE 低于盈利公司 70 分位、行业≤5、缓冲 400、Top30、月度调仓。
同时统计入选股的市值分布（2020 年起有市值数据）。

python -m ashare.research.size_eval --feat data/features --final data/final --research data/research \
       --panels data/panels --industry data/industry.csv --out data/size
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BTConfig, period_metrics, run_backtest
from ..factors.library import PROD_FACTORS
from ..model.scoring import GBMConfig, walk_forward_gbm
from ..model.tail import label_realized_dates, weights as tail_weights
from .common import month_ends
from .tail_composite import EXTRA

PERIODS = [("2016-01-05", "2026-12-31", "2016至今"), ("2016-01-05", "2024-12-31", "2016~2024"),
           ("2025-01-01", "2026-12-31", "2025至今"), ("2020-01-01", "2022-12-31", "2020~2022"),
           ("2023-01-01", "2024-12-31", "2023~2024")]


def tail_scores(F: Path, final: Path, exclude: set[str]) -> pd.DataFrame:
    u = pd.read_parquet(F / "universe.parquet")
    me = month_ends(u.index)
    me = me[me >= pd.Timestamp("2014-06-01")][:-1]
    names = [n for n in PROD_FACTORS + EXTRA if (F / f"f_{n}.parquet").exists()]
    feats = {n: pd.read_parquet(F / f"f_{n}.parquet").loc[me].where(u.loc[me]) for n in names}
    tt = pd.read_csv(final / "tail_table.csv", index_col=0, parse_dates=True)
    realized = label_realized_dates(tt.index, u.index)
    avail = set(names) - exclude
    rows, wlog = [], {}
    for d in me:
        w = tail_weights(tt, d, 36, 0.5, available=avail, realized=realized)
        if not w:
            continue
        tot, ws = 0.0, 0.0
        for n, x in w.items():
            tot = tot + feats[n].loc[d].fillna(0.5) * x
            ws += x
        rows.append((tot / ws).where(u.loc[d]).rename(d))
        wlog[str(d.date())] = w
    return pd.DataFrame(rows), wlog


def gbm_scores(F: Path, exclude: set[str], cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_parquet(cache)
    u = pd.read_parquet(F / "universe.parquet")
    names = [n for n in PROD_FACTORS if n not in exclude]
    X3 = np.stack([pd.read_parquet(F / f"f_{n}.parquet").values.astype("float32") for n in names], axis=-1)
    y5 = pd.read_parquet(F / "y_5.parquet")
    dates = u.index
    me = month_ends(dates)
    me = me[me >= pd.Timestamp("2015-01-01")][:-1]
    pos = {d: i for i, d in enumerate(dates)}
    rows = np.array([pos[d] for d in me])
    pred, imps = walk_forward_gbm(X3, y5.values, u.values, dates, rows, 5, GBMConfig(),
                                  retrain_months=6, train_years=6, sample_every=5)
    out = pd.DataFrame(pred[rows], index=me, columns=u.columns)
    out.to_parquet(cache)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--final", default="data/final")
    ap.add_argument("--research", default="data/research")
    ap.add_argument("--panels", default="data/panels")
    ap.add_argument("--industry", default="data/industry.csv")
    ap.add_argument("--out", default="data/size")
    a = ap.parse_args()
    F, out = Path(a.feat), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(F / "universe.parquet")
    vwap = pd.read_parquet(F / "vwap.parquet")
    bb = pd.read_parquet(F / "buy_block.parquet")
    sb = pd.read_parquet(F / "sell_block.parquet")
    pe = pd.read_parquet(Path(a.panels) / "pe_ttm.parquet").reindex(columns=u.columns)
    mv = pd.read_parquet(Path(a.panels) / "total_mv.parquet").reindex(columns=u.columns)   # 万元
    bench = pd.read_csv(Path(a.research) / "bench_全A等权(股票池).csv", index_col=0, parse_dates=True).iloc[:, 0]
    ind = pd.read_csv(a.industry).set_index("code")["l1"].reindex(u.columns)
    g_now = pd.read_parquet(Path(a.research) / "score_GBM5_P.parquet")

    variants = {
        "现状": (set(), None),
        "A1 价值分去掉小市值": ({"size"}, None),
        "A 两个分数都去掉小市值": ({"size"}, {"size"}),
        "B 去掉小市值+非流动性": ({"size", "amihud_20"}, {"size", "amihud_20"}),
    }
    rows, picks_mv = [], {}
    for name, (ex_tail, ex_gbm) in variants.items():
        t, wlog = tail_scores(F, Path(a.final), ex_tail)
        g = g_now if ex_gbm is None else gbm_scores(F, ex_gbm, out / f"gbm5_ex_{'_'.join(sorted(ex_gbm))}.parquet")
        me = t.index
        base = u.reindex(me).fillna(False)
        T = t.where(base).rank(axis=1, pct=True)
        S = g.reindex(me).where(base).rank(axis=1, pct=True)
        score = 0.6 * T + 0.4 * S
        pem = pe.reindex(me).where(base)
        has = pem.notna().sum(axis=1) > 1000
        prof = pem.gt(0)
        el = base & ((prof & pem.where(prof).rank(axis=1, pct=True).le(0.7)) | ~has.values[:, None])
        ind_df = pd.DataFrame([ind.values] * len(me), index=me, columns=u.columns)
        res = run_backtest(score.where(el), el, vwap, bb, sb, BTConfig(top_n=30, industry_cap=5, hold_buffer=400),
                           industry=ind_df, start="2015-06-01")
        res.nav.to_csv(out / f"nav_{name.split()[0]}.csv")
        # 入选股市值（亿元）：每次调仓后的持仓
        mvs = []
        for dte, codes in res.holdings.items():
            prev = mv.index[mv.index < dte]
            if len(prev) and prev[-1] >= pd.Timestamp("2020-01-01"):
                x = mv.loc[prev[-1]].reindex(codes).dropna() / 1e4
                mvs.append({"date": dte, "median": float(x.median()), "lt50": float((x < 50).mean()),
                            "gt100": float((x >= 100).mean())})
        mvdf = pd.DataFrame(mvs).set_index("date")
        # 最新一个有市值数据的交易日（daily_basic 可能比价格少一天）
        last_mv = mv.index[mv.notna().sum(axis=1) > 3000][-1]
        uni = mv.loc[last_mv].where(u.loc[last_mv]) / 1e4
        uni_all = [(mv.loc[x].where(u.loc[x]) / 1e4) for x in mvdf.index.map(lambda t: mv.index[mv.index < t][-1])]
        row = {"方案": name,
               "最新一期_市值中位数(亿)": round(mvdf["median"].iloc[-1], 1),
               "最新一期_低于50亿占比": round(mvdf["lt50"].iloc[-1], 3),
               "2020起平均_市值中位数(亿)": round(mvdf["median"].mean(), 1),
               "2020起平均_低于50亿占比": round(mvdf["lt50"].mean(), 3),
               "2020起平均_100亿以上占比": round(mvdf["gt100"].mean(), 3),
               "股票池_最新_市值中位数(亿)": round(float(uni.median()), 1),
               "股票池_最新_低于50亿占比": round(float((uni.dropna() < 50).mean()), 3),
               "股票池_2020起平均_低于50亿占比": round(float(np.mean([(x.dropna() < 50).mean() for x in uni_all])), 3),
               "最新权重": wlog[max(wlog)]}
        for s, e, lab in PERIODS:
            m = period_metrics(res.nav, bench, s, e, res.turnover)
            row.update({f"{lab}_年化": m["年化收益"], f"{lab}_超额": m["超额年化"], f"{lab}_信息比": m["信息比"],
                        f"{lab}_回撤": m["最大回撤"]})
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=float), flush=True)
    pd.DataFrame(rows).to_csv(out / "size_eval.csv", index=False)


if __name__ == "__main__":
    main()
