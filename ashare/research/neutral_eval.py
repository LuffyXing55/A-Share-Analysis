"""「市值中性化」评估：在方案 A 的基础上，让大盘股和小盘股在同一起跑线上比较。

为什么要做：方案 A 已经去掉了「小市值」因子，但「非流动性」(amihud_20) 与对数市值的截面相关约 −0.82，
本质上仍在给小盘股加分，结果入选股常常没有一只市值超过 100 亿。

方案（其余规则与最终策略一致：0.6/0.4、盈利且 PE 低于盈利公司 70 分位、行业≤5、缓冲 400、Top30、月度调仓）：
  A        现状（方案 A）
  Q5       综合分在 5 个市值分组内各自取分位（每组约 1/5 的股票）
  Q10      综合分在 10 个市值分组内各自取分位
  LIN      综合分对对数市值做截面回归，取残差
  V-Q5     只对价值分做 5 组中性化，短线分不变
2020 年以前没有市值数据，所有方案统一从 2020-01 月末（1-23）开始选股、2-03 首次成交，便于比较。

python -m ashare.research.neutral_eval --feat data/features --final data/final --research data/research \
       --panels data/panels --industry data/industry.csv --size data/size --out data/neutral
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BTConfig, period_metrics, run_backtest
from .size_eval import tail_scores

START_ME = pd.Timestamp("2020-01-01")   # 2020 年 1 月最后一个交易日是 1-23（春节），2-03 首次成交
PERIODS = [("2020-02-04", "2026-12-31", "2020至今"), ("2020-02-04", "2024-12-31", "2020~2024"),
           ("2025-01-01", "2026-12-31", "2025至今"), ("2020-02-04", "2022-12-31", "2020~2022"),
           ("2023-01-01", "2024-12-31", "2023~2024")]


def bucket_pct(score: pd.DataFrame, logmv: pd.DataFrame, k: int) -> pd.DataFrame:
    """每个截面按市值分成 k 组，综合分在组内取分位。没有市值的股票不参与。"""
    out = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    for d in score.index:
        s, m = score.loc[d], logmv.loc[d]
        ok = s.notna() & m.notna()
        if ok.sum() < k * 20:
            continue
        g = pd.qcut(m[ok].rank(method="first"), k, labels=False)
        out.loc[d, ok[ok].index] = s[ok].groupby(g).rank(pct=True).values
    return out


def linear_resid(score: pd.DataFrame, logmv: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=score.index, columns=score.columns)
    for d in score.index:
        s, m = score.loc[d], logmv.loc[d]
        ok = s.notna() & m.notna()
        if ok.sum() < 100:
            continue
        x = m[ok] - m[ok].mean()
        beta = float((x * (s[ok] - s[ok].mean())).sum() / (x * x).sum())
        out.loc[d, ok[ok].index] = (s[ok] - beta * x).rank(pct=True).values
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--final", default="data/final")
    ap.add_argument("--research", default="data/research")
    ap.add_argument("--panels", default="data/panels")
    ap.add_argument("--industry", default="data/industry.csv")
    ap.add_argument("--size", default="data/size", help="size_eval 的输出目录（复用不含市值的短线模型走步预测）")
    ap.add_argument("--out", default="data/neutral")
    a = ap.parse_args()
    F, out = Path(a.feat), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(F / "universe.parquet")
    vwap = pd.read_parquet(F / "vwap.parquet")
    bb = pd.read_parquet(F / "buy_block.parquet")
    sb = pd.read_parquet(F / "sell_block.parquet")
    pe = pd.read_parquet(Path(a.panels) / "pe_ttm.parquet").reindex(columns=u.columns)
    mv = pd.read_parquet(Path(a.panels) / "total_mv.parquet").reindex(columns=u.columns) / 1e4   # 亿元
    bench = pd.read_csv(Path(a.research) / "bench_全A等权(股票池).csv", index_col=0, parse_dates=True).iloc[:, 0]
    ind = pd.read_csv(a.industry).set_index("code")["l1"].reindex(u.columns)
    g = pd.read_parquet(Path(a.size) / "gbm5_ex_size.parquet")

    t, _ = tail_scores(F, Path(a.final), {"size"})
    me = t.index[t.index >= START_ME]
    base = u.reindex(me).fillna(False)
    # 信号日收盘时已知的市值（daily_basic 与价格同日）
    mv_me = mv.reindex(mv.index.union(me)).ffill().reindex(me).where(base)
    logmv = np.log(mv_me.where(mv_me > 0))
    T = t.reindex(me).where(base).rank(axis=1, pct=True)
    S = g.reindex(me).where(base).rank(axis=1, pct=True)
    raw = 0.6 * T + 0.4 * S
    pem = pe.reindex(me).where(base)
    prof = pem.gt(0)
    el = base & prof & pem.where(prof).rank(axis=1, pct=True).le(0.7)
    ind_df = pd.DataFrame([ind.values] * len(me), index=me, columns=u.columns)

    variants = {
        "A 现状": raw,
        "Q5 五组中性化": bucket_pct(raw, logmv, 5),
        "Q10 十组中性化": bucket_pct(raw, logmv, 10),
        "LIN 回归残差": linear_resid(raw, logmv),
        "V-Q5 只中性化价值分": 0.6 * bucket_pct(T, logmv, 5) + 0.4 * S,
    }
    rows = []
    for name, score in variants.items():
        res = run_backtest(score.where(el), el, vwap, bb, sb, BTConfig(top_n=30, industry_cap=5, hold_buffer=400),
                           industry=ind_df, start="2020-01-01")
        tag = name.split()[0]
        res.nav.to_csv(out / f"nav_{tag}.csv")
        stats = []
        for dte, codes in res.holdings.items():
            prev = mv.index[mv.index < dte]
            if len(prev):
                x = mv.loc[prev[-1]].reindex(codes).dropna()
                stats.append({"date": dte, "median": float(x.median()), "lt50": float((x < 50).mean()),
                              "gt100": float((x >= 100).mean()), "n_gt100": int((x >= 100).sum())})
        st = pd.DataFrame(stats).set_index("date")
        row = {"方案": name,
               "最新_市值中位数(亿)": round(st["median"].iloc[-1], 1), "最新_百亿以上只数": int(st["n_gt100"].iloc[-1]),
               "平均_市值中位数(亿)": round(st["median"].mean(), 1), "平均_低于50亿占比": round(st["lt50"].mean(), 3),
               "平均_百亿以上占比": round(st["gt100"].mean(), 3),
               "百亿以上为0的月份占比": round(float((st["n_gt100"] == 0).mean()), 3)}
        for s, e, lab in PERIODS:
            m = period_metrics(res.nav, bench, s, e, res.turnover)
            row.update({f"{lab}_年化": m["年化收益"], f"{lab}_超额": m["超额年化"], f"{lab}_信息比": m["信息比"],
                        f"{lab}_回撤": m["最大回撤"]})
        row["年换手"] = period_metrics(res.nav, bench, *PERIODS[0][:2], res.turnover).get("年换手(单边)")
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, default=float), flush=True)
    pool = mv_me.where(base)
    print("股票池：百亿以上占比（平均）", round(float((pool >= 100).sum(axis=1).div(pool.notna().sum(axis=1)).mean()), 3))
    pd.DataFrame(rows).to_csv(out / "neutral_eval.csv", index=False)


if __name__ == "__main__":
    main()
