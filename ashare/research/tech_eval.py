"""科技股周榜（每周 15 只）的研究与评估。

- 科技股范围：申万一级 电子 / 计算机 / 通信 / 传媒（config/settings.json 的 tech.industries）；
- 调仓：每周最后一个交易日收盘后选出，次日均价成交，持有一周；
- 候选打分：短线模型（5 日 LightGBM，全市场训练、走步）、尾部合成（月度权重，按周取因子分位）、两者加权；
- 网格：权重 × 估值约束 × 持仓缓冲 × 行业上限；只用 2016-01 ~ 2024-12 选参，2025-01 起为保留期。
- 基准：科技股池等权（每周再平衡）。

python -m ashare.research.tech_eval --feat data/features --final data/final --industry data/industry.csv \
       --panels data/panels --out data/tech
"""
from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BTConfig, equal_weight_index, period_metrics, run_backtest
from ..factors.library import PROD_FACTORS
from ..model.scoring import GBMConfig, walk_forward_gbm
from ..model.tail import label_realized_dates, weights as tail_weights
from .common import month_ends

TECH = ["电子", "计算机", "通信", "传媒"]
SEL = ("2016-01-05", "2024-12-31")   # 以首次成交日 2016-01-04 为基点
HOLD = ("2025-01-01", "2026-12-31")
FULL = ("2016-01-05", "2026-12-31")
G = {}


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.values, iso.week.values]).max().sort_values().values)


EXCLUDE = {"size"}   # 与主榜一致：不再给小市值直接加分（方案 A）


def build_scores(F: Path, final: Path, out: Path, industry: Path) -> None:
    u = pd.read_parquet(F / "universe.parquet")
    dates, cols = u.index, u.columns
    we = week_ends(dates)
    we = we[(we >= pd.Timestamp("2015-06-01")) & (we < dates[-6])]
    feats = {n: pd.read_parquet(F / f"f_{n}.parquet") for n in PROD_FACTORS}
    from .tail_composite import EXTRA

    tail_feats = {**feats, **{n: pd.read_parquet(F / f"f_{n}.parquet") for n in EXTRA
                              if (F / f"f_{n}.parquet").exists()}}
    # 1) 短线模型：每周预测（全市场训练、每半年重训、训练样本与预测日之间留出持有期）
    fp = out / "score_GBM5_weekly_exsize.parquet"
    if not fp.exists():
        X3 = np.stack([feats[n].values.astype("float32") for n in PROD_FACTORS if n not in EXCLUDE], axis=-1)
        y5 = pd.read_parquet(F / "y_5.parquet")
        pos = {d: i for i, d in enumerate(dates)}
        rows = np.array([pos[d] for d in we])
        pred, _ = walk_forward_gbm(X3, y5.values, u.values, dates, rows, 5, GBMConfig(),
                                   retrain_months=6, train_years=6, sample_every=5)
        pd.DataFrame(pred[rows], index=we, columns=cols).to_parquet(fp)
        del X3
    # 2) 尾部合成：每周使用当时已实现的月度尾部表算出的权重，作用在当周的因子分位上
    tt = pd.read_csv(final / "tail_table.csv", index_col=0, parse_dates=True)
    realized = label_realized_dates(tt.index, dates)
    rows = []
    for d in we:
        w = tail_weights(tt, d, 36, 0.5, realized=realized, available=set(tail_feats) - EXCLUDE)
        if not w:
            continue
        tot, ws = 0.0, 0.0
        for n, x in w.items():
            tot = tot + tail_feats[n].loc[d].where(u.loc[d]).fillna(0.5) * x
            ws += x
        rows.append((tot / ws).where(u.loc[d]).rename(d))
    pd.DataFrame(rows).to_parquet(out / "score_TAIL_weekly.parquet")
    # 3) 科技股池与基准
    ind = pd.read_csv(industry).set_index("code")["l1"].reindex(cols)
    tech_cols = ind.isin(TECH).values
    tech_u = u & tech_cols[None, :]
    tech_u.to_parquet(out / "tech_universe.parquet")
    vwap = pd.read_parquet(F / "vwap.parquet")
    equal_weight_index(vwap, tech_u, we).to_csv(out / "bench_tech_ew.csv")
    print("scores built", len(we), "weeks; tech universe now", int(tech_u.iloc[-1].sum()), flush=True)


def _init(args):
    F, out, panels, industry = map(Path, args)
    G["tu"] = pd.read_parquet(out / "tech_universe.parquet")
    G["vwap"] = pd.read_parquet(F / "vwap.parquet")
    G["bb"] = pd.read_parquet(F / "buy_block.parquet")
    G["sb"] = pd.read_parquet(F / "sell_block.parquet")
    G["pe"] = pd.read_parquet(panels / "pe_ttm.parquet").reindex(columns=G["tu"].columns)
    G["gbm"] = pd.read_parquet(out / "score_GBM5_weekly_exsize.parquet")
    G["tail"] = pd.read_parquet(out / "score_TAIL_weekly.parquet")
    G["bench"] = pd.read_csv(out / "bench_tech_ew.csv", index_col=0, parse_dates=True).iloc[:, 0]
    G["ind"] = pd.read_csv(industry).set_index("code")["l1"].reindex(G["tu"].columns)
    G["out"] = out


def run_one(cfg):
    w, vk, buf, cap = cfg
    we = G["gbm"].index.intersection(G["tail"].index)
    base = G["tu"].reindex(we).fillna(False)
    S = G["gbm"].reindex(we).where(base).rank(axis=1, pct=True)
    T = G["tail"].reindex(we).where(base).rank(axis=1, pct=True)
    score = w * T + (1 - w) * S
    pem = G["pe"].reindex(we).where(base)
    has = pem.notna().sum(axis=1) > 300
    prof = pem.gt(0)
    pct = pem.where(prof).rank(axis=1, pct=True)
    cond = {"none": base, "profit": prof, "pct70": prof & pct.le(0.7)}[vk]
    el = base & (cond | ~has.values[:, None])
    ind_df = pd.DataFrame([G["ind"].values] * len(we), index=we, columns=base.columns)
    res = run_backtest(score.where(el), el, G["vwap"], G["bb"], G["sb"],
                       BTConfig(top_n=15, industry_cap=cap, hold_buffer=buf), industry=ind_df,
                       start="2015-12-01")
    tag = f"w{w}_{vk}_buf{buf}_cap{cap}"
    res.nav.to_csv(G["out"] / f"nav_{tag}.csv")
    row = {"tag": tag, "w_tail": w, "value": vk, "buffer": buf, "cap": cap}
    for lab, (s, e) in {"sel": SEL, "hold": HOLD, "full": FULL}.items():
        m = period_metrics(res.nav, G["bench"], s, e, res.turnover)
        row.update({f"{lab}_excess": m["超额年化"], f"{lab}_ir": m["信息比"], f"{lab}_ann": m["年化收益"],
                    f"{lab}_bench": m["基准年化"], f"{lab}_mdd": m["最大回撤"],
                    f"{lab}_win": m["月胜率(跑赢基准)"], f"{lab}_turn": m["年换手(单边)"]})
    return row


def export_site_json(out: Path, tag: str, site: Path, note: str) -> dict:
    """把选定组合的净值与分区间指标写成网站用的 backtest_tech.json（口径与主榜 build_backtest_json 一致）。"""
    from ..pipeline.daily import sanitize

    nav = pd.read_csv(out / f"nav_{tag}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    b = pd.read_csv(out / "bench_tech_ew.csv", index_col=0, parse_dates=True).iloc[:, 0]
    moved = nav[(nav - nav.iloc[0]).abs() > 1e-9]
    first = moved.index[0]                       # 首次成交日：策略与基准都以这一天收盘为基点
    s0 = str(nav.index[nav.index > first][0].date())
    rows = []
    for st_, en, lab in [(s0, "2026-12-31", f"{s0[:7]}至今"), (s0, "2024-12-31", f"{s0[:7]}~2024（选参期）"),
                         ("2025-01-01", "2026-12-31", "2025至今"), ("2020-01-01", "2022-12-31", "2020~2022"),
                         ("2023-01-01", "2024-12-31", "2023~2024")]:
        m = period_metrics(nav, b, st_, en)
        rows.append({"区间": lab, "年化收益": m["年化收益"], "基准年化": m["基准年化"], "超额年化": m["超额年化"],
                     "最大回撤": m["最大回撤"], "信息比": m["信息比"], "月胜率": m["月胜率(跑赢基准)"]})
    n = nav[nav.index >= first]
    bb = b.reindex(n.index).ffill()
    wk = n.resample("W-FRI").last().dropna()
    bw = bb.resample("W-FRI").last().reindex(wk.index)
    obj = {"metrics": rows, "note": note, "tag": tag,
           "nav": {"dates": [d.strftime("%Y-%m") for d in wk.index],
                   "strategy": [round(float(x / n.iloc[0]), 4) for x in wk.values],
                   "bench": [round(float(x / bb.iloc[0]), 4) for x in bw.values],
                   "strategy_name": "科技股周榜（扣费后）", "bench_name": "科技股池等权"}}
    site.mkdir(parents=True, exist_ok=True)
    (site / "backtest_tech.json").write_text(json.dumps(sanitize(obj), ensure_ascii=False, allow_nan=False))
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--final", default="data/final")
    ap.add_argument("--industry", default="data/industry.csv")
    ap.add_argument("--panels", default="data/panels")
    ap.add_argument("--out", default="data/tech")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--round", type=int, default=1, help="1=初始网格；2=加大持仓缓冲（降换手）的第二轮")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--export", default="", help="只导出网站用的 backtest_tech.json，例如 --export w0.5_pct70_buf100_cap6")
    ap.add_argument("--site", default="site_data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.export:
        note = ("走步回测：每周最后一个交易日收盘后在科技股池（申万一级 电子/计算机/通信/传媒）选 15 只，次日均价成交，"
                "含交易成本与涨跌停限制。参数只用 2016~2024 年选出；但「加大持仓缓冲」的第二轮网格是在看过第一轮结果后设计的，"
                "2025 至今只能算部分干净的检验。第一轮按同样规则选出的组合 2025 年以来跑输科技股池。回测不代表未来收益。")
        obj = export_site_json(out, a.export, Path(a.site), note)
        print(json.dumps(obj["metrics"], ensure_ascii=False, default=float))
        return
    if not a.skip_build:
        build_scores(Path(a.feat), Path(a.final), out, Path(a.industry))
    if a.round == 1:
        grid = list(itertools.product([0.0, 0.3, 0.5, 0.7, 1.0], ["none", "profit", "pct70"], [0, 15, 30], [None, 6]))
    else:
        # 第一轮发现：缓冲越大越好（换手从每年 31 倍降到 20 倍）。第二轮只加大缓冲，其余维度取第一轮的代表值
        grid = list(itertools.product([0.0, 0.5, 1.0], ["profit", "pct70"], [30, 60, 100], [6]))
    rows = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=((a.feat, a.out, a.panels, a.industry),)) as ex:
        for r in ex.map(run_one, grid):
            rows.append(r)
            print(r["tag"], f"sel {r['sel_excess']:+.3f}/{r['sel_ir']:.2f} hold {r['hold_excess']:+.3f}/{r['hold_ir']:.2f} "
                  f"turn {r['full_turn']:.0f}", flush=True)
    tab = pd.DataFrame(rows).sort_values("sel_ir", ascending=False)
    tab.to_csv(out / f"grid_round{a.round}.csv", index=False)
    best = tab.iloc[0].to_dict()
    hold = tab["hold_excess"]
    summary = {"n_configs": len(tab), "chosen_by_selection_ir": best,
               "holdout_excess_all": {"median": float(hold.median()), "min": float(hold.min()),
                                      "max": float(hold.max()), "share_positive": float((hold > 0).mean())},
               "chosen_holdout_rank": int((hold > best["hold_excess"]).sum()) + 1}
    (out / f"summary_round{a.round}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
