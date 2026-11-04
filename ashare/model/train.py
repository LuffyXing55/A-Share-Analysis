"""训练最终模型并生成网站展示用的回测数据。

python -m ashare.model.train --feat data/features --research data/research --models models --site site_data

价值模型：标签 = 未来 20 个交易日收益（截面 rank-gauss）；短线模型：标签 = 未来 5 个交易日收益。
训练窗口 = 最近 6 年（与走步回测一致），只用标签已完全实现的样本。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import metrics, period_metrics
from ..factors.library import PROD_FACTORS
from .scoring import GBMConfig, make_xy, train_gbm


def train_one(feat_dir: Path, label: str, horizon: int, features: list[str], years: float = 6.0):
    feats = {n: pd.read_parquet(feat_dir / f"f_{n}.parquet") for n in features}
    u = pd.read_parquet(feat_dir / "universe.parquet")
    y = pd.read_parquet(feat_dir / f"y_{label}.parquet")
    X3 = np.stack([feats[n].values.astype("float32") for n in features], axis=-1)
    T = len(u)
    end = T - horizon - 2
    start = max(0, end - int(years * 244))
    rows = np.arange(start, end, 3)
    X, yy = make_xy(X3, y.values, u.values, rows)
    model = train_gbm(X, yy, GBMConfig())
    info = {"train_start": str(u.index[start].date()), "train_end": str(u.index[end].date()),
            "rows": int(len(yy)), "horizon": horizon}
    return model, info


def build_backtest_json(research: Path, nav_path: Path, site: Path) -> None:
    nav = pd.read_csv(nav_path, index_col=0, parse_dates=True).iloc[:, 0]
    nav_full = nav.copy()
    b1 = pd.read_csv(research / "bench_全A等权(股票池).csv", index_col=0, parse_dates=True).iloc[:, 0]
    b1_full = b1.copy()
    b2 = pd.read_csv(research / "bench_csi500等权.csv", index_col=0, parse_dates=True).iloc[:, 0]
    # 从策略第一次真正建仓的日期开始比较（尾部加权需要约 18 个月历史才能给出第一组权重，2016-02 才首次建仓）
    moved = nav[(nav - nav.iloc[0]).abs() > 1e-9]
    start = moved.index[0] - pd.Timedelta(days=1) if len(moved) else nav.index[0]
    nav = nav[nav.index >= start]
    b1, b2 = b1.reindex(nav.index).ffill(), b2.reindex(nav.index).ffill()
    nav, b1, b2 = nav / nav.iloc[0], b1 / b1.iloc[0], b2 / b2.iloc[0]
    wk = nav.resample("W-FRI").last().dropna()
    rows = []
    s0 = str(nav.index[0].date())
    for s, e, lab in [(s0, "2026-12-31", f"{s0[:7]}至今"), (s0, "2024-12-31", f"{s0[:7]}~2024"),
                      ("2025-01-01", "2026-12-31", "2025至今"), (s0, "2019-12-31", f"{s0[:7]}~2019"),
                      ("2020-01-01", "2022-12-31", "2020~2022"), ("2023-01-01", "2024-12-31", "2023~2024")]:
        mt = period_metrics(nav_full, b1_full, s, e)
        rows.append({"区间": lab, "年化收益": mt["年化收益"], "基准年化": mt["基准年化"], "超额年化": mt["超额年化"],
                     "最大回撤": mt["最大回撤"], "夏普": mt["夏普"], "信息比": mt["信息比"],
                     "月胜率": mt["月胜率(跑赢基准)"]})
    out = {"nav": {"dates": [d.strftime("%Y-%m") for d in wk.index],
                   "strategy": [round(float(x), 4) for x in wk.values],
                   "bench": [round(float(x), 4) for x in b1.reindex(wk.index).values],
                   "bench2": [round(float(x), 4) for x in b2.reindex(wk.index).values],
                   "strategy_name": "30强策略（扣费后）", "bench_name": "全A等权（股票池）", "bench2_name": "中证500成分等权"},
           "metrics": rows,
           "note": "走步回测：价值分的因子权重每月只用当时已实现的历史数据确定，短线模型每半年只用此前数据重训。"
                   "回看期、估值约束、分数权重、持仓缓冲在 2016~2024 年数据上选出（72 组中选 1 组）；"
                   "此后根据风险偏好决定不再给小市值加分（方案 A），这一决定是在看过 2025 年以后的结果后做出的，"
                   "所以「2025至今」不是严格的样本外检验。每月最后一个交易日收盘后选 30 只（同行业≤5 只，"
                   "老持仓仍在前 430 名则保留），次日按均价成交，含交易成本（买 0.15% / 卖 0.25%）、"
                   "一字涨跌停无法成交、停牌顺延。2020 年前无估值数据，只用量价因子。回测不代表未来收益。",
           "model": "尾部加权合成×0.6 + LightGBM短线×0.4（均不使用小市值因子），盈利且PE低于盈利公司70分位，行业≤5只，持仓缓冲400名", "generated": datetime.now().strftime("%Y-%m-%d")}
    site.mkdir(parents=True, exist_ok=True)
    from ..pipeline.daily import sanitize

    (site / "backtest.json").write_text(json.dumps(sanitize(out), ensure_ascii=False, allow_nan=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--research", default="data/research")
    ap.add_argument("--models", default="models")
    ap.add_argument("--site", default="site_data")
    ap.add_argument("--nav", default="data/size/nav_A.csv")
    args = ap.parse_args()
    feat_dir, mdir = Path(args.feat), Path(args.models)
    mdir.mkdir(parents=True, exist_ok=True)
    settings = json.loads((Path(__file__).resolve().parents[2] / "config" / "settings.json").read_text())
    excluded = set(settings.get("exclude_factors", []))
    features = [f for f in PROD_FACTORS if (feat_dir / f"f_{f}.parquet").exists() and f not in excluded]
    # 中期分（价值投资分）使用「尾部加权合成」（ashare/model/tail.py，无需训练）；这里只训练短线模型
    for name, label, hz in [("short", "5", 5)]:
        model, info = train_one(feat_dir, label, hz, features)
        model.save_model(str(mdir / f"{name}_model.txt"))
        imp = dict(zip(features, model.feature_importance("gain").round(1).tolist()))
        meta = {"features": features, "label": f"未来{hz}个交易日收益（截面 rank-gauss）", **info,
                "importance_gain": imp, "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
        (mdir / f"{name}_model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(name, info)
    build_backtest_json(Path(args.research), Path(args.nav), Path(args.site))


if __name__ == "__main__":
    main()
