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

from ..backtest.engine import metrics
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


def build_backtest_json(research: Path, tag: str, site: Path) -> None:
    nav = pd.read_csv(research / f"nav_{tag}.csv", index_col=0, parse_dates=True).iloc[:, 0]
    b1 = pd.read_csv(research / "bench_全A等权(股票池).csv", index_col=0, parse_dates=True).iloc[:, 0]
    b2 = pd.read_csv(research / "bench_csi500等权.csv", index_col=0, parse_dates=True).iloc[:, 0]
    start = pd.Timestamp("2016-01-01")
    nav, b1, b2 = nav[nav.index >= start], b1.reindex(nav.index).ffill(), b2.reindex(nav.index).ffill()
    nav, b1, b2 = nav / nav.iloc[0], b1 / b1.iloc[0], b2 / b2.iloc[0]
    wk = nav.resample("W-FRI").last().dropna()
    rows = []
    for s, e, lab in [("2016-01-01", "2026-12-31", "2016至今"), ("2016-01-01", "2019-12-31", "2016-2019"),
                      ("2020-01-01", "2026-12-31", "2020至今"), ("2023-01-01", "2026-12-31", "2023至今"),
                      ("2025-01-01", "2026-12-31", "2025至今")]:
        m = (nav.index >= s) & (nav.index <= e)
        n_, bb = nav[m] / nav[m].iloc[0], b1[m] / b1[m].iloc[0]
        mt = metrics(n_, bb)
        rows.append({"区间": lab, "年化收益": mt["年化收益"], "基准年化": mt["基准年化"], "超额年化": mt["超额年化"],
                     "最大回撤": mt["最大回撤"], "夏普": mt["夏普"], "信息比": mt["信息比"],
                     "月胜率": mt["月胜率(跑赢基准)"]})
    out = {"nav": {"dates": [d.strftime("%Y-%m") for d in wk.index],
                   "strategy": [round(float(x), 4) for x in wk.values],
                   "bench": [round(float(x), 4) for x in b1.reindex(wk.index).values],
                   "bench2": [round(float(x), 4) for x in b2.reindex(wk.index).values],
                   "strategy_name": "30强策略（扣费后）", "bench_name": "全A等权（股票池）", "bench2_name": "中证500成分等权"},
           "metrics": rows,
           "note": "走步回测（样本外）：每半年用此前 6 年数据重训模型，每月最后一个交易日收盘后选 30 只、次日按均价成交，"
                   "含交易成本（买 0.15% / 卖 0.25%）、一字涨跌停无法成交、停牌顺延。2020 年前无估值数据，只用量价因子。"
                   "回测不代表未来收益。",
           "model": tag, "generated": datetime.now().strftime("%Y-%m-%d")}
    site.mkdir(parents=True, exist_ok=True)
    (site / "backtest.json").write_text(json.dumps(out, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--research", default="data/research")
    ap.add_argument("--models", default="models")
    ap.add_argument("--site", default="site_data")
    ap.add_argument("--bt-tag", default="GBM_blend_P_buf")
    args = ap.parse_args()
    feat_dir, mdir = Path(args.feat), Path(args.models)
    mdir.mkdir(parents=True, exist_ok=True)
    features = [f for f in PROD_FACTORS if (feat_dir / f"f_{f}.parquet").exists()]
    for name, label, hz in [("value", "20", 20), ("short", "5", 5)]:
        model, info = train_one(feat_dir, label, hz, features)
        model.save_model(str(mdir / f"{name}_model.txt"))
        imp = dict(zip(features, model.feature_importance("gain").round(1).tolist()))
        meta = {"features": features, "label": f"未来{hz}个交易日收益（截面 rank-gauss）", **info,
                "importance_gain": imp, "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
        (mdir / f"{name}_model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(name, info)
    build_backtest_json(Path(args.research), args.bt_tag, Path(args.site))


if __name__ == "__main__":
    main()
