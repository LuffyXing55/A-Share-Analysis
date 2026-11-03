"""模型对比实验：等权合成 / ICIR 加权 / LightGBM 走步，月度 Top30 回测。

python -m ashare.research.experiments --feat data/features --panels data/panels --out data/research
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BTConfig, equal_weight_index, metrics, run_backtest
from ..factors.library import PROD_FACTORS, SHORT_FACTORS, VALUE_FACTORS
from ..model.scoring import GBMConfig, linear_score, rolling_icir_weights, walk_forward_gbm
from .common import month_ends

ALL = sorted(set(VALUE_FACTORS) | set(SHORT_FACTORS) | {"vol_20", "dist_ma20", "high52",
                                                         "abn_turn", "pv_corr_20", "intra_20", "ovn_20"})
EW_SET = ["ivol_20", "limit_up_5", "maxret_20", "turn_20", "abn_turn", "rev_20", "dist_ma20",
          "pv_corr_20", "intra_20", "vol_60", "bp", "sp", "ep", "dy", "amihud_20"]


def load(feat_dir: Path, names):
    feats = {}
    for n in names:
        fp = feat_dir / f"f_{n}.parquet"
        if fp.exists():
            feats[n] = pd.read_parquet(fp)
    return feats


def ic_table(feats, y, u, dates):
    rows = {}
    yv = y.reindex(dates)
    uv = u.reindex(dates)
    for d in dates:
        m0 = uv.loc[d].values & np.isfinite(yv.loc[d].values)
        yr = pd.Series(yv.loc[d].values[m0]).rank().values
        row = {}
        for n, f in feats.items():
            x = f.loc[d].values[m0]
            mm = np.isfinite(x)
            if mm.sum() < 100:
                continue
            row[n] = np.corrcoef(pd.Series(x[mm]).rank().values, pd.Series(yr[mm]).rank().values)[0, 1]
        rows[d] = row
    return pd.DataFrame(rows).T


def run_all(args):
    feat_dir, out = Path(args.feat), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    feats = load(feat_dir, ALL)
    names = sorted(feats)
    u = pd.read_parquet(feat_dir / "universe.parquet")
    y20 = pd.read_parquet(feat_dir / "y_20.parquet")
    y5 = pd.read_parquet(feat_dir / "y_5.parquet")
    vwap = pd.read_parquet(feat_dir / "vwap.parquet")
    bb = pd.read_parquet(feat_dir / "buy_block.parquet")
    sb = pd.read_parquet(feat_dir / "sell_block.parquet")
    dates = u.index
    cols = u.columns
    me = month_ends(dates)
    me = me[me >= pd.Timestamp("2015-01-01")][:-1]
    print(f"loaded {len(names)} features in {time.time()-t0:.0f}s; {len(me)} rebalances")

    # ---------- 等权合成
    scores = {}
    w_ew = {n: 1.0 for n in EW_SET if n in feats}
    scores["EW15"] = linear_score(feats, w_ew, me)

    # ---------- ICIR 加权（滚动 24 个月）
    ict = ic_table(feats, y20, u, me)
    ict.to_csv(out / "monthly_ic_table.csv")
    rows = []
    for d in me:
        w = rolling_icir_weights(ict, d, lookback=24, min_obs=10)
        if not w:
            rows.append(pd.Series(np.nan, index=cols, name=d))
            continue
        rows.append(linear_score(feats, w, [d]).iloc[0].rename(d))
    scores["ICIR24"] = pd.DataFrame(rows)

    # ---------- LightGBM 走步（研究全集 vs 实盘可得因子集）
    U = u.values
    pos = {d: i for i, d in enumerate(dates)}
    pred_rows = np.array([pos[d] for d in me])
    prod = [n for n in PROD_FACTORS if n in feats]
    for set_tag, fset in [("", names), ("_P", prod)]:
        X3 = np.stack([feats[n].values.astype("float32") for n in fset], axis=-1)
        for tag, Y, hz in [("GBM20", y20, 20), ("GBM5", y5, 5)]:
            if set_tag == "" and tag == "GBM5":
                continue
            pred, imps = walk_forward_gbm(X3, Y.values, U, dates, pred_rows, hz, GBMConfig(),
                                          retrain_months=6, train_years=6, sample_every=5)
            scores[tag + set_tag] = pd.DataFrame(pred[pred_rows], index=me, columns=cols)
            if imps:
                imp = pd.DataFrame([x[1] for x in imps], index=[x[0] for x in imps], columns=fset)
                imp.to_csv(out / f"importance_{tag}{set_tag}.csv")
        del X3
    g20 = scores["GBM20_P"].rank(axis=1, pct=True)
    g5 = scores["GBM5_P"].rank(axis=1, pct=True)
    scores["GBM_blend_P"] = 0.6 * g20 + 0.4 * g5
    scores["GBM_ICIR_P"] = 0.5 * g20 + 0.5 * scores["ICIR24"].rank(axis=1, pct=True)
    for k, v in scores.items():
        v.to_parquet(out / f"score_{k}.parquet")

    # ---------- 回测
    elig = u.reindex(me).fillna(False)
    panel_dir = Path(args.panels)
    benches = {"全A等权(股票池)": equal_weight_index(vwap, u, me)}
    for idx in ["csi500", "csi1000", "csi300"]:
        m = pd.read_parquet(panel_dir / f"member_{idx}.parquet").reindex(index=dates, columns=cols).fillna(False)
        benches[f"{idx}等权"] = equal_weight_index(vwap, m, me)
    results = {}
    periods = [("2016-01-01", "2026-11-03"), ("2016-01-01", "2019-12-31"),
               ("2020-01-01", "2026-11-03"), ("2023-01-01", "2026-11-03")]
    for tag, sc in scores.items():
        sc = sc.where(elig)
        for cap_buf in [0, 15]:
            cfg = BTConfig(top_n=30, industry_cap=None, hold_buffer=cap_buf, name=tag)
            res = run_backtest(sc, elig, vwap, bb, sb, cfg, start="2015-06-01")
            key = f"{tag}{'_buf' if cap_buf else ''}"
            results[key] = {}
            for s, e in periods:
                nav = res.nav[(res.nav.index >= s) & (res.nav.index <= e)]
                nav = nav / nav.iloc[0]
                tv = res.turnover[(res.turnover.index >= s) & (res.turnover.index <= e)]
                results[key][f"{s[:4]}-{e[:4]}"] = {
                    b: metrics(nav, benches[b][(benches[b].index >= s) & (benches[b].index <= e)], tv)
                    for b in ["全A等权(股票池)", "csi500等权"]}
            res.nav.to_csv(out / f"nav_{key}.csv")
            print(key, json.dumps(results[key]["2016-2026"]["全A等权(股票池)"], ensure_ascii=False))
    for b, s in benches.items():
        s.to_csv(out / f"bench_{b}.csv")
    (out / "experiments.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    # 汇总表
    rows = []
    for k, v in results.items():
        for per, d in v.items():
            m = d["全A等权(股票池)"]
            rows.append({"model": k, "period": per, "ann": m["年化收益"], "excess": m["超额年化"],
                         "ir": m["信息比"], "mdd": m["最大回撤"], "ex_mdd": m["超额最大回撤"],
                         "win": m["月胜率(跑赢基准)"], "turn": m["年换手(单边)"],
                         "ex_csi500": d["csi500等权"]["超额年化"]})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "experiments_table.csv", index=False)
    print(tab.pivot(index="model", columns="period", values=["excess", "ir"]).round(3).to_string())
    print(f"total {time.time()-t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--panels", default="data/panels")
    ap.add_argument("--out", default="data/research")
    run_all(ap.parse_args())


if __name__ == "__main__":
    main()
