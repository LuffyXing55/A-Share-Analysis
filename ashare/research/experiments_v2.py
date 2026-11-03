"""实验 v2：针对「只做多 Top30」优化。

发现（v1）：多数因子的 alpha 集中在「排除最差的 5%」，前 5% 的超额很小；回归目标的 LightGBM
在 2020 年后 Top30 无超额，且换手过高（单边 10 倍/年）。
v2 改进：
  1) 排序目标：LambdaRank（按日期分组，强调排在最前面的股票），以及 Top20% 二分类；
  2) 新增行业动量（行业 20/60 日等权收益）与行业内相对强弱；
  3) 降换手：老持仓只要仍在股票池前 7% 就保留；同一行业最多 5 只。

python -m ashare.research.experiments_v2 --feat data/features --panels data/panels --out data/research_v2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BTConfig, metrics, run_backtest
from ..factors.library import PROD_FACTORS
from ..model.scoring import GBMConfig
from .common import load_panels, month_ends


def industry_features(panel_dir: Path, industry_csv: Path, u: pd.DataFrame) -> dict[str, pd.DataFrame]:
    p = load_panels("2013-01-01", basic=False, panel_dir=panel_dir)
    close = p["close"].reindex(columns=u.columns)
    r = close.pct_change(fill_method=None)
    ind = pd.read_csv(industry_csv).set_index("code")["l1"]
    groups = ind.reindex(close.columns)
    out = {}
    for h in (20, 60):
        sr = close / close.shift(h) - 1
        ir = pd.DataFrame(index=close.index, columns=close.columns, dtype="float32")
        grp_mean = {}
        for g, cols in groups.dropna().groupby(groups.dropna()).groups.items():
            cols = list(cols)
            m = sr[cols].where(u.reindex(index=close.index, columns=cols).fillna(False)).mean(axis=1)
            grp_mean[g] = m
            ir[cols] = np.repeat(m.values[:, None], len(cols), axis=1)
        out[f"ind_mom_{h}"] = ir
        out[f"rel_ind_{h}"] = (sr - ir).astype("float32")
    for k in list(out):
        out[k] = out[k].reindex(u.index).where(u).rank(axis=1, pct=True).astype("float32")
    return out


def relevance(y: np.ndarray, n_levels: int = 5) -> np.ndarray:
    r = pd.Series(y).rank(pct=True).values
    lev = np.floor(r * n_levels).clip(0, n_levels - 1)
    return lev.astype(int)


def walk_forward(X3, Y, U, dates, pred_rows, horizon, kind, retrain_months=6, train_years=5,
                 sample_every=5, log=print):
    import lightgbm as lgb

    T, N, F = X3.shape
    pred = np.full((T, N), np.nan, dtype="float32")
    model, last = None, None
    for t in pred_rows:
        d = dates[t]
        if last is None or (d.year * 12 + d.month) - (last.year * 12 + last.month) >= retrain_months:
            end = t - horizon - 2
            if end < 200:
                continue
            start = max(0, end - int(train_years * 244))
            xs, ys, grp = [], [], []
            for s in range(start, end, sample_every):
                m = U[s] & np.isfinite(Y[s])
                if m.sum() < 200:
                    continue
                xs.append(X3[s][m])
                yv = Y[s][m]
                if kind == "rank":
                    ys.append(relevance(yv))
                elif kind == "cls":
                    ys.append((pd.Series(yv).rank(pct=True).values >= 0.8).astype(int))
                else:
                    from ..model.scoring import _rank_gauss
                    ys.append(_rank_gauss(yv))
                grp.append(int(m.sum()))
            X, yy = np.concatenate(xs), np.concatenate(ys)
            base = dict(num_leaves=31, learning_rate=0.04, min_child_samples=300, feature_fraction=0.7,
                        bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, max_depth=6, seed=7,
                        verbose=-1, num_threads=4)
            if kind == "rank":
                params = {**base, "objective": "lambdarank", "lambdarank_truncation_level": 200,
                          "label_gain": [0, 1, 3, 7, 15], "eval_at": [30]}
                ds = lgb.Dataset(X, yy, group=grp)
                # lambdarank 在大分组上很慢：把每个日期截面随机分成 ~400 只一组
                sub_grp = []
                for g in grp:
                    k = max(1, g // 400)
                    sizes = [g // k] * k
                    sizes[-1] += g - sum(sizes)
                    sub_grp += sizes
                ds = lgb.Dataset(X, yy, group=sub_grp)
            elif kind == "cls":
                params = {**base, "objective": "binary"}
                ds = lgb.Dataset(X, yy)
            else:
                params = {**base, "objective": "regression"}
                ds = lgb.Dataset(X, yy)
            model = lgb.train(params, ds, num_boost_round=300)
            last = d
            log(f"  [{kind}] retrain @ {d.date()} rows={len(yy)}")
        if model is None:
            continue
        m = U[t]
        pred[t, m] = model.predict(X3[t][m])
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--panels", default="data/panels")
    ap.add_argument("--industry", default="data/industry.csv")
    ap.add_argument("--out", default="data/research_v2")
    ap.add_argument("--v1", default="data/research")
    args = ap.parse_args()
    feat_dir, out = Path(args.feat), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    u = pd.read_parquet(feat_dir / "universe.parquet")
    feats = {n: pd.read_parquet(feat_dir / f"f_{n}.parquet") for n in PROD_FACTORS}
    indf = industry_features(Path(args.panels), Path(args.industry), u)
    for k, v in indf.items():
        v.to_parquet(feat_dir / f"f_{k}.parquet")
    feats.update(indf)
    names = sorted(feats)
    y20 = pd.read_parquet(feat_dir / "y_20.parquet")
    y5 = pd.read_parquet(feat_dir / "y_5.parquet")
    vwap = pd.read_parquet(feat_dir / "vwap.parquet")
    bb = pd.read_parquet(feat_dir / "buy_block.parquet")
    sb = pd.read_parquet(feat_dir / "sell_block.parquet")
    dates, cols = u.index, u.columns
    me = month_ends(dates)
    me = me[me >= pd.Timestamp("2015-01-01")][:-1]
    pos = {d: i for i, d in enumerate(dates)}
    pred_rows = np.array([pos[d] for d in me])
    X3 = np.stack([feats[n].values.astype("float32") for n in names], axis=-1)
    print(f"features {len(names)} loaded {time.time()-t0:.0f}s", flush=True)
    scores = {}
    for kind, Y, hz, tag in [("rank", y20, 20, "RANK20"), ("cls", y20, 20, "CLS20"), ("reg", y5, 5, "REG5I")]:
        pr = walk_forward(X3, Y.values, u.values, dates, pred_rows, hz, kind)
        scores[tag] = pd.DataFrame(pr[pred_rows], index=me, columns=cols)
        scores[tag].to_parquet(out / f"score_{tag}.parquet")
        print(tag, f"{time.time()-t0:.0f}s", flush=True)
    v1 = Path(args.v1)
    for k in ["GBM5_P", "GBM20_P"]:
        fp = v1 / f"score_{k}.parquet"
        if fp.exists():
            scores[k] = pd.read_parquet(fp)
    R = {k: v.rank(axis=1, pct=True) for k, v in scores.items()}
    scores["RANK20+REG5I"] = 0.5 * R["RANK20"] + 0.5 * R["REG5I"]
    scores["RANK20+CLS20"] = 0.5 * R["RANK20"] + 0.5 * R["CLS20"]
    scores["ALL3"] = (R["RANK20"] + R["CLS20"] + R["REG5I"]) / 3

    ind = pd.read_csv(args.industry).set_index("code")["l1"]
    ind_df = pd.DataFrame([ind.reindex(cols).values] * len(me), index=me, columns=cols)
    elig = u.reindex(me).fillna(False)
    bench = pd.read_csv(Path(args.v1) / "bench_全A等权(股票池).csv", index_col=0, parse_dates=True).iloc[:, 0]
    periods = [("2016-01-01", "2026-11-03"), ("2016-01-01", "2019-12-31"), ("2020-01-01", "2022-12-31"),
               ("2023-01-01", "2026-11-03")]
    rows = []
    for tag, sc in scores.items():
        sc = sc.where(elig)
        for buf, cap in [(0, None), (250, 5)]:
            cfg = BTConfig(top_n=30, industry_cap=cap, hold_buffer=buf, name=tag)
            res = run_backtest(sc, elig, vwap, bb, sb, cfg, industry=ind_df, start="2015-06-01")
            key = f"{tag}|buf{buf}|cap{cap}"
            res.nav.to_csv(out / f"nav_{tag}_buf{buf}.csv")
            for s, e in periods:
                nav = res.nav[(res.nav.index >= s) & (res.nav.index <= e)]
                nav = nav / nav.iloc[0]
                tv = res.turnover[(res.turnover.index >= s) & (res.turnover.index <= e)]
                b = bench[(bench.index >= s) & (bench.index <= e)]
                m = metrics(nav, b, tv)
                rows.append({"model": key, "period": f"{s[2:4]}-{e[2:4]}", "ann": m["年化收益"],
                             "excess": m["超额年化"], "ir": m["信息比"], "mdd": m["最大回撤"],
                             "win": m["月胜率(跑赢基准)"], "turn": m["年换手(单边)"]})
            print(key, json.dumps([r for r in rows if r["model"] == key][0], ensure_ascii=False), flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "table.csv", index=False)
    print(tab.pivot(index="model", columns="period", values="excess").round(3).to_string())
    print(tab.pivot(index="model", columns="period", values="turn").round(1).to_string())
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
