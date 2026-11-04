"""按外部反馈做的几项检验（2026-11-04）：

1. 长周期回测：数据从 2005 年起建因子，2007-01 起选股，重点看 2008 年熊市与 2015 年股灾；同时给出夏普比率。
2. 估值约束「分行业」：周期 / 资源行业用 PB（避免 PE 陷阱），其余行业仍用 PE，与现行「全部用 PE」对比（估值数据 2020 年起才有）。
3. 选股日敏感性：月末最后一天选 vs 倒数第 2 / 第 3 天选 vs 「最后三天分数取平均」 vs 「资金分三份、最后三天各调一份」。

现行策略的其余规则保持不变：价值分（尾部加权，不含小市值；有市值数据时 5 组市值中性化）× 0.6 + 短线分（5 日 LightGBM，
不含小市值）× 0.4，盈利且 PE 低于盈利公司 70 分位，同行业 ≤ 5 只，Top30，缓冲 400，次日均价成交，含成本与一字板限制。

依赖：先用 build_features 生成 2005 年起的因子（--start 2005-01-01 --panel-start 2004-01-01）。

python -m ashare.research.feedback_eval --stage scores --feat data/features --out data/feedback
python -m ashare.research.feedback_eval --stage eval --feat data/features --panels data/panels \
       --industry data/industry.csv --out data/feedback
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BTConfig, equal_weight_index, period_metrics, run_backtest
from ..factors.library import PROD_FACTORS
from ..model import tail as TAIL
from ..model.scoring import GBMConfig, walk_forward_gbm
from .common import month_ends
from .neutral_eval import bucket_pct
from .tail_composite import EXTRA, tail_table

EXCLUDE = {"size"}
RF = 0.02            # 无风险利率（年化），用于夏普比率；2007–2026 年一年期存款 / 短端利率大致在 1.5%–4% 之间
CYCLICAL = ["有色金属", "煤炭", "钢铁", "石油石化", "基础化工", "建筑材料"]
CYCLICAL_WIDE = CYCLICAL + ["交通运输", "农林牧渔"]


def last_k_days(index: pd.DatetimeIndex, k: int = 3) -> dict[int, pd.DatetimeIndex]:
    """每个月倒数第 1..k 个交易日。返回 {0: 最后一天, 1: 倒数第二天, ...}"""
    s = pd.Series(index, index=index)
    g = s.groupby([index.year, index.month])
    return {j: pd.DatetimeIndex(g.apply(lambda x: x.iloc[-1 - j] if len(x) > j else pd.NaT).dropna().values)
            for j in range(k)}


# ---------------------------------------------------------------- 第一步：分数
def build_scores(F: Path, out: Path, gbm_only: bool = False, seed: int = 7, anchor_shift: int = 0,
                 gbm_name: str = "score_gbm") -> None:
    u = pd.read_parquet(F / "universe.parquet")
    dates = u.index
    me = month_ends(dates)
    me = me[me >= pd.Timestamp("2005-06-01")][:-1]
    lk = last_k_days(dates, 3)
    sig = sorted(set().union(*[set(v[v >= pd.Timestamp("2006-06-01")]) for v in lk.values()]))
    sig = pd.DatetimeIndex(sig)
    allp = me.union(sig)

    if gbm_only:
        _gbm(F, out, u, dates, sig, seed, anchor_shift, gbm_name)
        return
    # 尾部表（月末前 5% 的下月超额），与 final_eval 口径一致
    y = pd.read_parquet(F / "y_20.parquet")
    uu = u.loc[me]
    ym = y.loc[me].where(uu)
    ym = ym.sub(ym.mean(axis=1), axis=0)
    names = [n for n in PROD_FACTORS + EXTRA if (F / f"f_{n}.parquet").exists()]
    feats = {n: pd.read_parquet(F / f"f_{n}.parquet").reindex(allp) for n in names}
    tt = tail_table({n: f.loc[me].where(uu) for n, f in feats.items()}, ym)
    tt.to_csv(out / "tail_table.csv")
    realized = TAIL.label_realized_dates(tt.index, dates)
    avail = set(names) - EXCLUDE
    rows, wlog = [], {}
    for d in sig:
        w = TAIL.weights(tt, d, 36, 0.5, available=avail, realized=realized)
        if not w:
            continue
        tot, ws = 0.0, 0.0
        for n, x in w.items():
            tot = tot + feats[n].loc[d].fillna(0.5) * x
            ws += x
        rows.append((tot / ws).where(u.loc[d]).rename(d))
        if d in me:
            wlog[str(d.date())] = {k: round(v, 2) for k, v in sorted(w.items(), key=lambda z: -z[1])}
    pd.DataFrame(rows).to_parquet(out / "score_tail.parquet")
    (out / "tail_weights.json").write_text(json.dumps(wlog, ensure_ascii=False, indent=0))
    print("tail scores", len(rows), flush=True)
    del feats

    _gbm(F, out, u, dates, sig, seed, anchor_shift, gbm_name)
    vwap = pd.read_parquet(F / "vwap.parquet")
    bench = equal_weight_index(vwap, u, me)
    bench.to_csv(out / "bench_pool_ew.csv")
    for idx in ("csi300",):
        p = Path("data/panels") / f"member_{idx}.parquet"
        if p.exists():
            m = pd.read_parquet(p).reindex(index=u.index, columns=u.columns).fillna(False).astype(bool)
            equal_weight_index(vwap, m & u, me).to_csv(out / f"bench_{idx}_ew.csv")
    print("done", flush=True)


def _gbm(F, out, u, dates, sig, seed, anchor_shift, gbm_name) -> None:
    """短线模型：不含小市值，走步训练（每 6 个月重训，最多用 6 年数据），在每月最后 3 个交易日给出预测。
    anchor_shift：把第一次训练推后几个月（改变重训月份），用来检验结果对「模型版本」的敏感性。"""
    gn = [n for n in PROD_FACTORS if n not in EXCLUDE]
    X3 = np.stack([pd.read_parquet(F / f"f_{n}.parquet").values.astype("float32") for n in gn], axis=-1)
    y5 = pd.read_parquet(F / "y_5.parquet")
    pos = {d: i for i, d in enumerate(dates)}
    prow = np.array([pos[d] for d in sig])
    if anchor_shift:
        prow = prow[sig >= sig[0] + pd.DateOffset(months=anchor_shift)]
    pred, _ = walk_forward_gbm(X3, y5.values, u.values, dates, prow, 5, GBMConfig(seed=seed),
                               retrain_months=6, train_years=6, sample_every=5)
    pd.DataFrame(pred[prow], index=dates[prow], columns=u.columns).to_parquet(out / f"{gbm_name}.parquet")


# ---------------------------------------------------------------- 第二步：回测
G: dict = {}


def _init(args):
    F, panels, industry, out = map(Path, args)
    u = pd.read_parquet(F / "universe.parquet")
    G["u"], G["out"] = u, out
    G["vwap"] = pd.read_parquet(F / "vwap.parquet")
    G["bb"] = pd.read_parquet(F / "buy_block.parquet")
    G["sb"] = pd.read_parquet(F / "sell_block.parquet")
    G["pe"] = pd.read_parquet(panels / "pe_ttm.parquet").reindex(columns=u.columns)
    G["pb"] = pd.read_parquet(panels / "pb.parquet").reindex(columns=u.columns)
    G["mv"] = pd.read_parquet(panels / "total_mv.parquet").reindex(columns=u.columns) / 1e4
    G["tail"] = pd.read_parquet(out / "score_tail.parquet")
    G["gbms"] = {p.stem: pd.read_parquet(p) for p in sorted(out.glob("score_gbm*.parquet"))}
    G["ind"] = pd.read_csv(industry).set_index("code")["l1"].reindex(u.columns)


def _asof(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    return df.reindex(df.index.union(idx)).ffill(limit=5).reindex(idx)


def scores_on(days: pd.DatetimeIndex, neutral: bool = True, wv: float = 0.6, gbm: str = "score_gbm") -> pd.DataFrame:
    u = G["u"]
    G["gbm"] = G["gbms"][gbm]
    days = days.intersection(G["tail"].index).intersection(G["gbm"].index)
    base = u.reindex(days).fillna(False)
    T = G["tail"].reindex(days).where(base).rank(axis=1, pct=True)
    S = G["gbm"].reindex(days).where(base).rank(axis=1, pct=True)
    V = T
    if neutral:
        mv = _asof(G["mv"], days).where(base)
        logmv = np.log(mv.where(mv > 0))
        Vn = bucket_pct(T, logmv, 5)
        has = Vn.notna().sum(axis=1) > 500          # 2020 年前没有市值数据：用全池分位
        V = Vn.where(pd.DataFrame(np.broadcast_to(has.values[:, None], Vn.shape), index=Vn.index, columns=Vn.columns), T)
    return V if wv >= 1 else wv * V + (1 - wv) * S


def eligible(days: pd.DatetimeIndex, rule: str = "pe") -> pd.DataFrame:
    """估值约束。2020 年前没有估值数据：不加约束（与原回测一致）。
    pe        现行：盈利且 PE 低于全市场盈利公司 70 分位
    pe_ind    PE 在本行业盈利公司内取 70 分位
    cyc_pb    周期 / 资源行业：PB>0 且 PB 低于本行业 70 分位（不要求盈利）；其余行业同现行
    cyc_pb_w  同上，周期行业口径更宽（加交通运输、农林牧渔）
    cyc_pb_profit 周期行业：盈利 且 PB 低于本行业 70 分位；其余同现行
    none      不加估值约束"""
    u = G["u"]
    base = u.reindex(days).fillna(False)
    pe = _asof(G["pe"], days).where(base)
    pb = _asof(G["pb"], days).where(base)
    has = pe.notna().sum(axis=1) > 1000
    prof = pe.gt(0)
    if rule == "none":
        return base
    pe_ok = prof & pe.where(prof).rank(axis=1, pct=True).le(0.7)
    ind = G["ind"]
    ind_df = pd.DataFrame([ind.values] * len(days), index=days, columns=u.columns)
    if rule == "pe":
        cond = pe_ok
    else:
        def within_ind(x: pd.DataFrame) -> pd.DataFrame:
            out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
            for d in x.index:
                s = x.loc[d]
                out.loc[d] = s.groupby(ind_df.loc[d]).rank(pct=True)
            return out
        if rule == "pe_ind":
            cond = prof & within_ind(pe.where(prof)).le(0.7)
        else:
            cyc = CYCLICAL_WIDE if rule == "cyc_pb_w" else CYCLICAL
            is_c = pd.DataFrame(np.broadcast_to(ind.isin(cyc).values, base.shape), index=days, columns=u.columns)
            pbp = pb.gt(0)
            pb_ok = pbp & within_ind(pb.where(pbp & is_c)).le(0.7)
            if rule == "cyc_pb_profit":
                pb_ok = pb_ok & prof
            cond = (pe_ok & ~is_c) | (pb_ok & is_c)
    return base & (cond | ~has.values[:, None])


def bt(score: pd.DataFrame, el: pd.DataFrame, start: str) -> tuple[pd.Series, pd.Series, dict]:
    days = score.index
    ind_df = pd.DataFrame([G["ind"].values] * len(days), index=days, columns=G["u"].columns)
    res = run_backtest(score.where(el), el, G["vwap"], G["bb"], G["sb"],
                       BTConfig(top_n=30, industry_cap=5, hold_buffer=400), industry=ind_df, start=start)
    return res.nav, res.turnover, res.holdings


def job(spec: tuple) -> tuple[str, pd.Series, pd.Series, dict]:
    tag, kind, start, rule, neutral = spec[:5]
    extra = list(spec[5:])
    wv = extra[0] if extra else 0.6                    # 价值分权重（1.0 = 只用价值分，不受短线模型版本影响）
    gbm = extra[1] if len(extra) > 1 else "score_gbm"  # 短线模型版本
    lk = last_k_days(G["u"].index, 3)
    s0 = pd.Timestamp(start) - pd.Timedelta(days=40)
    lk = {j: v[v >= s0] for j, v in lk.items()}
    if kind.startswith("day"):
        j = int(kind[3:])
        d = lk[j]
        nav, tv, h = bt(scores_on(d, neutral, wv, gbm), eligible(d, rule), start)
    elif kind == "avg3":
        d0 = lk[0]
        parts = []
        for j in range(3):
            s = scores_on(lk[j], neutral, wv, gbm)
            s.index = [d0[d0 >= x][0] for x in s.index]          # 对齐到同月最后一天
            parts.append(s)
        idx = parts[0].index
        sc = sum(p.reindex(idx) for p in parts) / 3
        sc = sc.where(sum(p.reindex(idx).notna().astype(int) for p in parts) == 3)
        nav, tv, h = bt(sc, eligible(idx, rule), start)
    elif kind == "tranche3":
        navs, tvs = [], []
        for j in range(3):
            d = lk[j]
            n_, t_, _ = bt(scores_on(d, neutral, wv, gbm), eligible(d, rule), start)
            navs.append(n_)
            tvs.append(t_)
        idx = navs[0].index
        nav = sum(n.reindex(idx).ffill() / n.reindex(idx).ffill().iloc[0] for n in navs) / 3
        tv = sum(t.reindex(idx).fillna(0) for t in tvs) / 3
        h = {}
    else:
        raise ValueError(kind)
    nav.to_csv(G["out"] / f"nav_{tag}.csv")
    return tag, nav, tv, h


def sharpe(nav: pd.Series, rf: float = RF) -> float:
    r = nav.pct_change().dropna()
    ex = r - rf / 244
    return float(ex.mean() / ex.std() * np.sqrt(244)) if ex.std() > 0 else np.nan


def stats(nav: pd.Series, bench: pd.Series, s: str, e: str, tv=None) -> dict:
    m = period_metrics(nav, bench, s, e, tv)
    st, en = pd.Timestamp(s), pd.Timestamp(e)
    prior = nav.index[nav.index < st]
    begin = prior[-1] if len(prior) else nav.index[0]
    n = nav[(nav.index >= begin) & (nav.index <= en)]
    b = bench.reindex(n.index).ffill()
    return {"年化": m["年化收益"], "基准年化": m["基准年化"], "超额": m["超额年化"], "信息比": m["信息比"],
            "最大回撤": m["最大回撤"], "基准回撤": float((b / b.cummax() - 1).min()),
            "夏普": round(sharpe(n), 2), "夏普rf0": round(sharpe(n, 0.0), 2), "基准夏普": round(sharpe(b), 2),
            "区间收益": round(float(n.iloc[-1] / n.iloc[0] - 1), 4), "基准区间收益": round(float(b.iloc[-1] / b.iloc[0] - 1), 4),
            "年换手": m.get("年换手(单边)")}


LONG_PERIODS = [("2007-02-01", "2026-12-31", "2007至今"), ("2007-02-01", "2015-12-31", "2007~2015"),
                ("2016-01-01", "2026-12-31", "2016至今"), ("2020-02-04", "2026-12-31", "2020至今"),
                ("2025-01-01", "2026-12-31", "2025至今"),
                ("2007-10-17", "2008-11-04", "2008熊市(07-10-16顶~08-11-04底)"),
                ("2015-01-01", "2015-12-31", "2015全年"),
                ("2014-07-01", "2015-06-12", "2015牛市(14-07~15-06-12)"),
                ("2015-06-15", "2015-09-15", "2015股灾(06-12顶~09-15)"),
                ("2015-06-15", "2016-01-28", "股灾+熔断(06-12~16-01-28)"),
                ("2018-01-01", "2018-12-31", "2018熊市")]
SHORT_PERIODS = [("2020-02-04", "2026-12-31", "2020至今"), ("2020-02-04", "2024-12-31", "2020~2024"),
                 ("2025-01-01", "2026-12-31", "2025至今")]


def run_eval(a) -> None:
    out = Path(a.out)
    args = (a.feat, a.panels, a.industry, a.out)
    bench = pd.read_csv(out / "bench_pool_ew.csv", index_col=0, parse_dates=True).iloc[:, 0]
    gbms = [g.stem for g in sorted(out.glob("score_gbm*.parquet"))]
    specs = []
    for gi, g in enumerate(gbms):
        sfx = "" if g == "score_gbm" else "@" + g.replace("score_gbm_", "")
        # 1. 长周期（2007 起）：现行规则（2020 前没有市值 / 估值数据，自动退化为不中性化、不加估值约束）
        for kind in ("day0", "day1", "day2", "avg3", "tranche3"):
            specs.append((f"L_{kind}{sfx}", kind, "2007-01-01", "pe", True, 0.6, g))
        # 2. 估值约束分行业（2020 起，现行中性化）
        for rule in ("pe", "pe_ind", "cyc_pb", "cyc_pb_profit", "cyc_pb_w", "none"):
            specs.append((f"S_{rule}{sfx}", "day0", "2020-01-01", rule, True, 0.6, g))
        # 3. 选股日（2020 起，现行规则）
        for kind in ("day1", "day2", "avg3", "tranche3"):
            specs.append((f"S_{kind}{sfx}", kind, "2020-01-01", "pe", True, 0.6, g))
    # 只用价值分（不受短线模型版本影响）的对照
    for kind in ("day0", "day1", "day2", "avg3", "tranche3"):
        specs.append((f"L_{kind}#V", kind, "2007-01-01", "pe", True, 1.0))
    for rule in ("pe", "pe_ind", "cyc_pb", "cyc_pb_profit", "cyc_pb_w", "none"):
        specs.append((f"S_{rule}#V", "day0", "2020-01-01", rule, True, 1.0))
    for kind in ("day1", "day2", "avg3", "tranche3"):
        specs.append((f"S_{kind}#V", kind, "2020-01-01", "pe", True, 1.0))
    if a.only:
        keep = a.only.split(",")
        specs = [s for s in specs if s[0] in keep or any(s[0].startswith(k[:-1]) for k in keep if k.endswith("*"))]
    if a.skip:
        specs = [s for s in specs if not any(x in s[0] for x in a.skip.split(","))]
    res = {}
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(args,)) as ex:
        for tag, nav, tv, h in ex.map(job, specs):
            res[tag] = (nav, tv, h)
            print("done", tag, flush=True)
    rows = []
    for tag, (nav, tv, h) in res.items():
        periods = LONG_PERIODS if tag.startswith("L_") else SHORT_PERIODS
        if tag.startswith("S_day0"):
            continue
        for s, e, lab in periods:
            rows.append({"方案": tag, "区间": lab, **stats(nav, bench, s, e, tv)})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"feedback_eval{a.tag}.csv", index=False)
    print(tab.to_string())
    # 周期行业入选情况
    _init(args)
    for tag in ("S_pe", "S_cyc_pb", "S_cyc_pb_profit"):
        if tag in res:
            h = res[tag][2]
            ind = G["ind"]
            cnt = [sum(ind.get(c) in CYCLICAL for c in codes) for codes in h.values()]
            print(tag, "平均周期股只数", round(float(np.mean(cnt)), 2) if cnt else None)
    # 逐年
    for tag in [t for t in res if t.startswith("L_day0")]:
        nav = res[tag][0]
        nav = nav[nav.index >= pd.Timestamp("2007-01-31")]
        ye = nav.resample("YE").last()
        b = bench.reindex(nav.index).ffill()
        be = b.resample("YE").last()
        yr = pd.DataFrame({"策略": ye.pct_change(), "股票池等权": be.pct_change()})
        yr.iloc[0] = [ye.iloc[0] / nav.iloc[0] - 1, be.iloc[0] / b.iloc[0] - 1]
        yr["超额"] = yr["策略"] - yr["股票池等权"]
        yr.index = yr.index.year
        yr.round(4).to_csv(out / f"yearly_{tag.replace('#', '_').replace('@', '_')}.csv")
        print(tag)
        print(yr.round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["scores", "eval"], required=True)
    ap.add_argument("--feat", default="data/features")
    ap.add_argument("--panels", default="data/panels")
    ap.add_argument("--industry", default="data/industry.csv")
    ap.add_argument("--out", default="data/feedback")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="", help="跳过名字里含这些字符串的方案")
    ap.add_argument("--tag", default="")
    ap.add_argument("--gbm-only", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--anchor-shift", type=int, default=0)
    ap.add_argument("--gbm-name", default="score_gbm")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "scores":
        build_scores(Path(a.feat), out, a.gbm_only, a.seed, a.anchor_shift, a.gbm_name)
    else:
        run_eval(a)


if __name__ == "__main__":
    main()
