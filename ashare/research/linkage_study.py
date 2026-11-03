"""海外 → A 股联动的量化检验。

对每条产业链里的每只 A 股、每个海外标的：
  用「美股 d 日涨跌幅」解释「A 股 d 之后第一个交易日的超额收益（相对全市场等权）」，
  分别计算 开盘跳空（open/prev_close）与 全天（close/prev_close）两种口径。
样本内 2016-2022 估计 β，样本外 2023-2026 检验预测相关性。结果写入 config/linkage_stats.json。

python -m ashare.research.linkage_study --global data/xfer/research_out/global --panels data/panels
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..macro.linkage import CHAINS, INDUSTRY_OVERSEAS
from .common import load_panels


def load_global(gdir: Path) -> dict[str, pd.Series]:
    out = {}
    for sym in {s for c in CHAINS for s in c["overseas"]} | {s for v in INDUSTRY_OVERSEAS.values() for s in v}:
        fn = sym.replace("^", "IDX_").replace("=", "_").replace(".", "_")
        fp = gdir / f"{fn}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, parse_dates=["date"])
        if len(df) < 300:
            continue
        s = df.set_index("date")["close"].astype(float)
        s = s[~s.index.duplicated()].sort_index()
        out[sym] = s.pct_change(fill_method=None)
    return out


def align_next(us_ret: pd.Series, a_dates: pd.DatetimeIndex) -> pd.Series:
    """美股 d 日收益 → 映射到 A 股 d 之后第一个交易日。"""
    pos = a_dates.searchsorted(us_ret.index, side="right")
    ok = pos < len(a_dates)
    s = pd.Series(us_ret.values[ok], index=a_dates[pos[ok]])
    return s.groupby(level=0).sum()   # 同一 A 股日对应多个美股日（A 股长假）时收益相加


def ols(x: np.ndarray, y: np.ndarray):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 120:
        return None
    x0 = x - x.mean()
    b = (x0 * (y - y.mean())).sum() / (x0 ** 2).sum()
    resid = y - y.mean() - b * x0
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / (x0 ** 2).sum())
    return {"beta": float(b), "t": float(b / se), "corr": float(np.corrcoef(x, y)[0, 1]), "n": int(len(x))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--global", dest="gdir", required=True)
    ap.add_argument("--out", default="config/linkage_stats.json")
    ap.add_argument("--split", default="2023-01-01")
    ap.add_argument("--industry", default="")
    args = ap.parse_args()
    p = load_panels("2015-06-01", basic=False)
    close, open_ = p["close"], p["open"]
    r_cc = close.pct_change(fill_method=None)
    r_gap = open_ / close.shift(1) - 1
    mkt_cc, mkt_gap = r_cc.mean(axis=1), r_gap.mean(axis=1)
    ex_cc, ex_gap = r_cc.sub(mkt_cc, axis=0), r_gap.sub(mkt_gap, axis=0)
    a_dates = close.index
    g = load_global(Path(args.gdir))
    split = pd.Timestamp(args.split)
    stats, oos_rows = {}, []
    for ch in CHAINS:
        for sym in ch["overseas"]:
            if sym not in g:
                continue
            x_all = align_next(g[sym].dropna(), a_dates).reindex(a_dates)
            for code in ch["a_shares"]:
                if code not in close.columns:
                    continue
                res = {}
                for tag, ex in [("cc", ex_cc), ("gap", ex_gap)]:
                    y = ex[code]
                    ins = (a_dates < split)
                    full = ols(x_all.values, y.values)
                    tr = ols(x_all.values[ins], y.values[ins])
                    te_mask = ~ins
                    if tr is not None:
                        xt, yt = x_all.values[te_mask], y.values[te_mask]
                        m = np.isfinite(xt) & np.isfinite(yt)
                        if m.sum() > 60:
                            pred = tr["beta"] * xt[m]
                            oos_rows.append({"chain": ch["id"], "sym": sym, "code": code, "tag": tag,
                                             "in_t": tr["t"], "oos_corr": float(np.corrcoef(pred, yt[m])[0, 1]),
                                             "oos_n": int(m.sum())})
                    res[tag] = {"full": full, "train": tr}
                stats.setdefault(ch["id"], {}).setdefault(code, {})[sym] = res
    oos = pd.DataFrame(oos_rows)
    summary = {}
    if len(oos):
        for tag in ["cc", "gap"]:
            o = oos[oos.tag == tag]
            sig = o[o.in_t.abs() > 2]
            summary[tag] = {"pairs": int(len(o)), "sig_pairs": int(len(sig)),
                            "oos_corr_all": round(float(o.oos_corr.mean()), 4),
                            "oos_corr_sig": round(float(sig.oos_corr.mean()), 4) if len(sig) else None,
                            "oos_corr_sig_pos_share": round(float((sig.oos_corr > 0).mean()), 3) if len(sig) else None}
        by_chain = oos[oos.tag == "cc"].groupby("chain")["oos_corr"].mean().sort_values(ascending=False)
        summary["by_chain_oos_corr_cc"] = {k: round(float(v), 4) for k, v in by_chain.items()}
    # ---------------- 行业层面
    ind_stats, ind_oos = {}, []
    if args.industry:
        ind = pd.read_csv(args.industry).set_index("code")["l1"].reindex(close.columns)
        for l1, syms in INDUSTRY_OVERSEAS.items():
            cols = list(ind[ind == l1].index)
            if len(cols) < 5:
                continue
            y = ex_cc[cols].mean(axis=1)
            for sym in syms:
                if sym not in g:
                    continue
                x = align_next(g[sym].dropna(), a_dates).reindex(a_dates)
                ins = a_dates < split
                full, tr = ols(x.values, y.values), ols(x.values[ins], y.values[ins])
                ind_stats.setdefault(l1, {})[sym] = {"cc": {"full": full, "train": tr}}
                if tr is not None:
                    xt, yt = x.values[~ins], y.values[~ins]
                    m = np.isfinite(xt) & np.isfinite(yt)
                    if m.sum() > 60:
                        ind_oos.append({"ind": l1, "sym": sym, "in_t": tr["t"],
                                        "oos_corr": float(np.corrcoef(tr["beta"] * xt[m], yt[m])[0, 1])})
        io = pd.DataFrame(ind_oos)
        if len(io):
            sig = io[io.in_t.abs() > 2]
            summary["industry"] = {"pairs": int(len(io)), "sig_pairs": int(len(sig)),
                                   "oos_corr_sig": round(float(sig.oos_corr.mean()), 4) if len(sig) else None,
                                   "oos_pos_share": round(float((sig.oos_corr > 0).mean()), 3) if len(sig) else None,
                                   "top": sig.sort_values("oos_corr", ascending=False).head(12)
                                   .round(3).to_dict(orient="records")}
    out = {"_说明": "β：海外标的涨1%时，A股次一交易日相对全市场等权的平均超额收益（%）。cc=收盘到收盘，gap=开盘跳空。"
                    "train 为 2016-2022 样本内估计，full 为全样本。summary 为 2023 年以后样本外检验。",
           "summary": summary, "stats": stats, "industry_stats": ind_stats}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=0, default=float))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
