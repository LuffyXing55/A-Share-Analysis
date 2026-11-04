"""海外行情：最近一个交易日的涨跌幅与异动程度（z 值），以及对入选股的联动信号。"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..live import sources as S
from .linkage import CHAINS, INDUSTRY_OVERSEAS, MACRO_SYMBOLS

DISPLAY = {  # 网站「市场概览」里展示的海外指标
    "^IXIC": "纳斯达克", "^GSPC": "标普500", "^SOX": "半导体ETF(SOXX)", "PGJ": "中概金龙ETF",
    "NVDA": "英伟达", "AAPL": "苹果", "TSLA": "特斯拉", "GC=F": "COMEX黄金", "HG=F": "COMEX铜", "CL=F": "WTI原油",
}


def fetch_overseas(symbols: list[str], days: int = 90) -> dict[str, dict]:
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    out = {}
    for sym in symbols:
        try:
            df = S.overseas_daily(sym, start)
        except Exception as e:  # noqa: BLE001
            out[sym] = {"ok": False, "error": str(e)[:120]}
            continue
        df = df.dropna(subset=["close"])
        if len(df) < 3:
            continue
        r = df["close"].pct_change()
        # 日常波动用「今天以前」的 60 天，避免今天的大波动把自己的基准抬高；
        # 此前出现过的数据错误也不计入基准（否则之后 60 天的「几倍日常波动」都会被压低）
        hist = r.iloc[:-1].dropna()
        hist = hist[[not suspicious_move(float(x)) for x in hist.values]]
        sd = hist.tail(60).std() if len(hist) >= 5 else None
        last = df.iloc[-1]
        pct = float(r.iloc[-1])
        z = float(pct / sd) if sd is not None and np.isfinite(sd) and sd > 0 else None
        suspect = suspicious_move(pct, z)
        out[sym] = {"ok": not suspect, "date": str(pd.Timestamp(last["date"]).date()), "price": float(last["close"]),
                    "pct": pct, "z": z,
                    "ret5": float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1) if len(df) > 6 else None}
        if suspect:
            # 拆股 / 合股、代码切换或数据源错误：不参与信号和海外联动打分
            out[sym]["error"] = f"数据疑似异常（单日 {pct:+.0%}），已忽略"
    return out


SPLIT_RATIOS = (2, 3, 4, 5, 8, 10, 15, 20, 1.5)


def suspicious_move(pct: float, z: float | None = None) -> bool:
    """判断单日涨跌是不是「数据错误」而不是真实行情（z 参数保留兼容，不再使用）：
    - 涨幅 ≥ 100% 或跌幅 ≥ 60%：这里跟踪的都是指数、大宗、ETF 和美股大中盘股，几乎不可能；
    - 涨跌 ≥ 30% 且前后价格之比接近拆股 / 合股比例（2、3、4、5、8、10、15、20、1.5 倍或其倒数，误差 2.5% 以内）：
      多半是数据源没有复权。
    真实的大幅波动（例如财报后单日 +25%）不会被当成错误。"""
    if pct is None or not np.isfinite(pct):
        return True
    r = 1.0 + pct
    if r <= 0 or pct >= 1.0 or pct <= -0.60:
        return True
    if abs(pct) >= 0.30:
        for k in SPLIT_RATIOS:
            for target in (k, 1.0 / k):
                if abs(r / target - 1) <= 0.025:
                    return True
    return False


def needed_symbols(pick_codes: list[str], industries: list[str] | None = None) -> list[str]:
    syms = set(DISPLAY)
    for ind in industries or []:
        syms.update(INDUSTRY_OVERSEAS.get(ind, []))
    for c in CHAINS:
        if any(code in c["a_shares"] for code in pick_codes):
            syms.update(c["overseas"])
    return sorted(syms)


def load_linkage_stats(path: str | Path = "config/linkage_stats.json") -> dict:
    """返回 {"chain": 个股-产业链统计, "industry": 行业统计}。"""
    p = Path(path)
    if not p.exists():
        return {"chain": {}, "industry": {}}
    js = json.loads(p.read_text())
    return {"chain": js.get("stats", {}), "industry": js.get("industry_stats", {})}


def chain_signals(code: str, quotes: dict, stats: dict, min_t: float = 2.0,
                  industry: str | None = None, min_date: str | None = None) -> tuple[float, list[dict]]:
    """返回（预测次日超额收益 %，链条明细）。只使用样本内 |t|>min_t 的海外标的。
    个股不在任何产业链时，用其申万一级行业的海外参照（行业层面 β，权重减半）。"""
    chain_stats = stats.get("chain", stats) if "chain" in stats else stats

    def fresh(q: dict) -> bool:
        # 海外收盘日期早于 A 股数据日期 → 已经反映在 A 股价格里，不再计入
        # 数据疑似异常（ok=False）的不参与打分
        return bool(q.get("ok")) and q.get("pct") is not None and \
            (min_date is None or str(q.get("date", "")) >= min_date)

    pred_total, details = 0.0, []
    for ch in CHAINS:
        if code not in ch["a_shares"]:
            continue
        st = chain_stats.get(ch["id"], {}).get(code, {})
        ov, preds = [], []
        beta_show, corr_show, sample = None, None, ""
        for sym in ch["overseas"]:
            q = quotes.get(sym) or {}
            bad = q.get("pct") is not None and not q.get("ok")
            item = {"symbol": sym, "name": MACRO_SYMBOLS.get(sym, DISPLAY.get(sym, sym)),
                    "pct": None if bad else q.get("pct"), "date": q.get("date"), "suspect": bad}
            ov.append(item)
            s = (st.get(sym) or {}).get("cc") or {}
            tr, full = s.get("train") or {}, s.get("full") or {}
            if full and (beta_show is None or abs(full.get("t", 0)) > abs(corr_show or 0)):
                beta_show, corr_show = full.get("beta"), full.get("corr")
                sample = f"{full.get('n', 0)} 个交易日"
            if tr and abs(tr.get("t", 0)) >= min_t and fresh(q):
                preds.append(full.get("beta", tr["beta"]) * q["pct"] * 100)
        p = float(np.mean(preds)) if preds else 0.0
        pred_total += p
        details.append({"id": ch["id"], "name": ch["name"], "logic": ch["logic"], "strength": ch["strength"],
                        "overseas": ov, "beta": None if beta_show is None else round(beta_show, 3),
                        "corr": None if corr_show is None else round(corr_show, 3), "sample": sample,
                        "pred_excess_pct": round(p, 3)})
    if not details and industry and industry in INDUSTRY_OVERSEAS:
        ist = (stats.get("industry") or {}).get(industry, {})
        ov, preds, best = [], [], None
        for sym in INDUSTRY_OVERSEAS[industry]:
            q = quotes.get(sym) or {}
            bad = q.get("pct") is not None and not q.get("ok")
            ov.append({"symbol": sym, "name": MACRO_SYMBOLS.get(sym, DISPLAY.get(sym, sym)),
                       "pct": None if bad else q.get("pct"), "date": q.get("date"), "suspect": bad})
            s_ = (ist.get(sym) or {}).get("cc") or {}
            tr, full = s_.get("train") or {}, s_.get("full") or {}
            if full and (best is None or abs(full.get("t", 0)) > abs(best.get("t", 0))):
                best = full
            if tr and abs(tr.get("t", 0)) >= min_t and fresh(q):
                preds.append(0.5 * full.get("beta", tr["beta"]) * q["pct"] * 100)
        p = float(np.mean(preds)) if preds else 0.0
        pred_total += p
        details.append({"id": "industry", "name": f"行业参照：{industry}", "strength": "行业层面",
                        "logic": "该股不在具体产业链映射中，用所属申万一级行业整体与海外标的的历史联动作参照（权重减半）。",
                        "overseas": ov, "beta": None if best is None else round(best["beta"], 3),
                        "corr": None if best is None else round(best["corr"], 3),
                        "sample": f"{best['n']} 个交易日（行业等权）" if best else "", "pred_excess_pct": round(p, 3)})
    return pred_total, details
