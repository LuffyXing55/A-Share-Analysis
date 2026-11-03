"""海外行情：最近一个交易日的涨跌幅与异动程度（z 值），以及对入选股的联动信号。"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..live import sources as S
from .linkage import CHAINS, MACRO_SYMBOLS

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
        sd = r.tail(60).std()
        last = df.iloc[-1]
        out[sym] = {"ok": True, "date": str(pd.Timestamp(last["date"]).date()), "price": float(last["close"]),
                    "pct": float(r.iloc[-1]), "z": float(r.iloc[-1] / sd) if sd and sd > 0 else None,
                    "ret5": float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1) if len(df) > 6 else None}
    return out


def needed_symbols(pick_codes: list[str]) -> list[str]:
    syms = set(DISPLAY)
    for c in CHAINS:
        if any(code in c["a_shares"] for code in pick_codes):
            syms.update(c["overseas"])
    return sorted(syms)


def load_linkage_stats(path: str | Path = "config/linkage_stats.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("stats", {})


def chain_signals(code: str, quotes: dict, stats: dict, min_t: float = 2.0) -> tuple[float, list[dict]]:
    """返回（预测次日超额收益 %，链条明细）。只使用样本内 |t|>min_t 的海外标的。"""
    pred_total, details = 0.0, []
    for ch in CHAINS:
        if code not in ch["a_shares"]:
            continue
        st = stats.get(ch["id"], {}).get(code, {})
        ov, preds = [], []
        beta_show, corr_show, sample = None, None, ""
        for sym in ch["overseas"]:
            q = quotes.get(sym) or {}
            item = {"symbol": sym, "name": MACRO_SYMBOLS.get(sym, DISPLAY.get(sym, sym)),
                    "pct": q.get("pct"), "date": q.get("date")}
            ov.append(item)
            s = (st.get(sym) or {}).get("cc") or {}
            tr, full = s.get("train") or {}, s.get("full") or {}
            if full and (beta_show is None or abs(full.get("t", 0)) > abs(corr_show or 0)):
                beta_show, corr_show = full.get("beta"), full.get("corr")
                sample = f"{full.get('n', 0)} 个交易日"
            if tr and abs(tr.get("t", 0)) >= min_t and q.get("pct") is not None:
                preds.append(full.get("beta", tr["beta"]) * q["pct"] * 100)
        p = float(np.mean(preds)) if preds else 0.0
        pred_total += p
        details.append({"id": ch["id"], "name": ch["name"], "logic": ch["logic"], "strength": ch["strength"],
                        "overseas": ov, "beta": None if beta_show is None else round(beta_show, 3),
                        "corr": None if corr_show is None else round(corr_show, 3), "sample": sample,
                        "pred_excess_pct": round(p, 3)})
    return pred_total, details
