"""「近期要观察的事」：事件日历 + 实时信号（海外异动、利率、两融、情绪），并标出受影响的入选股。"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .linkage import CHAINS, MACRO_SYMBOLS
from .overseas import DISPLAY


def load_events(path: str | Path = "config/events.json") -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {"events": [], "themes": []}


def _affected(picks: list[dict], industries: list[str], chains: list[str] | None = None) -> list[str]:
    inds = set(industries or [])
    out = []
    for p in picks:
        if p.get("industry") in inds:
            out.append(p["code"])
        elif chains and any(c.get("id") in chains for c in p.get("chains", [])):
            out.append(p["code"])
    return out


def upcoming_events(cfg: dict, picks: list[dict], today: date, days: int = 21) -> list[dict]:
    out = []
    for e in cfg.get("events", []):
        d0 = date.fromisoformat(e["date"])
        d1 = date.fromisoformat(e.get("end", e["date"]))
        if d1 < today - timedelta(days=1) or d0 > today + timedelta(days=days):
            continue
        item = dict(e)
        item["picks"] = _affected(picks, e.get("industries", []), e.get("chains"))
        item["days_to"] = (d0 - today).days
        out.append(item)
    out.sort(key=lambda x: (x["date"], -x.get("importance", 1)))
    return out


def themes(cfg: dict, picks: list[dict]) -> list[dict]:
    return [{**t, "picks": _affected(picks, t.get("industries", []))} for t in cfg.get("themes", [])]


def live_signals(quotes: dict, picks: list[dict], rates: pd.DataFrame | None = None,
                 margin: pd.DataFrame | None = None, breadth: dict | None = None) -> list[dict]:
    sig = []
    # 1) 海外异动：|z|>=1.5 或 |涨跌|>=3%
    for sym, q in quotes.items():
        if not q.get("ok") or q.get("pct") is None:
            continue
        z = q.get("z") or 0
        if abs(z) < 1.5 and abs(q["pct"]) < 0.03:
            continue
        name = MACRO_SYMBOLS.get(sym, DISPLAY.get(sym, sym))
        affected, chain_names = [], []
        for ch in CHAINS:
            if sym in ch["overseas"]:
                chain_names.append(ch["name"])
                affected += [p["code"] for p in picks if p["code"] in ch["a_shares"]]
        if not chain_names and sym not in DISPLAY:
            continue
        direction = "大涨" if q["pct"] > 0 else "大跌"
        text = f"{q['date']} 收盘 {q['pct']*100:+.2f}%（约 {abs(z):.1f} 倍日常波动）。"
        if chain_names:
            text += "关联 A 股产业链：" + "、".join(dict.fromkeys(chain_names)) + "。"
            text += "历史统计显示这类异动对相关 A 股次日开盘影响最明显（样本外相关 0.24），对全天超额影响较弱。"
        sig.append({"title": f"海外异动：{name}{direction}", "text": text, "picks": sorted(set(affected)),
                    "level": 3 if abs(z) >= 2.5 else 2, "when": f"美股 {q['date']}"})
    # 2) 美债收益率
    if rates is not None and len(rates) > 6 and "us10y" in rates:
        r = rates.dropna(subset=["us10y"])
        chg1 = (r["us10y"].iloc[-1] - r["us10y"].iloc[-2]) * 100
        chg5 = (r["us10y"].iloc[-1] - r["us10y"].iloc[-6]) * 100
        if abs(chg1) >= 7 or abs(chg5) >= 15:
            growth = [p["code"] for p in picks if p.get("industry") in {"电子", "计算机", "通信", "传媒", "医药生物"}]
            sig.append({"title": f"美债10年期收益率{'上行' if chg1 > 0 else '下行'}",
                        "text": f"最新 {r['us10y'].iloc[-1]:.2f}%，1 日 {chg1:+.0f}bp，5 日 {chg5:+.0f}bp。"
                                "收益率上行通常压制成长股估值、利好银行保险；下行则相反。",
                        "picks": growth, "level": 2, "when": str(pd.Timestamp(r["date"].iloc[-1]).date())})
    # 3) 两融
    if margin is not None and len(margin) > 6 and "fin_balance" in margin:
        m = margin.dropna(subset=["fin_balance"])
        c5 = m["fin_balance"].iloc[-1] / m["fin_balance"].iloc[-6] - 1
        if abs(c5) >= 0.01:
            sig.append({"title": f"融资余额 5 日{'增加' if c5 > 0 else '减少'} {abs(c5)*100:.1f}%",
                        "text": f"最新融资余额 {m['fin_balance'].iloc[-1]/1e4:.2f} 万亿元。融资盘持续流出时，高融资占比的科技成长板块承压；止跌回升往往伴随超跌反弹。",
                        "picks": [], "level": 1, "when": str(pd.Timestamp(m["date"].iloc[-1]).date())})
    # 4) 情绪
    if breadth and breadth.get("zt") is not None:
        zt, dt = breadth.get("zt", 0), breadth.get("dt", 0) or 0
        if dt >= 50 or zt >= 120:
            sig.append({"title": "市场情绪极端", "text": f"今日涨停 {zt} 家、跌停 {dt} 家。极端情绪日之后短线波动通常放大。",
                        "picks": [], "level": 2, "when": "今日"})
    sig.sort(key=lambda x: -x["level"])
    return sig[:12]


def regime(index_quotes: list[dict], breadth: dict | None, ew_ret20: float | None,
           margin_chg5: float | None) -> dict:
    """简单的市场状态打分：仅作提示，不参与选股。"""
    score, notes = 0, []
    if ew_ret20 is not None:
        score += 1 if ew_ret20 > 0.03 else -1 if ew_ret20 < -0.03 else 0
        notes.append(f"全A等权近20日 {ew_ret20*100:+.1f}%")
    if margin_chg5 is not None:
        score += 1 if margin_chg5 > 0.005 else -1 if margin_chg5 < -0.005 else 0
        notes.append(f"融资余额5日 {margin_chg5*100:+.1f}%")
    if breadth and breadth.get("zt") is not None:
        z, d = breadth.get("zt") or 0, breadth.get("dt") or 0
        score += 1 if z > 3 * max(d, 1) else -1 if d > z else 0
        notes.append(f"涨停{z}/跌停{d}")
    level = "偏强" if score >= 2 else "偏弱" if score <= -2 else "中性"
    return {"level": level, "score": score, "text": "；".join(notes) + "。状态偏弱时短线分的参考价值下降，宜更看重价值分。"}
