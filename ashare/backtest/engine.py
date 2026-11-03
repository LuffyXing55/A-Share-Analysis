"""组合回测引擎。

规则（尽量贴近 A 股实盘）：
- 信号在调仓日 t 收盘后生成，t+1 日按成交均价（vwap）成交；
- t+1 一字涨停或停牌 → 买不进，顺延用排名下一位的股票替补；
- 一字跌停或停牌 → 卖不出，继续持有，下一个交易日再尝试卖出；
- 交易成本：买入 0.15%（佣金约 0.025% + 冲击/滑点约 0.12%）、卖出 0.25%（另含印花税 0.05%）；
- 净值按 vwap→vwap 计算，停牌期间价格前向填充。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BTConfig:
    top_n: int = 30
    buy_cost: float = 0.0015
    sell_cost: float = 0.0025
    industry_cap: int | None = 5          # 每个行业最多几只
    weight: str = "equal"                  # equal / score
    hold_buffer: int = 0                   # 老持仓排名在 top_n + buffer 内则保留（降低换手）
    name: str = "strategy"


@dataclass
class BTResult:
    nav: pd.Series
    turnover: pd.Series
    holdings: dict = field(default_factory=dict)
    trades: int = 0


def select(scores: pd.Series, eligible: pd.Series, cfg: BTConfig, industry: pd.Series | None,
           current: set[str] | None = None) -> list[str]:
    s = scores[eligible.reindex(scores.index).fillna(False)].dropna().sort_values(ascending=False)
    if s.empty:
        return []
    keep: list[str] = []
    if current and cfg.hold_buffer > 0:
        ranks = pd.Series(np.arange(len(s)), index=s.index)
        keep = [c for c in current if c in ranks.index and ranks[c] < cfg.top_n + cfg.hold_buffer]
    chosen: list[str] = []
    count: dict = {}
    for c in keep + [c for c in s.index if c not in keep]:
        if len(chosen) >= cfg.top_n:
            break
        if cfg.industry_cap and industry is not None:
            g = industry.get(c)
            if g is not None and pd.notna(g):
                if count.get(g, 0) >= cfg.industry_cap:
                    continue
                count[g] = count.get(g, 0) + 1
        chosen.append(c)
    return chosen


def run_backtest(scores: pd.DataFrame, eligible: pd.DataFrame, vwap: pd.DataFrame,
                 buy_block: pd.DataFrame, sell_block: pd.DataFrame, cfg: BTConfig,
                 industry: pd.DataFrame | None = None, start: str | None = None,
                 end: str | None = None) -> BTResult:
    """scores：仅在调仓日有行（index 为调仓日）。"""
    px = vwap.ffill(limit=60)
    dates = px.index
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    reb_dates = set(scores.index)
    cols = px.columns
    col_idx = {c: i for i, c in enumerate(cols)}
    w = np.zeros(len(cols))                  # 当前持仓权重（按 vwap 计价）
    pending_target: list[str] | None = None
    pending_scores: pd.Series | None = None
    nav, navs, turns = 1.0, [], []
    holdings = {}
    trades = 0
    prev_px = None
    prev_d = None
    for i, d in enumerate(dates):
        p_now = px.loc[d].values
        # 1) 持仓按 vwap 涨跌
        if prev_px is not None and w.sum() > 0:
            rel = np.where(np.isfinite(p_now) & np.isfinite(prev_px) & (prev_px > 0), p_now / prev_px, 1.0)
            gross = (w * rel).sum()
            nav *= gross / w.sum() if w.sum() > 0 else 1.0
            w = w * rel
            w = w / w.sum() * 1.0
        turn = 0.0
        # 2) 执行上一个调仓日生成的目标（今天 = t+1）
        if pending_target is not None:
            cur = {cols[j] for j in np.nonzero(w > 1e-12)[0]}
            target = list(pending_target)
            sb = sell_block.loc[prev_d] if prev_d in sell_block.index else None
            bb = buy_block.loc[prev_d] if prev_d in buy_block.index else None
            stuck = {c for c in cur - set(target) if sb is not None and bool(sb.get(c, False))}
            # 买不进的用候选名单后续股票替补
            final = []
            for c in target:
                if c in cur or bb is None or not bool(bb.get(c, False)):
                    final.append(c)
            if pending_scores is not None and len(final) < len(target):
                for c in pending_scores.index:
                    if len(final) >= len(target):
                        break
                    if c not in final and (bb is None or not bool(bb.get(c, False))):
                        final.append(c)
            new_w = np.zeros(len(cols))
            n_slots = max(1, len(final) + len(stuck))
            for c in stuck:
                new_w[col_idx[c]] = w[col_idx[c]] if w.sum() > 0 else 0
            free = 1.0 - new_w.sum()
            if cfg.weight == "score" and pending_scores is not None and len(final):
                sc = pending_scores.reindex(final).rank().values
                sc = sc / sc.sum()
                for c, x in zip(final, sc):
                    new_w[col_idx[c]] += free * x
            else:
                for c in final:
                    new_w[col_idx[c]] += free / max(1, len(final))
            delta = new_w - w
            buy = delta[delta > 0].sum()
            sell = -delta[delta < 0].sum()
            cost = buy * cfg.buy_cost + sell * cfg.sell_cost
            nav *= (1 - cost)
            turn = (buy + sell) / 2
            trades += int((np.abs(delta) > 1e-9).sum())
            w = new_w
            holdings[d] = final + sorted(stuck)
            pending_target = None
            _ = n_slots
        # 3) 调仓日：生成目标，下一交易日执行
        if d in reb_dates:
            sc = scores.loc[d]
            el = eligible.loc[d] if d in eligible.index else pd.Series(True, index=cols)
            ind = industry.loc[d] if industry is not None and d in industry.index else None
            cur = {cols[j] for j in np.nonzero(w > 1e-12)[0]}
            pending_target = select(sc, el, cfg, ind, cur)
            ranked = sc[el.reindex(sc.index).fillna(False)].dropna().sort_values(ascending=False)
            pending_scores = ranked
        navs.append(nav)
        turns.append(turn)
        prev_px = p_now
        prev_d = d
    return BTResult(pd.Series(navs, index=dates), pd.Series(turns, index=dates), holdings, trades)


def metrics(nav: pd.Series, bench: pd.Series | None = None, turnover: pd.Series | None = None) -> dict:
    nav = nav.dropna()
    r = nav.pct_change().dropna()
    years = len(r) / 244
    ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(244)
    dd = (nav / nav.cummax() - 1).min()
    out = {"年化收益": ann, "年化波动": vol, "夏普": (r.mean() * 244) / vol if vol > 0 else np.nan,
           "最大回撤": dd, "卡玛": ann / abs(dd) if dd < 0 else np.nan}
    if bench is not None:
        b = bench.reindex(nav.index).ffill()
        br = b.pct_change().dropna()
        ex = (r - br.reindex(r.index)).dropna()
        b_ann = b.iloc[-1] / b.iloc[0]
        b_ann = b_ann ** (1 / years) - 1
        out["基准年化"] = b_ann
        out["超额年化"] = ann - b_ann
        out["信息比"] = ex.mean() / ex.std() * np.sqrt(244) if ex.std() > 0 else np.nan
        exnav = (1 + ex).cumprod()
        out["超额最大回撤"] = (exnav / exnav.cummax() - 1).min()
        m = nav.resample("ME").last().pct_change().dropna()
        bm = b.resample("ME").last().pct_change().dropna()
        out["月胜率(跑赢基准)"] = float((m - bm.reindex(m.index) > 0).mean())
    if turnover is not None:
        out["年换手(单边)"] = turnover.sum() / years
    return {k: (round(float(v), 4) if v is not None and np.isfinite(v) else None) for k, v in out.items()}


def equal_weight_index(vwap: pd.DataFrame, members: pd.DataFrame, rebalance: pd.DatetimeIndex) -> pd.Series:
    """等权基准：每个调仓日按成分等权，vwap 计价，不计成本。"""
    px = vwap.ffill(limit=60)
    r = px.pct_change(fill_method=None)
    nav = [1.0]
    idx = px.index
    w = None
    reb = set(rebalance)
    for i in range(1, len(idx)):
        d_prev, d = idx[i - 1], idx[i]
        if w is not None:
            rr = r.loc[d].reindex(w.index).fillna(0.0)
            g = (w * (1 + rr)).sum()
            nav.append(nav[-1] * g)
            w = w * (1 + rr) / g
        else:
            nav.append(nav[-1])
        if d_prev in reb or w is None:
            m = members.loc[d_prev] if d_prev in members.index else None
            if m is not None:
                names = m[m].index
                names = [c for c in names if np.isfinite(px.loc[d_prev].get(c, np.nan))]
                if names:
                    w = pd.Series(1.0 / len(names), index=names)
    return pd.Series(nav, index=idx)
