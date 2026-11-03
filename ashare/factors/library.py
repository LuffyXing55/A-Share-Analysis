"""因子库：输入「日期 × 股票」宽表，输出同形状的因子宽表。

约定：所有因子都已调整为「数值越大 → 预期收益越高」的方向（依据资料汇编第 3 部分的 A 股实证，
并在本项目回测中复核，见 docs/回测报告.md）。回测与实盘使用同一份代码，保证口径一致。

输入字段（价格均为复权归一化价，原始价 = close / factor）：
  open high low close vwap volume amount factor        （必需）
  pe_ttm pb ps_ttm dv_ttm total_mv circ_mv turnover_rate_f （可选，基本面因子用）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Panel = dict[str, pd.DataFrame]

# 因子分组：value=价值投资分使用；short=短线分使用
VALUE_FACTORS = [
    "ep", "bp", "sp", "dy", "roe", "e_growth", "s_growth", "e_revision",
    "size", "vol_60", "ivol_20", "turn_20", "amihud_20", "rev_20", "maxret_20", "mom_250_20",
]
SHORT_FACTORS = [
    "rev_5", "rev_1", "rev_20", "vol_20", "ivol_20", "maxret_20", "turn_5", "abn_turn",
    "amihud_20", "pv_corr_20", "ovn_20", "intra_20", "gap_1", "close_pos_1", "dist_ma20",
    "high52", "vol_ratio_5", "limit_up_5", "size",
]


def _ret(close: pd.DataFrame) -> pd.DataFrame:
    return close.pct_change(fill_method=None)


def _roll_mean(df: pd.DataFrame, n: int, minp: float = 0.6) -> pd.DataFrame:
    return df.rolling(n, min_periods=max(2, int(n * minp))).mean()


def _roll_std(df: pd.DataFrame, n: int, minp: float = 0.6) -> pd.DataFrame:
    return df.rolling(n, min_periods=max(3, int(n * minp))).std()


def _rolling_corr(a: pd.DataFrame, b: pd.DataFrame, n: int) -> pd.DataFrame:
    ma, mb = _roll_mean(a, n), _roll_mean(b, n)
    cov = _roll_mean(a * b, n) - ma * mb
    va = _roll_mean(a * a, n) - ma * ma
    vb = _roll_mean(b * b, n) - mb * mb
    out = cov / np.sqrt((va * vb).where((va > 0) & (vb > 0)))
    return out


def compute_factors(p: Panel, which: set[str] | None = None) -> dict[str, pd.DataFrame]:
    close, open_, high, low = p["close"], p["open"], p["high"], p["low"]
    vol, amount = p["volume"], p["amount"]
    r = _ret(close)
    mkt = r.mean(axis=1)
    f: dict[str, pd.DataFrame] = {}

    def want(name: str) -> bool:
        return which is None or name in which

    # ---------------- 反转 / 动量
    if want("rev_1"):
        f["rev_1"] = -r
    if want("rev_5"):
        f["rev_5"] = -(close / close.shift(5) - 1)
    if want("rev_20"):
        f["rev_20"] = -(close / close.shift(20) - 1)
    if want("mom_250_20"):
        f["mom_250_20"] = close.shift(20) / close.shift(250) - 1
    # ---------------- 波动 / 特质波动 / 彩票
    if want("vol_20"):
        f["vol_20"] = -_roll_std(r, 20)
    if want("vol_60"):
        f["vol_60"] = -_roll_std(r, 60)
    if want("ivol_20"):
        resid = r.sub(mkt, axis=0)
        f["ivol_20"] = -_roll_std(resid, 20)
    if want("maxret_20"):
        f["maxret_20"] = -r.rolling(20, min_periods=12).max()
    # ---------------- 流动性 / 换手
    if want("amihud_20"):
        illiq = r.abs() / (amount.where(amount > 0) * 1000.0) * 1e8   # 每亿元成交的价格冲击
        f["amihud_20"] = np.log1p(_roll_mean(illiq, 20))
    if "turnover_rate_f" in p:
        t = p["turnover_rate_f"]
        if want("turn_20"):
            f["turn_20"] = -np.log1p(_roll_mean(t, 20))
        if want("turn_5"):
            f["turn_5"] = -np.log1p(_roll_mean(t, 5))
    else:
        # 2020 年以前没有换手率数据：用成交额 / 60 日均成交额近似（只影响研究，不影响实盘）
        if want("turn_20"):
            f["turn_20"] = -np.log1p(_roll_mean(amount, 20) / _roll_mean(amount, 250))
        if want("turn_5"):
            f["turn_5"] = -np.log1p(_roll_mean(amount, 5) / _roll_mean(amount, 250))
    if want("abn_turn"):
        f["abn_turn"] = -np.log((_roll_mean(vol, 20) / _roll_mean(vol, 250, 0.5)).where(lambda x: x > 0))
    if want("vol_ratio_5"):
        f["vol_ratio_5"] = -np.log((vol / _roll_mean(vol, 5).shift(1)).where(lambda x: x > 0))
    if want("pv_corr_20"):
        f["pv_corr_20"] = -_rolling_corr(close, vol, 20)
    # ---------------- 隔夜 / 日内（A 股隔夜为负、日内为正；资料汇编 3.3）
    if want("ovn_20") or want("intra_20") or want("gap_1"):
        ovn = open_ / close.shift(1) - 1
        intra = close / open_ - 1
        if want("ovn_20"):
            f["ovn_20"] = -_roll_mean(ovn, 20)
        if want("intra_20"):
            f["intra_20"] = _roll_mean(intra, 20)
        if want("gap_1"):
            f["gap_1"] = -ovn
    if want("close_pos_1"):
        rng = (high - low).where(lambda x: x > 0)
        f["close_pos_1"] = -((close - low) / rng)
    if want("dist_ma20"):
        f["dist_ma20"] = -(close / _roll_mean(close, 20) - 1)
    if want("high52"):
        f["high52"] = close / close.rolling(250, min_periods=120).max()
    if want("limit_up_5"):
        lim = _limit_pct(close.columns)
        hit = (r >= lim.values[None, :] - 0.003).astype("float32").where(r.notna())
        f["limit_up_5"] = -hit.rolling(5, min_periods=3).sum()

    # ---------------- 基本面（2020 年起有数据）
    if "pe_ttm" in p:
        pe, pb = p["pe_ttm"], p["pb"]
        mv = p["total_mv"]
        ep = (1.0 / pe).where(pe > 0)
        bp = (1.0 / pb).where(pb > 0)
        if want("ep"):
            # 亏损公司 EP 记为截面最低
            f["ep"] = ep.where(ep.notna(), ep.min(axis=1) - 0.01, axis=0).where(pb.notna())
        if want("bp"):
            f["bp"] = bp
        if want("sp") and "ps_ttm" in p:
            ps = p["ps_ttm"]
            f["sp"] = (1.0 / ps).where(ps > 0)
        if want("dy") and "dv_ttm" in p:
            f["dy"] = p["dv_ttm"].fillna(0.0).where(pb.notna())
        if want("roe"):
            f["roe"] = (pb / pe).where(pe > 0).clip(upper=0.6)
            f["roe"] = f["roe"].where(f["roe"].notna(), -0.05).where(pb.notna())
        e = (mv / pe).where(pe > 0)                     # TTM 净利润（万元）
        if want("e_growth"):
            e_l = e.shift(250)
            g = (e - e_l) / e_l.abs()
            f["e_growth"] = np.sign(g) * np.log1p(g.abs()).clip(upper=3)
        if want("e_revision"):
            e_q = e.shift(63)
            g = (e - e_q) / e_q.abs()
            f["e_revision"] = np.sign(g) * np.log1p(g.abs()).clip(upper=3)
        if want("s_growth") and "ps_ttm" in p:
            s = (mv / p["ps_ttm"]).where(p["ps_ttm"] > 0)
            g = s / s.shift(250) - 1
            f["s_growth"] = np.log1p(g.clip(lower=-0.9, upper=5))
        if want("size"):
            f["size"] = -np.log(mv.where(mv > 0))
    elif want("size"):
        f["size"] = -np.log(_roll_mean(amount, 60).where(lambda x: x > 0))
    return {k: v.astype("float32") for k, v in f.items()}


def _limit_pct(codes) -> pd.Series:
    """各板块涨跌幅限制（主板 10%，创业板/科创板 20%；2020-08-24 前创业板为 10%，研究里忽略这一细节）。"""
    vals = []
    for c in codes:
        n = c[2:]
        vals.append(0.2 if n.startswith(("300", "301", "688")) else 0.1)
    return pd.Series(vals, index=codes)


def limit_table(close: pd.DataFrame, is_st: pd.DataFrame | None = None) -> pd.DataFrame:
    """每日涨跌停幅度表（ST 为 5%）。创业板 2020-08-24 起为 20%。"""
    lim = pd.DataFrame(np.repeat(_limit_pct(close.columns).values[None, :], len(close), 0),
                       index=close.index, columns=close.columns)
    gem = [c for c in close.columns if c[2:].startswith(("300", "301"))]
    lim.loc[lim.index < pd.Timestamp("2020-08-24"), gem] = 0.1
    if is_st is not None:
        main = [c for c in close.columns if not c[2:].startswith(("300", "301", "688"))]
        st = is_st.reindex_like(lim).fillna(False)
        lim[main] = lim[main].where(~st[main], 0.05)
    return lim


# ------------------------------------------------------------------ 截面处理
def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """截面百分位（0~1），NaN 保持 NaN。"""
    return df.rank(axis=1, pct=True)


def cs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    m = df.mean(axis=1)
    s = df.std(axis=1)
    return df.sub(m, axis=0).div(s, axis=0).clip(-clip, clip)


def neutralize(df: pd.DataFrame, groups: pd.DataFrame | None = None,
               size: pd.DataFrame | None = None) -> pd.DataFrame:
    """对每个日期做截面回归残差：因子 ~ 行业哑变量 + 市值。用于行业 / 市值中性化。"""
    if groups is None and size is None:
        return df
    out = np.full(df.shape, np.nan, dtype="float32")
    vals = df.values
    for i in range(len(df)):
        y = vals[i]
        cols = []
        if size is not None:
            cols.append(size.values[i])
        mask = np.isfinite(y)
        for c in cols:
            mask &= np.isfinite(c)
        if groups is not None:
            g = groups.values[i]
            mask &= pd.notna(g)
        if mask.sum() < 50:
            continue
        X = [np.ones(mask.sum())]
        for c in cols:
            X.append((c[mask] - np.nanmean(c[mask])) / (np.nanstd(c[mask]) + 1e-9))
        if groups is not None:
            dummies = pd.get_dummies(pd.Series(g[mask])).values.astype(float)
            X = [x for x in X[1:]] + [dummies[:, j] for j in range(dummies.shape[1])]
        X = np.column_stack(X)
        beta, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        res = y[mask] - X @ beta
        out[i, mask] = res
    return pd.DataFrame(out, index=df.index, columns=df.columns)
