"""每日流水线：计算全市场因子与分数 → 月度 30 强 → 每日排名 → 海外/新闻/事件 → 网站 JSON。

用法：
  python -m ashare.pipeline.daily --mode close      # 收盘后（完整计算，GitHub Actions 每个交易日傍晚运行）
  python -m ashare.pipeline.daily --mode premarket  # 开盘前（只刷新海外信号、新闻与观察清单）

输出目录 site_data/：
  dashboard.json   网站主数据       backtest.json  回测展示数据（由训练脚本生成）
  state/picks_YYYY-MM.json   每月入选名单（选出后整月固定）
  state/daily_YYYY-MM.json   月内每日分数与排名历史
  state/latest.json          最近一次完整计算的 30 只股票基础分（供开盘前与本地实时模式复用）
  state/factor_dist.json     全市场因子分布分位点（本地实时模式用来给盘中数据重新排名）
"""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.industry import l1_of
from ..factors.library import compute_factors, industry_factors
from ..live import sources as S
from ..macro import overseas as OV
from ..macro import watch as W
from ..macro.linkage import CHAINS
from ..model import tail as TAIL
from ..model.predict import FACTOR_LABELS, Model, explain, matrix
from ..news.sentiment import aggregate, match_news_to_stocks
from ..research.common import month_ends, universe
from .history import default_cache_dir, ensure_qlib_data, recent_panels
from .snapshot import append_today_bar, basics_from_snapshot, market_snapshot

BJ = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parents[2]
INDEX_CODES = [("sh000001", "上证指数"), ("sz399001", "深证成指"), ("sz399006", "创业板指"),
               ("sh000688", "科创50"), ("sh000300", "沪深300"), ("sh000905", "中证500"), ("sh000852", "中证1000"),
               ("sh000985", "中证全指")]


def now_bj() -> datetime:
    return datetime.now(BJ)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return default


def sanitize(o):
    """递归把 NaN/Inf 换成 None（浏览器的 JSON.parse 不接受 NaN）。"""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sanitize(obj), ensure_ascii=False, indent=0, default=_default, allow_nan=False))
    tmp.replace(path)


def _default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return str(o)


def clean(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


# ------------------------------------------------------------------ 交易日历
def trading_calendar(qlib_root: Path | None) -> pd.DatetimeIndex:
    days = []
    if qlib_root is not None:
        for f in ["day.txt", "day_future.txt"]:
            p = qlib_root / "calendars" / f
            if p.exists():
                days += p.read_text().split()
    cal = pd.DatetimeIndex(sorted(set(pd.to_datetime(days)))) if days else pd.DatetimeIndex([])
    # 日历之外的未来日期用工作日近似（节假日未知，见资料汇编 0.3(i)）
    last = cal[-1] if len(cal) else pd.Timestamp(now_bj().date()) - pd.Timedelta(days=400)
    extra = pd.bdate_range(last + pd.Timedelta(days=1), last + pd.Timedelta(days=120))
    return cal.append(extra)


def is_last_trading_day_of_month(d: pd.Timestamp, cal: pd.DatetimeIndex) -> bool:
    nxt = cal[cal > d]
    return len(nxt) > 0 and nxt[0].month != d.month


# ------------------------------------------------------------------ 数据准备
class Context:
    def __init__(self, out_dir: Path, settings: dict):
        self.out = out_dir
        self.state = out_dir / "state"
        self.settings = settings
        self.sources: dict[str, str] = {}
        self.log: list[str] = []

    def note(self, key: str, ok: bool, msg: str = "") -> None:
        self.sources[key] = "ok" if ok else f"失败：{msg[:100]}"
        self.log.append(f"{'OK ' if ok else 'ERR'} {key} {msg}")
        print(self.log[-1], flush=True)


def load_industry(ctx: Context) -> pd.DataFrame:
    """申万行业（东财业绩报表 PUBLISHNAME），缓存一周。"""
    cache = ctx.state / "industry.json"
    cached = read_json(cache, {})
    if cached and cached.get("date", "") >= (now_bj() - timedelta(days=7)).strftime("%Y-%m-%d"):
        return pd.DataFrame(cached["rows"])
    rows = []
    try:
        qe = pd.date_range(end=pd.Timestamp(now_bj().date()), periods=6, freq="QE")[::-1]
        for rd in [str(x.date()) for x in qe]:
            df = S.em_performance(rd, max_pages=15)
            if len(df) > 1000:
                df["code"] = df["SECUCODE"].str.split(".").map(lambda x: x[1].lower() + x[0])
                rows = df[["code", "SECURITY_NAME_ABBR", "PUBLISHNAME", "WEIGHTAVG_ROE", "YSTZ", "SJLTZ",
                           "NOTICE_DATE", "DATATYPE"]].rename(columns={
                    "SECURITY_NAME_ABBR": "name", "PUBLISHNAME": "l2", "WEIGHTAVG_ROE": "roe_report",
                    "YSTZ": "rev_yoy", "SJLTZ": "np_yoy", "NOTICE_DATE": "notice_date", "DATATYPE": "report"})
                rows["l1"] = rows["l2"].map(l1_of)
                break
        ctx.note("东财业绩报表/行业", len(rows) > 0)
    except Exception as e:  # noqa: BLE001
        ctx.note("东财业绩报表/行业", False, str(e))
    if len(rows):
        write_json(cache, {"date": now_bj().strftime("%Y-%m-%d"),
                           "rows": rows.astype(object).where(rows.notna(), None).to_dict(orient="records")})
        return rows
    return pd.DataFrame(cached.get("rows", [])) if cached else pd.DataFrame(columns=["code", "l1"])


def compute_scores(ctx: Context, qlib_root: Path):
    st_cfg = ctx.settings["universe"]
    p = recent_panels(qlib_root, days=420)
    codes = list(p["close"].columns)
    ctx.note("GitHub 日线数据", True, f"{len(codes)} 只，截至 {p['close'].index[-1].date()}")
    snap = pd.DataFrame()
    try:
        snap = market_snapshot(codes)
        ctx.note("腾讯全市场快照", len(snap) > 1000, f"{len(snap)} 只")
    except Exception as e:  # noqa: BLE001
        ctx.note("腾讯全市场快照", False, str(e))
    today = pd.Timestamp(now_bj().date())
    after_close = now_bj().hour >= 15
    if len(snap) and after_close and p["close"].index[-1] < today:
        tstamps = snap["time"].astype(str).str[:8]
        if (tstamps == today.strftime("%Y%m%d")).mean() > 0.5:
            p = append_today_bar(p, snap, today)
            ctx.log.append("用腾讯快照补齐了今日日线")
            ctx.snapshot_bar = True
    if len(snap):
        p.update(basics_from_snapshot(p, snap))
    st = None
    if len(snap):
        st_row = snap["is_st"].reindex(p["close"].columns).fillna(False) | \
            snap["is_delist"].reindex(p["close"].columns).fillna(False)
        st = pd.DataFrame(False, index=p["close"].index, columns=p["close"].columns)
        st.iloc[-1] = st_row.values
    u = universe(p, st_cfg["min_amount_wan"], st_cfg["min_age_days"], st_cfg["drop_bottom_liquidity"], st=st)
    fac = compute_factors(p)
    ind = getattr(ctx, "industry_map", None)
    if ind:
        fac.update(industry_factors(p["close"], u, pd.Series(ind)))
    d = p["close"].index[-1]
    mask = u.loc[d]
    ranks = {k: v.loc[d].where(mask).rank(pct=True) for k, v in fac.items()}
    raw_last = {k: v.loc[d].where(mask) for k, v in fac.items()}
    ctx.fac_last = {k: v.loc[d] for k, v in fac.items()}
    ctx.fac_full, ctx.u_full = fac, u
    return p, snap, mask, ranks, raw_last, d


def rank_outside(ranks: dict, raw_last: dict, fac_last: dict, codes: list[str]) -> None:
    """不在股票池里的入选股（停牌、变成 ST 等）：用其原始因子值对照股票池分布换算分位，就地写入 ranks。"""
    for k, r in ranks.items():
        vals = np.sort(raw_last[k].dropna().values)
        if len(vals) == 0:
            continue
        for c in codes:
            x = fac_last.get(k, pd.Series(dtype=float)).get(c)
            if x is None or not np.isfinite(x):
                continue
            lo, hi = np.searchsorted(vals, x, "left"), np.searchsorted(vals, x, "right")
            r.loc[c] = (lo + hi) / 2 / len(vals)


def score_universe(ranks: dict, mask: pd.Series, short_model: Model, tail_w: dict) -> pd.DataFrame:
    """中期分（价值投资分）= 尾部加权合成；短线分 = LightGBM 5 日模型。均换算为股票池内分位（0–100）。"""
    codes = list(mask[mask].index)
    out = pd.DataFrame(index=codes)
    out["value_raw"] = TAIL.score(ranks, tail_w, codes).values
    out["value"] = out["value_raw"].rank(pct=True) * 100
    X = matrix(ranks, short_model.features, codes)
    out["short_raw"] = short_model.predict(X)
    out["short"] = out["short_raw"].rank(pct=True) * 100
    return out


def select_picks(scores: pd.DataFrame, industry: pd.Series, settings: dict,
                 previous: list[str] | None = None) -> list[str]:
    """按综合分选 top_n，同一行业最多 industry_cap 只；上月持仓只要仍在前 top_n+hold_buffer 名就优先保留（降换手）。"""
    s = scores["total"].dropna()
    vf = settings.get("value_filter", {})
    if vf and "pe" in scores:
        # 价值过滤：只在盈利、且 PE 低于「盈利公司中 PE 的 70 分位」的股票中选（排除盈利公司里最贵的 30%）。
        # 72 组参数的选参期 / 保留期检验都支持这一约束（见 docs/方法与回测报告.md 第 6 节）
        pe = scores["pe"].reindex(s.index)
        prof = pe > 0
        pe_pct = pe.where(prof).rank(pct=True)
        s = s[prof & (pe_pct <= vf.get("max_pe_pct_among_profitable", 0.7))]
    s = s.sort_values(ascending=False)
    cap, n = settings.get("industry_cap", 5), settings.get("top_n", 30)
    buf = settings.get("hold_buffer", 400)
    rank = pd.Series(np.arange(len(s)), index=s.index)
    keep = [c for c in (previous or []) if c in rank.index and rank[c] < n + buf]
    keep = sorted(keep, key=lambda c: rank[c])
    cnt, chosen = {}, []
    for code in keep + [c for c in s.index if c not in keep]:
        g = industry.get(code)
        if isinstance(g, str) and cnt.get(g, 0) >= cap:
            continue
        cnt[g] = cnt.get(g, 0) + 1
        chosen.append(code)
        if len(chosen) >= n:
            break
    return chosen


def factor_distribution(raw_last: dict) -> dict:
    qs = np.linspace(0, 1, 101)
    out = {}
    for k, v in raw_last.items():
        x = v.dropna().values
        if len(x) > 100:
            out[k] = [clean(q, 6) for q in np.quantile(x, qs)]
    return out


# ------------------------------------------------------------------ 主流程
def run(mode: str = "close", out_dir: str | Path = "site_data", models_dir: str | Path | None = None) -> dict:
    out = Path(out_dir)
    settings = read_json(ROOT / "config" / "settings.json", {})
    ctx = Context(out, settings)
    mdir = Path(models_dir or ROOT / "models")
    short_model = Model(mdir / "short_model.txt")
    tail_path = out / "state" / "tail_table.csv"   # 会被 Actions 提交；首次运行时用 config 里的研究期数据初始化
    if not tail_path.exists():
        tail_path.parent.mkdir(parents=True, exist_ok=True)
        tail_path.write_text((ROOT / "config" / "tail_table.csv").read_text())
    w = settings.get("score_weights", {"value": 0.6, "short": 0.4})
    latest = read_json(ctx.state / "latest.json", {})
    industry_df = load_industry(ctx)
    ind_map = dict(zip(industry_df.get("code", []), industry_df.get("l1", []))) if len(industry_df) else {}
    name_map = dict(zip(industry_df.get("code", []), industry_df.get("name", []))) if len(industry_df) else {}
    qlib_root = None
    cal = trading_calendar(None)

    ctx.industry_map = {k: v for k, v in ind_map.items() if isinstance(v, str)}
    if mode == "premarket" and not latest:
        ctx.log.append("还没有收盘后的结果，改为完整计算")
        mode = "close"
    if mode == "close":
        qlib_root = ensure_qlib_data(default_cache_dir())
        cal = trading_calendar(qlib_root)
        p, snap, mask, ranks, raw_last, d = compute_scores(ctx, qlib_root)
        if len(snap):
            name_map.update(snap["name"].to_dict())
        raw = p["close"] / p["factor"]
        # 补记名单：价格面板里「晚于尾部表最后一行、早于今天、又没有名单文件」的所有月末（流水线没跑成的月份）。
        # 历史月末没有当时的估值快照，估值类因子会缺失，文件里用 partial 字段标明。
        table0 = TAIL.load_table(tail_path)
        last_row = table0.index.max() if len(table0) else pd.Timestamp("1900-01-01")
        st_now = (snap["is_st"] | snap["is_delist"]).reindex(raw.columns).fillna(False) if len(snap) else None
        for pd_me in month_ends(raw.index[raw.index < d]):
            if pd_me <= last_row or (ctx.state / f"tail_members_{pd_me.date()}.json").exists():
                continue
            if raw.index.get_loc(pd_me) < 260:          # 因子需要约 250 天历史
                continue
            m_me = ctx.u_full.loc[pd_me]
            if st_now is not None:
                m_me = m_me & ~st_now
            r_me, partial = {}, []
            for k, v in ctx.fac_full.items():
                x = v.loc[pd_me].where(m_me)
                if x.notna().sum() > 500:
                    r_me[k] = x.rank(pct=True)
                else:
                    partial.append(k)
            TAIL.backfill_members(ctx.state, pd_me, r_me, m_me, partial=partial)
            ctx.note(f"补记尾部名单 {pd_me.date()}", True, f"缺少：{','.join(partial) or '无'}")
        table = TAIL.update_table(ctx.state, tail_path, p["vwap"],
                                  last_is_snapshot=getattr(ctx, "snapshot_bar", False))
        tail_w = TAIL.weights(table, d, available=set(ranks))
        ctx.note("尾部加权因子", len(tail_w) > 0, ", ".join(f"{k}:{v:.2f}" for k, v in
                                                     sorted(tail_w.items(), key=lambda x: -x[1])))
        sc = score_universe(ranks, mask, short_model, tail_w)
        sc["total"] = (w["value"] * sc["value"] + w["short"] * sc["short"]).rank(pct=True) * 100
        if "pe_ttm" in p:
            sc["pe"] = p["pe_ttm"].iloc[-1].reindex(sc.index)
        month = f"{d.year}-{d.month:02d}"
        picks_file = ctx.state / f"picks_{month}.json"
        picks_state = read_json(picks_file)
        industry = pd.Series(ind_map)
        prev_month = (d.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
        # 快照覆盖率检查：股票池内估值 / 流通市值缺失超过 5% 时，选股口径与回测不一致，本次不锁定新组合
        cov = 0.0
        if len(snap) and "pe" in sc:
            cov = float(snap.reindex(sc.index)[["pe_ttm", "float_mv"]].notna().all(axis=1).mean())
        can_pick = cov >= 0.95
        if not can_pick:
            ctx.note("选股数据完整性", False, f"快照覆盖率 {cov:.0%} < 95%，本次不锁定新组合，等待下一次运行")
        prev_codes = (read_json(ctx.state / f"picks_{prev_month}.json", {}) or {}).get("codes")
        if picks_state is None and not can_pick:
            # 数据不完整又没有本月组合：沿用最近一份组合展示（不写文件），下一次运行再正式选；
            # 一份都没有时，临时选出一组只用于展示（同样不写文件），其余数据照常发布
            older = sorted(ctx.state.glob("picks_*.json"))
            picks_state = read_json(older[-1]) if older else None
            if picks_state is not None:
                picks_state = {**picks_state, "note": f"数据不完整，暂时沿用 {picks_state['month']} 的组合，等待下一次运行正式选出"}
            else:
                chosen = select_picks(sc, industry, settings, None)
                picks_state = {"month": month, "pick_date": str(d.date()), "codes": chosen,
                               "entry_close": {c: clean(raw.loc[d, c]) for c in chosen},
                               "note": "数据不完整，这是临时组合（未锁定），下一次运行会正式选出"}
        if picks_state is None:
            # 本月还没有组合（首次运行或上月最后一天未运行）：立即选出
            chosen = select_picks(sc, industry, settings, prev_codes)
            picks_state = {"month": month, "pick_date": str(d.date()), "codes": chosen,
                           "entry_close": {c: clean(raw.loc[d, c]) for c in chosen},
                           "note": "月中补选" if d.day > 7 else "月初选出"}
            write_json(picks_file, picks_state)
        if is_last_trading_day_of_month(d, cal) and can_pick:
            # 只在数据完整时记录月末名单（避免 21:07 的重跑用不完整数据覆盖）
            TAIL.record_members(ctx.state, d, {k: v.where(mask) for k, v in ranks.items()}, mask)
        if is_last_trading_day_of_month(d, cal):
            nxt = cal[cal > d][0]
            nmonth = f"{nxt.year}-{nxt.month:02d}"
            nfile = ctx.state / f"picks_{nmonth}.json"
            if not nfile.exists() and can_pick:
                chosen = select_picks(sc, industry, settings, picks_state["codes"])
                write_json(nfile, {"month": nmonth, "pick_date": str(d.date()), "codes": chosen,
                                   "entry_close": {c: clean(raw.loc[d, c]) for c in chosen},
                                   "note": "上月最后一个交易日收盘后选出"})
            # 从今晚起（含明早开盘前、明天盘中）网站显示的就是下月新组合，入选价 = 今日收盘价
            if nfile.exists():
                picks_state = read_json(nfile)
                month = nmonth
        codes = picks_state["codes"]
        # 入选股若已不在股票池（停牌/ST 等），按同样方法单独打分（相对全市场分布）
        missing = [c for c in codes if c not in sc.index]
        if missing:
            rank_outside(ranks, raw_last, getattr(ctx, "fac_last", {}), missing)
            vr = TAIL.score(ranks, tail_w, missing)
            sr = pd.Series(short_model.predict(matrix(ranks, short_model.features, missing)), index=missing)
            for c in missing:
                sc.loc[c, "value"] = (sc["value_raw"] < vr[c]).mean() * 100 if np.isfinite(vr[c]) else np.nan
                sc.loc[c, "short"] = (sc["short_raw"] < sr[c]).mean() * 100
                sc.loc[c, "total"] = np.nan
        expl_short = explain(short_model, matrix(ranks, short_model.features, codes), codes, ranks)
        base = {}
        for c in codes:
            pc = raw[c].dropna()
            rv = TAIL.contributions(ranks, tail_w, c)[:6]
            for r_ in rv:
                r_["name"] = FACTOR_LABELS.get(r_["factor"], r_["factor"])
            base[c] = {
                "code": c, "name": name_map.get(c, c), "industry": ind_map.get(c),
                "value": clean(sc.loc[c, "value"], 1), "short_model": clean(sc.loc[c, "short"], 1),
                "total_base": clean(sc.loc[c, "total"], 1),
                "price": clean(pc.iloc[-1], 3) if len(pc) else None,
                "pct_today": clean(pc.iloc[-1] / pc.iloc[-2] - 1) if len(pc) > 1 and pc.index[-1] == d else None,
                "suspended": bool(len(pc) and pc.index[-1] != d),
                "ret_since_pick": clean(pc.iloc[-1] / picks_state["entry_close"].get(c) - 1)
                if picks_state["entry_close"].get(c) else None,
                "is_st": bool(snap["is_st"].get(c, False)) if len(snap) else False,
                "reasons_value": rv,
                "reasons_short": expl_short.get(c, []),
                "pe_ttm": clean(snap["pe_ttm"].get(c)) if len(snap) else None,
                "pb": clean(snap["pb"].get(c)) if len(snap) else None,
                "total_mv_yi": clean(snap["total_mv"].get(c) / 1e8, 1) if len(snap) and c in snap.index else None,
                # 收盘时各因子的全市场分位：本地盘中模式下算不出来的因子（如行业动量）沿用这里的值
                "feat_ranks": {k: clean(v.get(c), 4) for k, v in ranks.items()},
            }
        ew = raw.pct_change(fill_method=None).loc[:, mask[mask].index].mean(axis=1)
        latest = {"data_date": str(d.date()), "month": month, "pick_date": picks_state["pick_date"],
                  "picks_note": picks_state.get("note", ""),
                  "base": base, "ew_ret20": clean((1 + ew.tail(20)).prod() - 1),
                  "universe_n": int(mask.sum()), "tail_weights": tail_w}
        write_json(ctx.state / "latest.json", latest)
        qs = np.linspace(0, 1, 101)
        pred_dist = {k: [clean(x, 6) for x in np.quantile(sc[k + "_raw"].dropna().values, qs)]
                     for k in ("value", "short")}
        mret = p["close"].pct_change(fill_method=None).mean(axis=1).tail(300)
        write_json(ctx.state / "factor_dist.json", {"date": str(d.date()), "dist": factor_distribution(raw_last),
                                                    "pred_dist": pred_dist, "weights": w, "tail_weights": tail_w,
                                                    "mkt_ret": {str(k.date()): clean(v, 6) for k, v in mret.items()}})
        track = update_track(ctx, raw, mask, d)
    else:
        track = read_json(ctx.state / "track.json", [])

    return build_dashboard(ctx, latest, track, mode)


def update_track(ctx: Context, raw: pd.DataFrame, mask: pd.Series, d: pd.Timestamp) -> list:
    """实盘跟踪：每个月组合自选出日收盘到月末（或今天）的等权收益，对比股票池等权。"""
    rows = []
    for f in sorted(ctx.state.glob("picks_*.json")):
        ps = read_json(f)
        if not ps:
            continue
        pdate = pd.Timestamp(ps["pick_date"])
        if pdate not in raw.index or pdate >= d:
            continue
        end_month = pd.Timestamp(ps["month"] + "-01") + pd.offsets.MonthEnd(0)
        end = min(d, raw.index[raw.index <= end_month].max())
        if end <= pdate:
            continue
        r = (raw.loc[end] / raw.loc[pdate] - 1)
        port = r.reindex(ps["codes"]).mean()
        bench = r.where(mask).mean()
        rows.append({"month": ps["month"], "pick_date": ps["pick_date"], "ret": clean(port), "bench": clean(bench),
                     "excess": clean(port - bench), "status": "已完成" if end < d else "进行中"})
    write_json(ctx.state / "track.json", rows)
    return rows


def build_dashboard(ctx: Context, latest: dict, track: list, mode: str) -> dict:
    s = ctx.settings
    ov_cfg = s.get("overlay", {})
    w = s.get("score_weights", {"value": 0.6, "short": 0.4})
    base = latest["base"]
    codes = list(base)
    # ---- 海外
    stats = OV.load_linkage_stats(ROOT / "config" / "linkage_stats.json")
    quotes = {}
    try:
        quotes = OV.fetch_overseas(OV.needed_symbols(codes, [base[c].get('industry') for c in codes]))
        ok = sum(1 for q in quotes.values() if q.get("ok"))
        ctx.note("海外行情(腾讯美股/新浪期货)", ok > 0, f"{ok}/{len(quotes)}")
    except Exception as e:  # noqa: BLE001
        ctx.note("海外行情(腾讯美股/新浪期货)", False, str(e))
    # ---- 新闻
    news = []
    try:
        news = S.all_news(80)
        ctx.note("新闻快讯", len(news) > 0, f"{len(news)} 条")
    except Exception as e:  # noqa: BLE001
        ctx.note("新闻快讯", False, str(e))
    names = {c: base[c]["name"] for c in codes}
    chain_kw = {}
    for c in codes:
        kws = []
        for ch in CHAINS:
            if c in ch["a_shares"]:
                kws += [ch["name"].split("/")[0].split("(")[0]]
        if kws:
            chain_kw[c] = kws
    matched = match_news_to_stocks(news, names, chain_kw)
    news_out = []
    for n in news[:60]:
        from ..news.sentiment import score_text

        st = score_text(n.get("title", "") + n.get("content", ""))
        txt = n.get("title", "") + n.get("content", "")
        picks_hit = [c for c, nm in names.items() if len(nm) >= 2 and nm.replace("*", "").replace("ST", "") in txt]
        news_out.append({**n, "content": n.get("content", "")[:300], "sentiment": st.score, "events": st.events,
                         "picks": picks_hit})
    # ---- 组装 30 只
    picks = []
    for c in codes:
        b = base[c]
        # 只使用「数据日期当天或之后」的海外收盘：收盘后（19:23）最新的美股是前一晚的，已经反映在今天 A 股收盘价里
        pred_ex, chains = OV.chain_signals(c, quotes, stats, ov_cfg.get("overseas_min_t", 2.0), b.get("industry"),
                                           min_date=latest["data_date"])
        ov_pts = float(np.clip(pred_ex * ov_cfg.get("overseas_points_per_pct", 8.0),
                               -ov_cfg.get("overseas_cap", 10.0), ov_cfg.get("overseas_cap", 10.0)))
        items = matched.get(c, [])
        ns = aggregate(items)
        news_pts = float(np.clip(ns * ov_cfg.get("news_points", 6.0), -6, 6)) if items else 0.0
        short = None if b["short_model"] is None else float(np.clip(b["short_model"] + ov_pts + news_pts, 0, 100))
        val = b["value"]
        total = None if val is None or short is None else w["value"] * val + w["short"] * short
        picks.append({**b, "short": clean(short, 1), "total": clean(total, 1), "chains": chains,
                      "signals": {"overseas": round(ov_pts / 4, 2), "overseas_pred_pct": round(pred_ex, 3),
                                  "news": ns, "news_points": round(news_pts, 1), "overseas_points": round(ov_pts, 1)},
                      "news": [{k: x.get(k) for k in ("time", "source", "title", "url", "sentiment", "events", "match")}
                               for x in items[:6]]})
    ranked = sorted(picks, key=lambda x: -(x["total"] or -1))
    hist_file = ctx.state / f"daily_{latest['month']}.json"
    hist = read_json(hist_file, {})
    prev_dates = sorted(k for k in hist if (k < latest["data_date"] if mode == "close" else k <= latest["data_date"]))
    prev = hist[prev_dates[-1]] if prev_dates else {}
    for i, p in enumerate(ranked, 1):
        p["rank"] = i
        p["prev_rank"] = (prev.get(p["code"]) or [None] * 4)[3]
    if mode == "close":
        hist[latest["data_date"]] = {p["code"]: [p["total"], p["value"], p["short"], p["rank"]] for p in ranked}
        write_json(hist_file, hist)
    history = {c: [[dte, *(hist[dte].get(c) or [None] * 4)[:3]] for dte in sorted(hist)] for c in codes}
    # ---- 市场
    market = {"indices": [], "global": []}
    try:
        idx = S.tencent_index([c for c, _ in INDEX_CODES])
        nm = dict(INDEX_CODES)
        market["indices"] = [{"code": x["code"], "name": nm.get(x["code"], x["name"]), "price": x["price"],
                              "pct": (x["pct"] or 0) / 100} for x in idx]
        ctx.note("指数行情", len(idx) > 0)
    except Exception as e:  # noqa: BLE001
        ctx.note("指数行情", False, str(e))
    for sym, nm in OV.DISPLAY.items():
        q = quotes.get(sym) or {}
        if q.get("ok"):
            market["global"].append({"symbol": sym, "name": nm, "price": q["price"], "pct": q["pct"], "date": q["date"]})
    rates = margin = None
    try:
        rates = S.em_treasury_yield(pages=1)
        r = rates.dropna(subset=["us10y"])
        market["global"].append({"symbol": "US10Y", "name": "美债10Y(%)", "price": float(r["us10y"].iloc[-1]),
                                 "pct": None, "bp": round(float(r["us10y"].iloc[-1] - r["us10y"].iloc[-2]) * 100, 1)})
        rc = rates.dropna(subset=["cn10y"])
        market["global"].append({"symbol": "CN10Y", "name": "中债10Y(%)", "price": float(rc["cn10y"].iloc[-1]),
                                 "pct": None, "bp": round(float(rc["cn10y"].iloc[-1] - rc["cn10y"].iloc[-2]) * 100, 1)})
        ctx.note("中美国债收益率", True)
    except Exception as e:  # noqa: BLE001
        ctx.note("中美国债收益率", False, str(e))
    try:
        margin = S.em_margin_daily(pages=1)
        ctx.note("两融余额", len(margin) > 0)
    except Exception as e:  # noqa: BLE001
        ctx.note("两融余额", False, str(e))
    breadth = None
    if mode == "close":
        try:
            dd = latest["data_date"].replace("-", "")
            breadth = {"zt": len(S.em_limit_pool("zt", dd)), "dt": len(S.em_limit_pool("dt", dd)),
                       "zb": len(S.em_limit_pool("zb", dd))}
            ctx.note("涨跌停池", True)
        except Exception as e:  # noqa: BLE001
            ctx.note("涨跌停池", False, str(e))
            breadth = latest.get("breadth")
        latest["breadth"] = breadth
        write_json(ctx.state / "latest.json", latest)
    else:
        breadth = latest.get("breadth")
    market["breadth"] = breadth
    mchg = None
    if margin is not None and len(margin) > 6:
        m = margin.dropna(subset=["fin_balance"])
        mchg = float(m["fin_balance"].iloc[-1] / m["fin_balance"].iloc[-6] - 1)
    market["regime"] = W.regime(market["indices"], breadth, latest.get("ew_ret20"), mchg)
    market["time"] = now_bj().strftime("%Y-%m-%d %H:%M")
    # ---- 观察清单
    ev_cfg = W.load_events(ROOT / "config" / "events.json")
    today = now_bj().date()
    watch = {"signals": W.live_signals(quotes, ranked, rates, margin, breadth),
             "events": W.upcoming_events(ev_cfg, ranked, today, s.get("watch_days_ahead", 21)),
             "themes": W.themes(ev_cfg, ranked)}
    dash = {
        "generated_at": now_bj().strftime("%Y-%m-%d %H:%M"), "mode": mode, "data_date": latest["data_date"],
        "month": latest["month"], "pick_date": latest["pick_date"], "universe_n": latest.get("universe_n"),
        "picks_note": latest.get("picks_note", ""),
        "market": market, "picks": ranked, "history": history, "watch": watch, "news": news_out,
        "track": track, "sources": ctx.sources, "method": METHOD_HTML,
        "factor_labels": FACTOR_LABELS,
    }
    write_json(ctx.out / "dashboard.json", dash)
    (ctx.state / "run_log.txt").write_text("\n".join(ctx.log))
    return dash


METHOD_HTML = """
<p><b>股票池</b>：沪深主板、创业板、科创板；剔除 ST/*ST/退市整理、上市不满约 1 年、近 60 日交易不足 80%、日均成交额低于 2000 万元或处于全市场最低 15%、股价低于 2 元的股票。</p>
<p><b>价值投资分（中期，约 1 个月）</b>：「尾部加权合成」。逐个因子统计其「全市场前 5% 股票」在随后 20 个交易日的超额收益，每月只用过去 36 个月标签已实现的数据，挑出稳定为正的因子（t 值 ≥ 0.5）并按 t 值加权。目前入选的是日内未追高、非流动性溢价、小市值、低于 20 日均线、量价背离、20 日回调、换手未异常放大。选股时还要求公司盈利、且 PE 低于盈利公司 PE 的 70 分位（排除盈利公司里最贵的 30%）。</p>
<p><b>短线分（约 1 周）</b>：LightGBM 模型，用 25 个量价与估值因子预测未来 5 个交易日的相对收益。在此基础上叠加两项实时调整：①海外联动——所在产业链（或所属行业）的海外标的隔夜涨跌 × 历史回归系数，只使用 2016–2022 年样本内显著、且 2023 年后样本外检验有效的配对，最多 ±10 分；②新闻情绪——快讯的词典法情绪，最多 ±6 分（未经回测，仅作提示）。</p>
<p><b>综合分</b> = 0.6 × 价值投资分 + 0.4 × 短线分。每月最后一个交易日收盘后选出下月 30 只（同一申万一级行业最多 5 只；上月持仓若仍在前 430 名则保留，以降低换手），整月固定；月内每天重新打分、重新排名。</p>
<p><b>回测</b>（扣交易成本，一字涨跌停无法成交）：回看期、估值约束、权重、缓冲这几个参数只用 2016–2024 年数据挑选；2025-01 至今是完全没有参与选择的「保留期」：年化 19.1%，股票池等权基准 11.4%，年化超额 +7.7%，信息比 0.70，72 组参数中 97% 在保留期跑赢基准。2016 年至今年化超额 +11.8%（含选参期，偏乐观）。最大回撤 −47%（不择时，大盘下跌时会一起下跌）。完整报告见仓库 docs/方法与回测报告.md。</p>
<p><b>局限</b>：入选股偏小市值、成交不活跃；不做择时，大盘下跌时会一起下跌；2020 年以前没有估值数据；行业分类为当前快照；新闻情绪与事件影响为规则判断。</p>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["close", "premarket"], default="close")
    ap.add_argument("--out", default=str(ROOT / "site_data"))
    args = ap.parse_args()
    try:
        run(args.mode, args.out)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
