"""研究：黄金/白银是否「领先」A股？中期走势是否相似？（只做研究，不进选股打分）

需要能访问国内行情源（新浪、东财、腾讯），请在本地 Mac 上运行：
    python scripts/gold_lead_study.py --out research_out/gold
输出：
    research_out/gold/results.json   —— 只含统计结果（相关系数、样本数、分段结果），可提交
    research_out/gold/raw/*.csv      —— 原始行情，仅本地使用，不要提交（.gitignore 已忽略 research_out/）

方法要点（避免事后解读）：
- 中期：用「收益率」而不是「价格」算相关（两条都上涨的价格线天然高度相关，没有意义）；
  按 2016–2019 / 2020–2022 / 2023–至今 分段；领先检验用 2016–2022 定阈值（样本内），2023 年起检验（样本外）。
- 日内：5 分钟收益的交叉相关（黄金在前 L 根 K 线 vs 指数在后），再做「黄金放量异动」事件研究；
  前一半天数为样本内、后一半为样本外。
- 复盘今天（或 --day 指定日期）1 分钟线：谁先动、隔了多久。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
S = requests.Session()
S.headers.update({"User-Agent": UA, "Referer": "https://finance.sina.com.cn/"})
LOG: list[str] = []

INDEXES = {"上证指数": ("sh000001", "1.000001"), "沪深300": ("sh000300", "1.000300"),
           "创业板指": ("sz399006", "0.399006"), "科创50": ("sh000688", "1.000688")}
METALS = {"沪金": ("AU0", "113.aum"), "沪银": ("AG0", "113.agm")}
COMEX = {"COMEX黄金": "GC", "COMEX白银": "SI"}


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def get(url: str, params: dict | None = None, tries: int = 3) -> str:
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=15)
            r.raise_for_status()
            time.sleep(0.6)
            return r.text
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def _jsonp_list(txt: str) -> list:
    return json.loads(txt[txt.find("["): txt.rfind("]") + 1])


# ------------------------------------------------------------------ 取数（多个来源，依次尝试）
def em_kline(secid: str, klt: int, lmt: int = 10000) -> pd.DataFrame:
    """东财 K 线。klt: 1/5 分钟，101 日线。"""
    txt = get("https://push2his.eastmoney.com/api/qt/stock/kline/get",
              {"secid": secid, "klt": klt, "fqt": 0, "lmt": lmt, "end": "20500101", "beg": "0",
               "fields1": "f1,f2,f3", "fields2": "f51,f52,f53,f54,f55,f56,f57"})
    rows = ((json.loads(txt).get("data") or {}).get("klines")) or []
    df = pd.DataFrame([r.split(",")[:7] for r in rows],
                      columns=["dt", "open", "close", "high", "low", "volume", "amount"])
    return _clean(df)


def sina_index_min(code: str, scale: int = 5) -> pd.DataFrame:
    txt = get("https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_x=/CN_MarketDataService.getKLineData",
              {"symbol": code, "scale": scale, "ma": "no", "datalen": 1970})
    df = pd.DataFrame(_jsonp_list(txt)).rename(columns={"day": "dt"})
    return _clean(df)


def tencent_index_day(code: str) -> pd.DataFrame:
    parts = []
    for y in range(2015, datetime.now().year + 1, 2):
        txt = get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                  {"param": f"{code},day,{y}-01-01,{y+1}-12-31,640,"})
        node = json.loads(txt).get("data", {}).get(code, {})
        rows = node.get("day") or node.get("qfqday") or []
        parts.append(pd.DataFrame([x[:6] for x in rows], columns=["dt", "open", "close", "high", "low", "volume"]))
    return _clean(pd.concat(parts, ignore_index=True))


def sina_inner_day(sym: str) -> pd.DataFrame:
    txt = get(f"https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_{sym}=/"
              "InnerFuturesNewService.getDailyKLine", {"symbol": sym})
    df = pd.DataFrame(_jsonp_list(txt)).rename(columns={"d": "dt", "o": "open", "h": "high", "l": "low",
                                                         "c": "close", "v": "volume"})
    return _clean(df)


def sina_inner_min(sym: str, typ: int = 5) -> pd.DataFrame:
    txt = get("https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_x=/"
              "InnerFuturesNewService.getFewMinLine", {"symbol": sym, "type": typ})
    df = pd.DataFrame(_jsonp_list(txt)).rename(columns={"d": "dt", "o": "open", "h": "high", "l": "low",
                                                         "c": "close", "v": "volume"})
    return _clean(df)


def sina_global_day(sym: str) -> pd.DataFrame:
    today = datetime.today().strftime("%Y_%m_%d")
    txt = get(f"https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_S{today}=/"
              "GlobalFuturesService.getGlobalFuturesDailyKLine", {"symbol": sym, "_": today, "source": "web"})
    df = pd.DataFrame(_jsonp_list(txt)).rename(columns={"date": "dt"})
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("empty")
    df["dt"] = pd.to_datetime(df["dt"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)[
        [c for c in ["dt", "open", "high", "low", "close", "volume"] if c in df.columns]]


def first_ok(name: str, fns: list) -> pd.DataFrame | None:
    for label, fn in fns:
        try:
            df = fn()
            log(f"[OK]   {name} ← {label}: {len(df)} 条，{df['dt'].iloc[0]} ~ {df['dt'].iloc[-1]}")
            return df
        except Exception as e:  # noqa: BLE001
            log(f"[FAIL] {name} ← {label}: {type(e).__name__} {str(e)[:120]}")
    return None


def fetch_all(raw: Path) -> dict:
    raw.mkdir(parents=True, exist_ok=True)
    data: dict = {"day": {}, "m5": {}, "m1": {}}
    for nm, (sina, em) in INDEXES.items():
        data["day"][nm] = first_ok(f"{nm} 日线", [("东财", lambda em=em: em_kline(em, 101)),
                                                ("腾讯", lambda sina=sina: tencent_index_day(sina))])
        data["m5"][nm] = first_ok(f"{nm} 5分钟", [("东财", lambda em=em: em_kline(em, 5)),
                                                 ("新浪", lambda sina=sina: sina_index_min(sina, 5))])
        data["m1"][nm] = first_ok(f"{nm} 1分钟", [("东财", lambda em=em: em_kline(em, 1))])
    for nm, (sina, em) in METALS.items():
        data["day"][nm] = first_ok(f"{nm} 日线", [("新浪", lambda s=sina: sina_inner_day(s)),
                                                ("东财", lambda em=em: em_kline(em, 101))])
        data["m5"][nm] = first_ok(f"{nm} 5分钟", [("东财", lambda em=em: em_kline(em, 5)),
                                                 ("新浪", lambda s=sina: sina_inner_min(s, 5))])
        data["m1"][nm] = first_ok(f"{nm} 1分钟", [("东财", lambda em=em: em_kline(em, 1)),
                                                 ("新浪", lambda s=sina: sina_inner_min(s, 1))])
    for nm, sym in COMEX.items():
        data["day"][nm] = first_ok(f"{nm} 日线", [("新浪外盘", lambda s=sym: sina_global_day(s))])
    for kind, d in data.items():
        for nm, df in d.items():
            if df is not None:
                df.to_csv(raw / f"{kind}_{nm}.csv", index=False)
    return data


def load_raw(raw: Path) -> dict:
    data: dict = {"day": {}, "m5": {}, "m1": {}}
    for f in raw.glob("*.csv"):
        kind, nm = f.stem.split("_", 1)
        data[kind][nm] = pd.read_csv(f, parse_dates=["dt"])
    return data


# ------------------------------------------------------------------ 统计工具
def corr(a, b) -> float | None:
    m = pd.concat([pd.Series(a), pd.Series(b)], axis=1).dropna()
    if len(m) < 10 or m.iloc[:, 0].std() == 0 or m.iloc[:, 1].std() == 0:
        return None
    return round(float(m.corr().iloc[0, 1]), 3)


def tstat_mean(x) -> tuple[float | None, float | None, int]:
    x = pd.Series(x).dropna()
    n = len(x)
    if n < 5:
        return None, None, n
    return round(float(x.mean()) * 100, 3), round(float(x.mean() / (x.std() / math.sqrt(n))), 2), n


PERIODS = [("2016-2019", "2016-01-01", "2019-12-31"), ("2020-2022", "2020-01-01", "2022-12-31"),
           ("2023-至今", "2023-01-01", "2100-01-01"), ("近3年", None, None)]


def daily_study(day: dict) -> dict:
    close = {}
    for nm, df in day.items():
        if df is not None:
            close[nm] = df.set_index(pd.to_datetime(df["dt"]).dt.normalize())["close"]
    if "上证指数" not in close:
        return {"error": "缺少指数日线"}
    cal = close["上证指数"].index
    px = pd.DataFrame(index=cal)
    for nm, s in close.items():
        s = s[~s.index.duplicated()]
        if nm.startswith("COMEX"):
            # 美盘收盘在北京时间次日清晨，A股当天开盘前已知 → 对齐为「A股日期 t 用 t-1 及以前最后一个美盘收盘」
            s = s.copy()
            s.index = s.index + pd.Timedelta(days=1)
            px[nm] = s.reindex(cal, method="ffill")
        else:
            px[nm] = s.reindex(cal)
    px = px[px.index >= "2015-06-01"].ffill(limit=3)
    out: dict = {"sample": f"{px.index[0].date()} ~ {px.index[-1].date()}", "contemporaneous": {}, "lead": {}}
    idx_names = [n for n in INDEXES if n in px]
    gold_names = [n for n in ["沪金", "沪银", "COMEX黄金", "COMEX白银"] if n in px]
    last3 = px.index[-1] - pd.DateOffset(years=3)

    # 1) 同期相关：日 / 周(5日) / 月(20日，不重叠) 收益
    for g in gold_names:
        for i in idx_names:
            row = {}
            for pname, a, b in PERIODS:
                sub = px[(px.index >= (last3 if a is None else a)) & (px.index <= (px.index[-1] if b is None else b))]
                r = {}
                for h, lab in [(1, "日"), (5, "周"), (20, "月")]:
                    ss = sub.iloc[::h]
                    r[lab] = corr(np.log(ss[g]).diff(), np.log(ss[i]).diff())
                row[pname] = r
            # 价格水平相关（仅展示，提醒没有意义）
            sub = px[px.index >= last3]
            row["近3年价格水平相关(参考,易误导)"] = corr(sub[g], sub[i])
            # 滚动 120 日收益相关的最新值与历史分位
            rc = np.log(px[g]).diff().rolling(120).corr(np.log(px[i]).diff())
            row["滚动120日相关_最新"] = None if pd.isna(rc.iloc[-1]) else round(float(rc.iloc[-1]), 3)
            row["滚动120日相关_历史中位数"] = round(float(rc.median()), 3)
            row["滚动120日相关_近3年每季末"] = {str(k.date()): round(float(v), 3)
                                         for k, v in rc[rc.index >= last3].resample("QE").last().dropna().items()}
            out["contemporaneous"][f"{g}~{i}"] = row

    # 2) 领先检验：黄金过去 k 日收益 → 指数未来 h 日收益。样本内 2016–2022，样本外 2023 起。
    IS_END = pd.Timestamp("2022-12-31")
    for g in gold_names:
        for i in idx_names:
            res = {}
            for k in [1, 5, 20]:
                past = np.log(px[g]).diff(k)
                for h in [1, 5, 20]:
                    fwd = np.log(px[i]).shift(-h) - np.log(px[i])
                    df = pd.DataFrame({"x": past, "y": fwd}).dropna()
                    ins, oos = df[df.index <= IS_END], df[df.index > IS_END]
                    step = max(h, 1)  # 用不重叠样本算相关，避免重叠收益夸大显著性
                    thr = ins["x"].quantile(0.8)
                    item = {"IS相关": corr(ins["x"].iloc[::step], ins["y"].iloc[::step]),
                            "OOS相关": corr(oos["x"].iloc[::step], oos["y"].iloc[::step]),
                            "OOS样本数": int(len(oos.iloc[::step]))}
                    for lab, part in [("IS", ins), ("OOS", oos)]:
                        hit = part[part["x"] > thr].iloc[::step]["y"]
                        base = part.iloc[::step]["y"]
                        m, t, n = tstat_mean(hit)
                        item[f"{lab}_黄金大涨后指数平均(%)"] = m
                        item[f"{lab}_t值"] = t
                        item[f"{lab}_次数"] = n
                        item[f"{lab}_全部平均(%)"] = tstat_mean(base)[0]
                    res[f"黄金过去{k}日→指数未来{h}日"] = item
            out["lead"][f"{g}→{i}"] = res

    # 3) 隔夜：COMEX 前一晚涨跌 → A股当天开盘跳空 / 当天收益
    if "COMEX黄金" in px:
        g = np.log(px["COMEX黄金"]).diff()
        for i in idx_names:
            y = np.log(px[i]).diff()
            out.setdefault("overnight", {})[f"COMEX黄金隔夜→{i}当日"] = {
                "IS相关": corr(g[g.index <= IS_END], y[y.index <= IS_END]),
                "OOS相关": corr(g[g.index > IS_END], y[y.index > IS_END])}
    # 网站可用的走势对比（月度归一化，近 5 年）
    sub = px[px.index >= px.index[-1] - pd.DateOffset(years=5)].resample("ME").last()
    out["normalized_monthly"] = {c: [round(float(v / sub[c].dropna().iloc[0] * 100), 1) if pd.notna(v) else None
                                     for v in sub[c]] for c in sub.columns}
    out["normalized_monthly"]["dates"] = [str(d.date()) for d in sub.index]
    return out


def _session_returns(df: pd.DataFrame) -> pd.Series:
    """只保留 A 股和上期所都在交易的白天时段的 K 线收益（不跨午休、不跨日、去掉每天第一根）。"""
    d = df.copy()
    d["dt"] = pd.to_datetime(d["dt"])
    t = d["dt"].dt.strftime("%H:%M")
    day_session = ((t > "09:30") & (t <= "10:15")) | ((t > "10:30") & (t <= "11:30")) | ((t > "13:30") & (t <= "15:00"))
    d = d.set_index("dt")
    r = np.log(d["close"]).diff()
    gap = d.index.to_series().diff() > pd.Timedelta(minutes=16)  # 跨休市的那一根不要
    r[gap.values] = np.nan
    return r[day_session.values]


def _volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    d = df.set_index(pd.to_datetime(df["dt"]))
    return d["volume"] / d["volume"].rolling(n, min_periods=10).mean().shift(1)


def intraday_study(m5: dict, bar_min: int = 5) -> dict:
    idx_names = [n for n in INDEXES if m5.get(n) is not None]
    gold_names = [n for n in METALS if m5.get(n) is not None]
    if not idx_names or not gold_names:
        return {"error": "缺少分钟数据"}
    out: dict = {"bar_minutes": bar_min, "xcorr": {}, "events": {}}
    for g in gold_names:
        rg = _session_returns(m5[g])
        vr = _volume_ratio(m5[g])
        for i in idx_names:
            ri = _session_returns(m5[i])
            both = pd.concat({"g": rg, "i": ri}, axis=1).dropna()
            if len(both) < 100:
                continue
            days = sorted(set(both.index.date))
            half = days[len(days) // 2]
            res = {"样本": f"{days[0]} ~ {days[-1]}，{len(days)} 天，{len(both)} 根",
                   "显著性参考线(约2/√N)": round(2 / math.sqrt(len(both)), 3)}
            # 交叉相关：L>0 表示黄金领先指数 L 根；只在同一天内错位
            full_g = rg.reindex(ri.index)
            lag_tab = {}
            for L in range(-6, 7):
                shifted = full_g.groupby(full_g.index.date).shift(L)
                m = pd.concat({"g": shifted, "i": ri}, axis=1).dropna()
                a = m[m.index.date < half]
                b = m[m.index.date >= half]
                lag_tab[str(L * bar_min)] = {"全部": corr(m["g"], m["i"]), "前半": corr(a["g"], a["i"]),
                                            "后半": corr(b["g"], b["i"])}
            res["交叉相关(分钟:正数=黄金先动)"] = lag_tab
            out["xcorr"][f"{g}→{i}"] = res

            # 事件：黄金 5 分钟涨跌超过 2.5 倍标准差 且 成交量 ≥ 2 倍近期均量
            sd = rg.groupby(rg.index.date).transform(lambda s: s.std())
            sd = sd.rolling(5 * 50, min_periods=50).median().shift(1).fillna(rg.std())
            ev = rg[(rg.abs() > 2.5 * sd) & (vr.reindex(rg.index) >= 2.0)]
            rows = []
            ci = np.log(m5[i].set_index(pd.to_datetime(m5[i]["dt"]))["close"])
            for ts, rv in ev.items():
                if ts not in ci.index:
                    continue
                pos = ci.index.get_loc(ts)
                fw = {}
                for k in [1, 2, 3, 6]:
                    if pos + k < len(ci) and ci.index[pos + k].date() == ts.date():
                        fw[k] = float(ci.iloc[pos + k] - ci.iloc[pos]) * np.sign(rv)
                rows.append({"ts": ts, "dir": int(np.sign(rv)), **{f"f{k}": fw.get(k) for k in [1, 2, 3, 6]}})
            evdf = pd.DataFrame(rows)
            item = {"事件数": int(len(evdf))}
            if len(evdf):
                for k in [1, 2, 3, 6]:
                    for lab, part in [("前半", evdf[evdf["ts"].dt.date < half]), ("后半", evdf[evdf["ts"].dt.date >= half])]:
                        m_, t_, n_ = tstat_mean(part[f"f{k}"])
                        item[f"{lab}_之后{k*bar_min}分钟同向收益(%)"] = m_
                        item[f"{lab}_之后{k*bar_min}分钟_t值"] = t_
                        item[f"{lab}_次数"] = n_
                item["最近10次"] = [f"{r.ts:%m-%d %H:%M} {'涨' if r.dir > 0 else '跌'}" for r in evdf.tail(10).itertuples()]
            out["events"][f"{g}→{i}"] = item
    return out


def replay_day(m1: dict, day: str | None) -> dict:
    """复盘某一天 1 分钟线：每个品种相对开盘的涨幅，以及第一次「明显异动」的时间。"""
    out: dict = {}
    names = [n for n in list(METALS) + list(INDEXES) if m1.get(n) is not None]
    if not names:
        return {"error": "缺少1分钟数据"}
    if day is None:
        day = str(pd.to_datetime(m1[names[0]]["dt"]).dt.date.max())
    out["day"] = day
    for nm in names:
        d = m1[nm].copy()
        d["dt"] = pd.to_datetime(d["dt"])
        d = d[(d["dt"].dt.date.astype(str) == day) & (d["dt"].dt.strftime("%H:%M") >= "09:00")]
        if nm in METALS:  # 上期所白天盘 9:00 开，A 股 9:30 开，统一从 9:30 起比较
            d = d[d["dt"].dt.strftime("%H:%M") >= "09:30"]
        if len(d) < 10:
            continue
        r = np.log(d["close"]).diff()
        sd = r.std()
        vr = d["volume"] / d["volume"].rolling(15, min_periods=5).mean().shift(1)
        jump = d[(r.abs() > 3 * sd) & (vr > 2)]
        cum = (d["close"] / d["close"].iloc[0] - 1) * 100
        out[nm] = {"首次放量异动": [f"{t:%H:%M} {'↑' if rr > 0 else '↓'}" for t, rr in zip(jump["dt"][:5], r[jump.index][:5])],
                   "每15分钟累计涨跌(%)": {f"{t:%H:%M}": round(float(v), 2) for t, v in zip(d["dt"][::15], cum[::15])},
                   "全天涨跌(%)": round(float(cum.iloc[-1]), 2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_out/gold")
    ap.add_argument("--offline", action="store_true", help="不联网，直接用 raw/ 里已下载的数据")
    ap.add_argument("--day", default=None, help="复盘日期，默认最新一天")
    args = ap.parse_args()
    out = Path(args.out)
    raw = out / "raw"
    data = load_raw(raw) if args.offline else fetch_all(raw)
    res = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "log": LOG}
    for key, fn in [("daily", lambda: daily_study(data["day"])),
                    ("intraday_5min", lambda: intraday_study(data["m5"], 5)),
                    ("intraday_1min", lambda: intraday_study(data["m1"], 1)),
                    ("replay", lambda: replay_day(data["m1"], args.day))]:
        try:
            res[key] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            res[key] = {"error": f"{type(e).__name__}: {e}"}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(f"写入 {out/'results.json'}")


if __name__ == "__main__":
    sys.exit(main())
