"""实时 / 日更数据源。所有函数都返回纯 Python 数据结构或 DataFrame，失败抛 SourceUnavailable。

股票代码内部统一用小写带市场前缀：sh600519 / sz000001。
接口细节出处：docs/资料汇编 第 1.2–1.5 节（腾讯、新浪、东财数据中心、push2ex、快讯）。
"""
from __future__ import annotations

import hashlib
import io
import re
import time
from datetime import datetime
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd

from .http import Fetcher, SourceUnavailable

_fetcher: Fetcher | None = None


def fetcher() -> Fetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = Fetcher()
    return _fetcher


def set_fetcher(f: Fetcher) -> None:
    global _fetcher
    _fetcher = f


def normalize_code(code: str) -> str:
    c = code.strip().lower().replace(".sh", "").replace(".sz", "").replace(".bj", "")
    if c[:2] in ("sh", "sz", "bj"):
        return c
    if c.startswith(("92", "8", "4")):
        return "bj" + c
    if c.startswith(("6", "9")):
        return "sh" + c
    return "sz" + c


def _f(x: str) -> float | None:
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 腾讯
def tencent_quotes(codes: Iterable[str], batch: int = 60) -> list[dict]:
    """腾讯 qt.gtimg.cn 实时行情（≤60 只 / 次，GBK）。"""
    codes = [normalize_code(c) for c in codes]
    out: list[dict] = []
    failed = 0
    for i in range(0, len(codes), batch):
        chunk = codes[i : i + batch]
        try:
            r = fetcher().get("https://qt.gtimg.cn/q=" + ",".join(chunk), encoding="gbk")
        except SourceUnavailable:
            failed += 1          # 单批失败不影响其他批次
            continue
        for line in r.text.strip().split(";"):
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            code = key.strip().replace("v_", "")
            f = val.strip().strip('"').split("~")
            if len(f) < 50:
                continue
            # 实测（2026-11-03）：科创板（688/689）腾讯返回的成交量单位是「股」，主板/创业板是「手」
            vol_div = 100.0 if code.startswith(("sh688", "sh689")) else 1.0
            out.append({
                "code": code,
                "name": f[1],
                "price": _f(f[3]),
                "prev_close": _f(f[4]),
                "open": _f(f[5]),
                "volume_lots": (_f(f[6]) or 0.0) / vol_div,
                "time": f[30],
                "pct": _f(f[32]),
                "high": _f(f[33]),
                "low": _f(f[34]),
                "amount": (_f(f[37]) or 0.0) * 1e4,  # 万元 → 元
                "turnover": _f(f[38]),
                "pe_ttm": _f(f[39]),
                "float_mv": (_f(f[44]) or 0.0) * 1e8,  # 亿元 → 元
                "total_mv": (_f(f[45]) or 0.0) * 1e8,
                "pb": _f(f[46]),
                "limit_up": _f(f[47]),
                "limit_down": _f(f[48]),
                "vol_ratio": _f(f[49]),
                "source": "tencent",
            })
    if failed and not out:
        raise SourceUnavailable(f"腾讯行情全部 {failed} 批失败")
    return out


def tencent_index(codes: Iterable[str]) -> list[dict]:
    """指数简版 q=s_sh000001：市场~名称~代码~最新价~涨跌额~涨跌幅~成交量~成交额(万)。"""
    q = ",".join("s_" + c for c in codes)
    r = fetcher().get("https://qt.gtimg.cn/q=" + q, encoding="gbk")
    out = []
    for line in r.text.strip().split(";"):
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        f = val.strip().strip('"').split("~")
        if len(f) < 6:
            continue
        out.append({"code": key.strip().replace("v_s_", ""), "name": f[1], "price": _f(f[3]),
                    "chg": _f(f[4]), "pct": _f(f[5]), "amount": (_f(f[7]) or 0) * 1e4})
    return out


def tencent_kline(code: str, count: int = 320, adj: str = "qfq", period: str = "day") -> pd.DataFrame:
    """腾讯日 K：每条 [日期,开,收,高,低,成交量(手)]。"""
    code = normalize_code(code)
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    r = fetcher().get(url, params={"param": f"{code},{period},,,{count},{adj}"})
    js = r.json()
    node = js.get("data", {}).get(code, {})
    rows = node.get(adj + period) or node.get(period) or []
    df = pd.DataFrame([x[:6] for x in rows], columns=["date", "open", "close", "high", "low", "volume"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if code.startswith(("sh688", "sh689")):
        df["volume"] = df["volume"] / 100.0      # 科创板：股 → 手
    return df


# ---------------------------------------------------------------- 新浪
SINA_HEADERS = {"Referer": "https://finance.sina.com.cn/"}


def sina_quotes(codes: Iterable[str], batch: int = 600) -> list[dict]:
    codes = [normalize_code(c) for c in codes]
    out = []
    for i in range(0, len(codes), batch):
        chunk = codes[i : i + batch]
        r = fetcher().get("https://hq.sinajs.cn/list=" + ",".join(chunk), headers=SINA_HEADERS,
                          encoding="gbk")
        for m in re.finditer(r'hq_str_(\w+)="([^"]*)"', r.text):
            code, body = m.group(1), m.group(2)
            f = body.split(",")
            if len(f) < 32:
                continue
            prev = _f(f[2])
            price = _f(f[3])
            out.append({
                "code": code, "name": f[0], "open": _f(f[1]), "prev_close": prev, "price": price,
                "high": _f(f[4]), "low": _f(f[5]), "volume_lots": (_f(f[8]) or 0) / 100,
                "amount": _f(f[9]), "time": f"{f[30]} {f[31]}",
                "pct": (price / prev - 1) * 100 if price and prev else None, "source": "sina",
            })
    return out


# ---------------------------------------------------------------- 东方财富数据中心
DC_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def em_datacenter(report: str, filter_: str = "", columns: str = "ALL", sort_columns: str = "",
                  sort_types: str = "", page_size: int = 500, max_pages: int = 40,
                  extra: dict | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    page = 1
    while page <= max_pages:
        params = {"reportName": report, "columns": columns, "pageSize": page_size,
                  "pageNumber": page, "source": "WEB", "client": "WEB"}
        if filter_:
            params["filter"] = filter_
        if sort_columns:
            params["sortColumns"] = sort_columns
            params["sortTypes"] = sort_types or "-1"
        if extra:
            params.update(extra)
        js = fetcher().get(DC_URL, params=params).json()
        res = js.get("result") or {}
        data = res.get("data") or []
        rows.extend(data)
        pages = int(res.get("pages") or 1)
        if page >= pages:
            break
        page += 1
    return pd.DataFrame(rows)


def em_performance(report_date: str, max_pages: int = 30) -> pd.DataFrame:
    """业绩报表（含公告日期 NOTICE_DATE、ROE、营收/净利同比、毛利率）。report_date: 'YYYY-MM-DD' 季末。"""
    return em_datacenter("RPT_LICO_FN_CPD", f"(REPORTDATE='{report_date}')",
                         sort_columns="UPDATE_DATE,SECURITY_CODE", sort_types="-1,-1",
                         max_pages=max_pages)


def em_forecast(report_date: str, max_pages: int = 20) -> pd.DataFrame:
    """业绩预告。"""
    url_df = []
    page = 1
    while page <= max_pages:
        params = {"reportName": "RPT_PUBLIC_OP_NEWPREDICT", "columns": "ALL", "pageSize": 500,
                  "pageNumber": page, "sortColumns": "NOTICE_DATE,SECURITY_CODE",
                  "sortTypes": "-1,-1", "filter": f"(REPORT_DATE='{report_date}')"}
        js = fetcher().get("https://datacenter.eastmoney.com/securities/api/data/v1/get",
                           params=params).json()
        res = js.get("result") or {}
        url_df.extend(res.get("data") or [])
        if page >= int(res.get("pages") or 1):
            break
        page += 1
    return pd.DataFrame(url_df)


def em_lift_schedule(start: str, end: str) -> pd.DataFrame:
    """限售解禁（个股，区间）。"""
    return em_datacenter("RPT_LIFT_STAGE", f"(FREE_DATE>='{start}')(FREE_DATE<='{end}')",
                         sort_columns="FREE_DATE", sort_types="1", max_pages=10)


def em_billboard(start: str, end: str) -> pd.DataFrame:
    """龙虎榜明细。"""
    return em_datacenter("RPT_DAILYBILLBOARD_DETAILSNEW",
                         f"(TRADE_DATE<='{end}')(TRADE_DATE>='{start}')",
                         sort_columns="SECURITY_CODE,TRADE_DATE", sort_types="1,-1",
                         page_size=500, max_pages=10)


# ---------------------------------------------------------------- 东财 push2ex 情绪
def em_limit_pool(kind: str = "zt", date: str | None = None) -> list[dict]:
    """涨停池 zt / 跌停 dt / 炸板 zb。单请求返回当日全部。"""
    name = {"zt": ("getTopicZTPool", "fbt:asc"), "dt": ("getTopicDTPool", "fund:asc"),
            "zb": ("getTopicZBPool", "fbt:asc")}[kind]
    date = date or datetime.now().strftime("%Y%m%d")
    url = f"https://push2ex.eastmoney.com/{name[0]}"
    params = {"ut": "7eea3edcaed734bea9cbfc24409ed989", "dpt": "wz.ztzt", "Pageindex": 0,
              "pagesize": 10000, "sort": name[1], "date": date}
    js = fetcher().get(url, params=params).json()
    return ((js.get("data") or {}).get("pool")) or []


def em_clist_page(page: int = 1, size: int = 100, fid: str = "f3") -> list[dict]:
    """东财全市场快照的单页（push2，低频使用）。"""
    params = {"pn": page, "pz": size, "po": 1, "np": 1, "fltt": 2, "invt": 2, "fid": fid,
              "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",
              "fields": "f12,f14,f2,f3,f5,f6,f8,f9,f10,f20,f21,f23,f62,f100",
              "ut": "bd1d9ddb04089700cf9c27f6f7426281"}
    js = fetcher().get("https://push2.eastmoney.com/api/qt/clist/get", params=params).json()
    return ((js.get("data") or {}).get("diff")) or []


# ---------------------------------------------------------------- 新闻快讯
def em_fast_news(size: int = 100) -> list[dict]:
    params = {"client": "web", "biz": "web_724", "fastColumn": "102", "sortEnd": "",
              "pageSize": size, "req_trace": int(time.time() * 1000)}
    js = fetcher().get("https://np-weblist.eastmoney.com/comm/web/getFastNewsList",
                       params=params).json()
    items = ((js.get("data") or {}).get("fastNewsList")) or []
    return [{"title": x.get("title", ""), "content": x.get("summary", ""),
             "time": x.get("showTime", ""), "source": "东方财富快讯",
             "url": f"https://finance.eastmoney.com/a/{x.get('code')}.html"} for x in items]


def sina_7x24(size: int = 50) -> list[dict]:
    params = {"page": 1, "page_size": size, "zhibo_id": 152, "tag_id": 0, "dire": "f",
              "dpc": 1, "pagesize": size, "type": 1}
    js = fetcher().get("https://zhibo.sina.com.cn/api/zhibo/feed", params=params).json()
    items = (((js.get("result") or {}).get("data") or {}).get("feed") or {}).get("list") or []
    out = []
    for x in items:
        text = re.sub(r"<[^>]+>", "", x.get("rich_text", ""))
        out.append({"title": text[:60], "content": text, "time": x.get("create_time", ""),
                    "source": "新浪7x24", "url": x.get("docurl") or ""})
    return out


def cls_telegraph(size: int = 50) -> list[dict]:
    p = {"app": "CailianpressWeb", "category": "", "last_time": int(time.time()), "os": "web",
         "refresh_type": "1", "rn": str(size), "sv": "8.4.6"}
    p["sign"] = hashlib.md5(hashlib.sha1(urlencode(p).encode()).hexdigest().encode()).hexdigest()
    js = fetcher().get("https://www.cls.cn/v1/roll/get_roll_list", params=p).json()
    items = ((js.get("data") or {}).get("roll_data")) or []
    return [{"title": x.get("title") or (x.get("content") or "")[:60],
             "content": x.get("content", ""),
             "time": datetime.fromtimestamp(int(x.get("ctime", 0))).strftime("%Y-%m-%d %H:%M:%S"),
             "source": "财联社", "important": x.get("level") in ("A", "B"),
             "url": x.get("shareurl") or ""} for x in items]


def all_news(size: int = 60) -> list[dict]:
    """合并多源快讯，任一源失败不影响其他源。"""
    news: list[dict] = []
    errors = {}
    for fn in (cls_telegraph, em_fast_news, sina_7x24):
        try:
            news.extend(fn(size))
        except Exception as e:  # noqa: BLE001
            errors[fn.__name__] = str(e)[:120]
    seen, uniq = set(), []
    for n in sorted(news, key=lambda x: x.get("time", ""), reverse=True):
        k = n["content"][:40]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    return uniq


# ---------------------------------------------------------------- 海外：Yahoo / FRED
def yahoo_chart(symbol: str, rng: str = "1y", interval: str = "1d") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    js = fetcher().get(url, params={"range": rng, "interval": interval}).json()
    res = (js.get("chart") or {}).get("result") or []
    if not res:
        raise SourceUnavailable(f"yahoo 无数据 {symbol}")
    r0 = res[0]
    ts = r0.get("timestamp") or []
    q = (r0.get("indicators") or {}).get("quote", [{}])[0]
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s"), "close": q.get("close"),
                       "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
                       "volume": q.get("volume")})
    df["date"] = df["date"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.normalize()
    df["date"] = df["date"].dt.tz_localize(None)
    return df.dropna(subset=["close"])


def fred_series(series_id: str) -> pd.DataFrame:
    r = fetcher().get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id})
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


# ---------------------------------------------------------------- 申万行业
SW_XLS = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"


def sw_industry_history() -> pd.DataFrame:
    """申万 2021 版个股行业归属历史（股票代码, 计入日期, 行业代码, 更新日期）。"""
    import requests

    try:
        r = requests.get(SW_XLS, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    except requests.exceptions.SSLError:
        # 申万官网证书链在部分环境不完整（资料汇编 1.0 / akshare 同样处理）。
        # 这里只下载一个公开的 xls 分类表，不发送任何凭据，因此仅对该文件放宽校验。
        import urllib3

        urllib3.disable_warnings()
        r = requests.get(SW_XLS, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, verify=False)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), dtype={"股票代码": str, "行业代码": str})
    return df


# ---------------------------------------------------------------- 利率 / 两融（东财数据中心，海外 IP 可用）
def em_treasury_yield(pages: int = 2) -> pd.DataFrame:
    """中美国债收益率（日度）：中国 2/5/10/30 年、美国 2/5/10/30 年。"""
    rows = []
    for page in range(1, pages + 1):
        params = {"type": "RPTA_WEB_TREASURYYIELD", "sty": "ALL", "st": "SOLAR_DATE", "sr": "-1",
                  "token": "894050c76af8597a853f5b408b759f5d", "p": page, "ps": 500,
                  "pageNo": page, "pageNum": page}
        js = fetcher().get("https://datacenter.eastmoney.com/api/data/get", params=params).json()
        rows.extend((js.get("result") or {}).get("data") or [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"SOLAR_DATE": "date", "EMM00588704": "cn2y", "EMM00166466": "cn10y",
                            "EMG00001306": "us2y", "EMG00001310": "us10y", "EMG00001312": "us30y"})
    keep = [c for c in ["date", "cn2y", "cn10y", "us2y", "us10y", "us30y"] if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def em_margin_daily(pages: int = 2) -> pd.DataFrame:
    """两融账户统计（日度）：融资余额、融券余额等。"""
    df = em_datacenter("RPTA_WEB_MARGIN_DAILYTRADE", sort_columns="STATISTICS_DATE",
                       sort_types="-1", page_size=500, max_pages=pages)
    if df.empty:
        return df
    df = df.rename(columns={"STATISTICS_DATE": "date", "FIN_BALANCE": "fin_balance",
                            "LOAN_BALANCE": "loan_balance"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------- 海外：腾讯美股 K 线 / 新浪外盘期货
US_SUFFIX = [".OQ", ".N", ".AM", ".PS", ""]
TENCENT_US_INDEX = {"^IXIC": "us.IXIC", "^GSPC": "us.INX", "^DJI": "us.DJI", "^NDX": "us.NDX"}
# 腾讯没有的指数用 ETF 代替（行情走势基本一致）
US_PROXY = {"^SOX": "SOXX", "^VIX": "VIXY", "^HSI": "EWH", "DX-Y.NYB": "UUP", "^TNX": "IEF"}
SINA_FUTURES = {"GC=F": "GC", "SI=F": "SI", "HG=F": "HG", "CL=F": "CL", "BZ=F": "OIL",
                "ZS=F": "S", "ZC=F": "C"}


def _tencent_us_try(code: str, start: str, end: str) -> pd.DataFrame:
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    r = fetcher().get(url, params={"_var": "kline_dayqfq", "param": f"{code},day,{start},{end},640,qfq"})
    txt = r.text
    js = __import__("json").loads(txt[txt.find("{"):])
    node = (js.get("data") or {}).get(code) or {}
    rows = node.get("qfqday") or node.get("day") or []
    return pd.DataFrame([x[:6] for x in rows], columns=["date", "open", "close", "high", "low", "volume"])


def tencent_us_kline(symbol: str, start: str = "2016-01-01", end: str | None = None) -> pd.DataFrame:
    """腾讯美股/美股指数日 K（前复权）。按年分段请求（每次最多 640 条）。"""
    end = end or datetime.now().strftime("%Y-%m-%d")
    symbol = US_PROXY.get(symbol, symbol)
    cands = [TENCENT_US_INDEX[symbol]] if symbol in TENCENT_US_INDEX else [f"us{symbol}{s}" for s in US_SUFFIX]
    for code in cands:
        try:
            probe = _tencent_us_try(code, (pd.Timestamp(end) - pd.Timedelta(days=30)).strftime("%Y-%m-%d"), end)
        except Exception:  # noqa: BLE001
            continue
        # 后缀不对时腾讯只返回 1 条「最新价」占位数据，因此要求至少 5 条
        if len(probe) < 5:
            continue
        if pd.Timestamp(start) >= pd.Timestamp(end) - pd.Timedelta(days=30):
            probe["date"] = pd.to_datetime(probe["date"])
            for c in ["open", "close", "high", "low", "volume"]:
                probe[c] = pd.to_numeric(probe[c], errors="coerce")
            probe.attrs["tencent_code"] = code
            return probe
        parts = []
        y0, y1 = pd.Timestamp(start).year, pd.Timestamp(end).year
        for y in range(y0, y1 + 1, 2):
            s = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")).strftime("%Y-%m-%d")
            e = min(pd.Timestamp(end), pd.Timestamp(f"{y+1}-12-31")).strftime("%Y-%m-%d")
            try:
                parts.append(_tencent_us_try(code, s, e))
            except Exception:  # noqa: BLE001
                pass
        df = pd.concat(parts, ignore_index=True) if parts else probe
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "close", "high", "low", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        df.attrs["tencent_code"] = code
        return df
    raise SourceUnavailable(f"腾讯美股无数据 {symbol}")


def sina_global_futures(symbol: str) -> pd.DataFrame:
    """新浪外盘期货日 K（GC 黄金、SI 白银、HG 铜、CL 原油、OIL 布伦特、S 大豆、C 玉米）。"""
    sym = SINA_FUTURES.get(symbol, symbol)
    today = f"{datetime.today().year}_{datetime.today().month}_{datetime.today().day}"
    url = (f"https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_S{today}=/"
           "GlobalFuturesService.getGlobalFuturesDailyKLine")
    r = fetcher().get(url, params={"symbol": sym, "_": today, "source": "web"})
    txt = r.text
    df = pd.read_json(io.StringIO(txt[txt.find("["): txt.rfind("]") + 1]))
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def overseas_daily(symbol: str, start: str = "2016-01-01") -> pd.DataFrame:
    """统一入口：期货走新浪，其余走腾讯；都失败再试 Yahoo（本地 Mac 网络下通常可用）。"""
    errors = []
    if symbol in SINA_FUTURES:
        try:
            df = sina_global_futures(symbol)
            return df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
    else:
        try:
            return tencent_us_kline(symbol, start)
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
    try:
        df = yahoo_chart(symbol, "10y")
        return df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        errors.append(str(e))
    raise SourceUnavailable(f"{symbol}: " + " | ".join(errors))
