"""离线端到端测试：把所有网络数据源替换为本地数据，完整跑一遍流水线。

用法（开发环境）：
  python tests/offline_harness.py --qlib data_raw/qlib --basic-panels data/panels \
      --xfer data/xfer/research_out --models models --out /tmp/site_test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ashare.live import sources as S  # noqa: E402
from ashare.macro import overseas as OV  # noqa: E402
from ashare.pipeline import daily as D  # noqa: E402


def install_mocks(qlib: Path, panels: Path, xfer: Path):
    ind = pd.read_csv(xfer.parent.parent / "industry.csv") if (xfer.parent.parent / "industry.csv").exists() else None
    names = dict(zip(ind["code"], ind["name"])) if ind is not None else {}

    D.ensure_qlib_data = lambda *_a, **_k: qlib

    def snapshot(codes):
        last = {}
        for f in ["pe_ttm", "pb", "total_mv", "circ_mv", "turnover_rate"]:
            df = pd.read_parquet(panels / f"{f}.parquet")
            last[f] = df.ffill().iloc[-1]
        close = pd.read_parquet(panels / "close.parquet").iloc[-1]
        factor = pd.read_parquet(panels / "factor.parquet").iloc[-1]
        price = close / factor
        df = pd.DataFrame({"name": [names.get(c, c) for c in codes]}, index=codes)
        df["price"] = price.reindex(codes)
        df["pe_ttm"] = last["pe_ttm"].reindex(codes)
        df["pb"] = last["pb"].reindex(codes)
        df["total_mv"] = last["total_mv"].reindex(codes) * 1e4
        df["float_mv"] = last["circ_mv"].reindex(codes) * 1e4
        df["turnover"] = last["turnover_rate"].reindex(codes)
        df["time"] = "20200101150000"
        # 模拟真实行情：数值列为 Python 对象、夹杂 None（GitHub Actions 首次运行就是在这里出错的）
        df = df.astype(object)
        df.iloc[::50, df.columns.get_loc("pe_ttm")] = None
        df.iloc[::70, df.columns.get_loc("price")] = None
        rows = df.reset_index().rename(columns={"index": "code"}).to_dict(orient="records")
        orig = S.tencent_quotes
        S.tencent_quotes = lambda _codes: rows
        try:
            return _real_snapshot(codes)
        finally:
            S.tencent_quotes = orig

    from ashare.pipeline.snapshot import market_snapshot as _real_snapshot
    D.market_snapshot = snapshot

    def perf(rd, max_pages=20):
        fp = xfer / f"perf_{rd}.csv"
        if not fp.exists():
            fp = xfer / "perf_2026-09-30.csv"
        return pd.read_csv(fp, dtype={"SECURITY_CODE": str})

    S.em_performance = perf

    def news(n=60):
        out = []
        sample = list(names.items())[:400:7]
        tmpl = ["{}三季度净利润大幅增长，超预期", "{}收到监管问询函", "{}中标重大项目订单", "{}股东拟减持计划",
                "美联储加息预期升温，美债收益率走强", "英伟达股价大涨，算力链受益"]
        for i, (c, nm) in enumerate(sample[:40]):
            t = tmpl[i % len(tmpl)].format(nm)
            out.append({"title": t, "content": t, "time": f"2026-11-03 1{i%10}:00:00", "source": "测试", "url": ""})
        return out

    S.all_news = news

    def overseas(symbols, days=90):
        out = {}
        g = xfer / "global"
        for sym in symbols:
            fp = g / f"{sym.replace('^', 'IDX_').replace('=', '_').replace('.', '_')}.csv"
            if not fp.exists():
                continue
            df = pd.read_csv(fp, parse_dates=["date"]).dropna(subset=["close"])
            if len(df) < 70:
                continue
            r = df["close"].pct_change()
            out[sym] = {"ok": True, "date": str(df["date"].iloc[-1].date()), "price": float(df["close"].iloc[-1]),
                        "pct": float(r.iloc[-1]), "z": float(r.iloc[-1] / r.tail(60).std()),
                        "ret5": float(df["close"].iloc[-1] / df["close"].iloc[-6] - 1)}
        return out

    OV.fetch_overseas = overseas
    S.tencent_index = lambda codes: [{"code": c, "name": c, "price": 3600.0 + i, "pct": 0.5 - i * 0.3}
                                     for i, c in enumerate(codes)]
    S.em_treasury_yield = lambda pages=1: pd.read_csv(xfer / "treasury.csv", parse_dates=["date"])
    S.em_margin_daily = lambda pages=1: pd.read_csv(xfer / "margin.csv", parse_dates=["date"]).sort_values("date")
    S.em_limit_pool = lambda kind="zt", date=None: [{}] * {"zt": 48, "dt": 12, "zb": 15}[kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlib", required=True)
    ap.add_argument("--basic-panels", required=True)
    ap.add_argument("--xfer", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="/tmp/site_test")
    ap.add_argument("--mode", default="close")
    a = ap.parse_args()
    install_mocks(Path(a.qlib), Path(a.basic_panels), Path(a.xfer))
    dash = D.run(a.mode, a.out, a.models)
    print("picks:", len(dash["picks"]), "events:", len(dash["watch"]["events"]),
          "signals:", len(dash["watch"]["signals"]), "news:", len(dash["news"]))
    for p in dash["picks"][:10]:
        print(p["rank"], p["code"], p["name"], p["industry"], p["total"], p["value"], p["short"],
              p["signals"], [r["name"] for r in p["reasons_value"][:3]])
    print("sources:", dash["sources"])


if __name__ == "__main__":
    main()
