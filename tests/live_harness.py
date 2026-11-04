"""本地实时服务的离线测试：用历史数据模拟腾讯日 K / 实时行情，跑一次 LiveEngine.refresh_once 并检查 API。

python tests/live_harness.py --site /tmp/site_test --panels data/panels --xfer data/xfer2/research_out
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ashare.live import realtime as RT  # noqa: E402
from ashare.live import sources as S  # noqa: E402
from ashare.server import app as APP  # noqa: E402
from offline_harness import install_mocks  # noqa: E402

BJ = timezone(timedelta(hours=8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--panels", required=True)
    ap.add_argument("--qlib", required=True)
    ap.add_argument("--xfer", required=True)
    a = ap.parse_args()
    panels = Path(a.panels)
    install_mocks(Path(a.qlib), panels, Path(a.xfer))
    close = pd.read_parquet(panels / "close.parquet")
    factor = pd.read_parquet(panels / "factor.parquet")
    p = {f: pd.read_parquet(panels / f"{f}.parquet") for f in ["open", "high", "low", "volume"]}
    raw = close / factor
    last_factor = factor.ffill().iloc[-1]

    def hist(codes, days=320):
        out = {}
        for c in codes:
            # 前复权：价格 / 最近复权因子；成交量（手）= volume × factor
            df = pd.DataFrame({"date": close.index,
                               "open": p["open"][c] / last_factor[c], "close": close[c] / last_factor[c],
                               "high": p["high"][c] / last_factor[c], "low": p["low"][c] / last_factor[c],
                               "volume": p["volume"][c] * factor[c]}).dropna().tail(days - 1)
            out[c] = df.reset_index(drop=True)
        return out

    def quotes(codes, batch=60):
        now = datetime.now(BJ)
        out = []
        for c in codes:
            px = float(raw[c].dropna().iloc[-1]) * 1.02          # 假设盘中上涨 2%
            out.append({"code": c, "name": c, "price": px, "prev_close": px / 1.02, "open": px / 1.01,
                        "high": px * 1.005, "low": px / 1.015, "volume_lots": 50000.0,
                        "time": now.strftime("%Y%m%d%H%M%S"), "pct": 2.0, "amount": px * 5e6,
                        "pe_ttm": 20.0, "pb": 2.0, "float_mv": px * 1e8, "total_mv": px * 2e8})
        return out

    RT.fetch_histories = hist
    S.tencent_quotes = quotes
    eng = APP.LiveEngine(Path(a.site), 120)
    eng.refresh_once()
    d = eng.dash
    assert d["mode"] == "live" and len(d["picks"]) == 30
    changed = sum(1 for x in d["picks"] if x.get("pct_today") == 0.02)
    print("live ok; picks with live quote:", changed)
    t = d.get("tech") or {}
    assert t and len(t["picks"]) == 15, "科技榜缺失"
    print("tech live ok:", [(x["rank"], x["code"], x["total"]) for x in t["picks"][:5]])
    for x in d["picks"][:5]:
        print(x["rank"], x["prev_rank"], x["code"], x["total"], x["value"], x["short"], x["signals"])
    from fastapi.testclient import TestClient

    client = TestClient(APP.create_app(Path(a.site), 120))
    r = client.get("/api/live")
    assert r.status_code == 200 and "picks" in r.json()
    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/data/backtest.json").status_code in (200, 404)
    print("api ok", json.dumps(client.get("/api/status").json(), ensure_ascii=False))


if __name__ == "__main__":
    main()
