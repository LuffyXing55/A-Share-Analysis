"""数据源自检：逐个测试网站用到的数据源是否可达，并输出报告。

在 Mac 上运行：  python scripts/probe_sources.py --out probe_out
在 GitHub Actions 上也会运行（.github/workflows/probe.yml），用于确认海外 IP 的可用性。
同时下载申万行业分类表（公开 xls），作为行业中性化的基础数据。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ashare.live import sources as S  # noqa: E402


def run(name, fn, report):
    t0 = time.time()
    try:
        res = fn()
        n = len(res) if hasattr(res, "__len__") else 1
        sample = None
        if isinstance(res, list) and res:
            sample = res[0]
        elif hasattr(res, "head"):
            sample = res.head(2).to_dict(orient="records")
        report[name] = {"ok": True, "n": n, "sec": round(time.time() - t0, 2),
                        "sample": json.loads(json.dumps(sample, ensure_ascii=False, default=str))}
        print(f"[OK]   {name}: n={n} {time.time()-t0:.1f}s")
        return res
    except Exception as e:  # noqa: BLE001
        report[name] = {"ok": False, "sec": round(time.time() - t0, 2),
                        "error": f"{type(e).__name__}: {str(e)[:300]}"}
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc(limit=1)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probe_out")
    ap.add_argument("--history", action="store_true", help="同时下载海外标的 10 年日线（研究用）")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"run_at_utc": datetime.utcnow().isoformat()}
    today = datetime.now().strftime("%Y-%m-%d")
    lastweek = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    run("tencent_quotes", lambda: S.tencent_quotes(["sh600519", "sz000001", "sz300750", "sh688981"]), rep)
    run("tencent_index", lambda: S.tencent_index(["sh000001", "sz399006", "sh000300"]), rep)
    run("tencent_kline", lambda: S.tencent_kline("sh600519", 30), rep)
    run("sina_quotes", lambda: S.sina_quotes(["sh600519", "sz000001"]), rep)
    run("em_performance", lambda: S.em_performance("2026-09-30", max_pages=1), rep)
    run("em_forecast", lambda: S.em_forecast("2026-12-31", max_pages=1), rep)
    run("em_lift", lambda: S.em_lift_schedule(today, (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")), rep)
    run("em_billboard", lambda: S.em_billboard(lastweek, today), rep)
    run("em_zt_pool", lambda: S.em_limit_pool("zt"), rep)
    run("em_clist_page", lambda: S.em_clist_page(1, 100), rep)
    run("em_fast_news", lambda: S.em_fast_news(20), rep)
    run("sina_7x24", lambda: S.sina_7x24(20), rep)
    run("cls_telegraph", lambda: S.cls_telegraph(20), rep)
    for sym in ["NVDA", "^SOX", "^IXIC", "^GSPC", "KWEB", "^TNX", "DX-Y.NYB", "CNH=X", "GC=F", "HG=F", "CL=F", "^VIX"]:
        run(f"yahoo_{sym}", lambda s=sym: S.yahoo_chart(s, "5d"), rep)
    run("fred_DGS10", lambda: S.fred_series("DGS10"), rep)
    sw = run("sw_industry", S.sw_industry_history, rep)
    if sw is not None:
        sw.to_csv(out / "sw_industry_history.csv", index=False)

    if args.history:
        # 研究用：海外标的长历史（用于检验「美股隔夜 → A 股产业链」联动），只作为临时 artifact
        hist = out / "global_history"
        hist.mkdir(exist_ok=True)
        from ashare.macro.linkage import GLOBAL_SYMBOLS

        for sym in GLOBAL_SYMBOLS:
            df = run(f"hist_{sym}", lambda s=sym: S.yahoo_chart(s, "10y"), rep)
            if df is not None and len(df):
                df.to_csv(hist / f"{sym.replace('^', 'IDX_').replace('=', '_').replace('.', '_')}.csv",
                          index=False)
        for sid in ["DGS10", "DGS2", "DFF", "DTWEXBGS", "VIXCLS", "T10YIE"]:
            df = run(f"fred_{sid}", lambda s=sid: S.fred_series(s), rep)
            if df is not None:
                df.to_csv(hist / f"FRED_{sid}.csv", index=False)

    (out / "probe_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    ok = sum(1 for v in rep.values() if isinstance(v, dict) and v.get("ok"))
    print(f"done: {ok} ok")


if __name__ == "__main__":
    main()
