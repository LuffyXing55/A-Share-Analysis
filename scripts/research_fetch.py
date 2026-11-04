"""研究数据抓取（在本地电脑上运行，供离线回测使用）。

抓取内容：东财业绩报表（含行业分类）、全部 A 股名称、海外标的与期货日线、中美国债收益率、两融余额。
输出：一个 tar.gz，只保存在本地，不要提交进仓库（.gitignore 已忽略 *.tar.gz）。
用法：python scripts/research_fetch.py --out research_out
"""
from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from ashare.live import sources as S  # noqa: E402
from ashare.macro.linkage import GLOBAL_SYMBOLS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research_out")
    ap.add_argument("--start", default="2016-01-01")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "global").mkdir(parents=True, exist_ok=True)
    log = []

    for rd in ["2026-09-30", "2026-06-30", "2025-12-31"]:
        try:
            df = S.em_performance(rd, max_pages=20)
            df.to_csv(out / f"perf_{rd}.csv", index=False)
            log.append(f"perf {rd}: {len(df)}")
        except Exception as e:  # noqa: BLE001
            log.append(f"perf {rd} FAIL {e}")

    for name, fn in [("treasury", lambda: S.em_treasury_yield(pages=8)),
                     ("margin", lambda: S.em_margin_daily(pages=6))]:
        try:
            df = fn()
            df.to_csv(out / f"{name}.csv", index=False)
            log.append(f"{name}: {len(df)}")
        except Exception as e:  # noqa: BLE001
            log.append(f"{name} FAIL {e}")

    for sym in GLOBAL_SYMBOLS:
        try:
            df = S.overseas_daily(sym, args.start)
            fn = sym.replace("^", "IDX_").replace("=", "_").replace(".", "_")
            df.to_csv(out / "global" / f"{fn}.csv", index=False)
            log.append(f"global {sym}: {len(df)} {df.attrs.get('tencent_code', '')} "
                       f"{df['date'].min().date() if len(df) else ''}")
        except Exception as e:  # noqa: BLE001
            log.append(f"global {sym} FAIL {str(e)[:150]}")

    # 单位核对：腾讯日 K 的成交量（主板 vs 科创板）与实时行情成交量（手）对比
    for code in ["sh600519", "sh688981", "sz300750"]:
        try:
            k = S.tencent_kline(code, 5)
            q = S.tencent_quotes([code])[0]
            log.append(f"unit {code}: kline_last={k['volume'].iloc[-1]} date={k['date'].iloc[-1].date()} "
                       f"quote_vol_lots={q['volume_lots']} quote_time={q['time']}")
        except Exception as e:  # noqa: BLE001
            log.append(f"unit {code} FAIL {e}")

    (out / "fetch_log.txt").write_text("\n".join(log))
    print("\n".join(log))
    with tarfile.open("research_out.tar.gz", "w:gz") as tar:
        tar.add(out, arcname="research_out")


if __name__ == "__main__":
    main()
