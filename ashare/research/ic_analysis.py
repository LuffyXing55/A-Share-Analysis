"""单因子 IC 分析：月度（20 日持有）与短线（5 日持有）。

python -m ashare.research.ic_analysis --out data/research
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..factors.library import compute_factors
from .common import forward_return, ic_summary, load_panels, month_ends, rank_ic, universe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research")
    ap.add_argument("--start", default="2014-01-01")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    p = load_panels("2012-01-01")
    u = universe(p)
    fac = compute_factors(p)
    y20 = forward_return(p, 20)
    y5 = forward_return(p, 5)
    dates = p["close"].index[p["close"].index >= pd.Timestamp(args.start)]
    me = month_ends(dates)[:-1]
    wk = dates[::5][:-2]
    rows = []
    for name, f in fac.items():
        for period, ds in [("2014-2019", lambda d: d < pd.Timestamp("2020-01-01")),
                           ("2020-2026", lambda d: d >= pd.Timestamp("2020-02-01")),
                           ("2023-2026", lambda d: d >= pd.Timestamp("2023-01-01"))]:
            m_dates = pd.DatetimeIndex([d for d in me if ds(d)])
            w_dates = pd.DatetimeIndex([d for d in wk if ds(d)])
            s20 = ic_summary(rank_ic(f, y20, u, m_dates))
            s5 = ic_summary(rank_ic(f, y5, u, w_dates))
            rows.append({"factor": name, "period": period,
                         **{f"m_{k}": v for k, v in s20.items()},
                         **{f"w_{k}": v for k, v in s5.items()}})
        print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(out / "single_factor_ic.csv", index=False)
    print(df.pivot(index="factor", columns="period", values=["m_ic", "w_ic"]).round(4).to_string())
    (out / "universe_size.json").write_text(json.dumps(
        {str(d.date()): int(u.loc[d].sum()) for d in me[::6]}, indent=1))


if __name__ == "__main__":
    main()
