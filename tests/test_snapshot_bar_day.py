from datetime import datetime

import pandas as pd

from ashare.pipeline.daily import snapshot_bar_day

CAL = pd.DatetimeIndex(pd.bdate_range("2026-11-02", "2026-11-13"))


def snap(day: str, n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({"time": [day + "150003"] * n})


def test_run_after_midnight_still_fills_previous_day():
    # 11-05 收盘后的任务被拖到 11-06 凌晨 2 点才跑
    d, why = snapshot_bar_day(snap("20261105"), pd.Timestamp("2026-11-04"), CAL, datetime(2026, 11, 6, 2, 53))
    assert d == pd.Timestamp("2026-11-05") and why == ""


def test_same_day_before_close_not_filled():
    d, _ = snapshot_bar_day(snap("20261105"), pd.Timestamp("2026-11-04"), CAL, datetime(2026, 11, 5, 12, 30))
    assert d is None


def test_same_day_after_close_filled():
    d, _ = snapshot_bar_day(snap("20261105"), pd.Timestamp("2026-11-04"), CAL, datetime(2026, 11, 5, 19, 23))
    assert d == pd.Timestamp("2026-11-05")


def test_gap_not_filled_and_explained():
    d, why = snapshot_bar_day(snap("20261106"), pd.Timestamp("2026-11-04"), CAL, datetime(2026, 11, 7, 3, 58))
    assert d is None and "缺 1 个交易日" in why


def test_upstream_already_current():
    d, why = snapshot_bar_day(snap("20261106"), pd.Timestamp("2026-11-06"), CAL, datetime(2026, 11, 7, 3, 58))
    assert d is None and why == ""
