"""核心逻辑单元测试（不需要联网）：python -m pytest tests -q"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ashare.backtest.engine import BTConfig, run_backtest
from ashare.factors.library import compute_factors
from ashare.live import sources as S
from ashare.live.realtime import pct_from_dist, session_fraction
from ashare.news.sentiment import aggregate, score_text

# 资料汇编 1.3 节中腾讯 qt.gtimg.cn 的真实返回样例（2026-11-03 16:14）
TENCENT_SAMPLE = ('v_sh600000="1~浦发银行~600000~10.09~10.06~10.06~977551~538836~438715~10.09~778~10.08~468~10.07~279'
                  '~10.06~699~10.05~238~10.10~179~10.11~1524~10.12~91~10.13~2695~10.14~220~~20261103161455~0.03~0.30'
                  '~10.18~9.85~10.09/977551/978658333~977551~97866~0.29~6.49~~10.18~9.85~3.28~3360.56~3360.56~0.44'
                  '~11.07~9.05~1.22~-2247~10.01~6.21~6.72~~~-0.01~97865.8333~18.7674~186~ A~GP-A~-16.06~2.75~4.16'
                  '~6.17~0.50~12.06~8.07~3.38~9.91~4.78~33305838300~33305838300~-31.33~4.89~33305838300~~~-8.19'
                  '~-0.39~~CNY~0~___D__F__N~10.18~-7132~";')


class _Resp:
    def __init__(self, text):
        self.text = text


class _FakeFetcher:
    def __init__(self, text):
        self.text = text

    def get(self, *a, **k):
        return _Resp(self.text)


def test_normalize_code():
    assert S.normalize_code("600519") == "sh600519"
    assert S.normalize_code("000001.SZ") == "sz000001"
    assert S.normalize_code("688981") == "sh688981"
    assert S.normalize_code("920000") == "bj920000"
    assert S.normalize_code("300750") == "sz300750"


def test_tencent_parse():
    S.set_fetcher(_FakeFetcher(TENCENT_SAMPLE))
    try:
        q = S.tencent_quotes(["sh600000"])[0]
    finally:
        S.set_fetcher(None)
    assert q["name"] == "浦发银行"
    assert q["price"] == 10.09 and q["prev_close"] == 10.06
    assert abs(q["amount"] - 97866e4) < 1
    assert q["pe_ttm"] == 6.49 and q["pb"] == 0.44
    assert abs(q["float_mv"] - 3360.56e8) < 1e3
    assert q["limit_up"] == 11.07 and q["limit_down"] == 9.05


def test_sentiment():
    assert score_text("公司三季度净利润大幅增长，超预期").score > 0.5
    assert score_text("公司收到证监会立案调查通知书").score < -0.5
    assert score_text("公司并未收到立案调查").score > 0
    assert "回购增持" in score_text("公司拟回购股份").events
    assert aggregate([]) == 0.0


def _toy_panels(n_days=320, n_stocks=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"sz{300000+i:06d}" if i % 2 else f"sh{600000+i:06d}" for i in range(n_stocks)]
    r = rng.normal(0, 0.02, (n_days, n_stocks))
    close = pd.DataFrame(10 * np.exp(np.cumsum(r, 0)), index=idx, columns=cols)
    open_ = close.shift(1).fillna(close) * (1 + rng.normal(0, 0.005, close.shape))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    vol = pd.DataFrame(rng.uniform(1e4, 1e5, close.shape), index=idx, columns=cols)
    p = {"open": open_, "high": high, "low": low, "close": close, "vwap": (open_ + close) / 2,
         "volume": vol, "amount": vol * close / 10, "factor": pd.DataFrame(1.0, index=idx, columns=cols)}
    return p


def test_factors_shapes():
    p = _toy_panels()
    f = compute_factors(p)
    for k, v in f.items():
        assert v.shape == p["close"].shape, k
    assert f["rev_5"].iloc[-1].notna().all()


def test_backtest_perfect_foresight_beats_random():
    p = _toy_panels(400, 80, seed=1)
    fwd = p["vwap"].shift(-21) / p["vwap"].shift(-1) - 1
    me = p["close"].index[25::21][:-2]
    good = fwd.loc[me]
    elig = pd.DataFrame(True, index=me, columns=p["close"].columns)
    no_block = pd.DataFrame(False, index=p["close"].index, columns=p["close"].columns)
    cfg = BTConfig(top_n=10, industry_cap=None)
    res_good = run_backtest(good, elig, p["vwap"], no_block, no_block, cfg)
    res_bad = run_backtest(-good, elig, p["vwap"], no_block, no_block, cfg)
    assert res_good.nav.iloc[-1] > res_bad.nav.iloc[-1]
    assert res_good.nav.iloc[-1] > 1.0


def test_session_fraction_and_dist():
    from datetime import timedelta, timezone

    tz = timezone(timedelta(hours=8))
    assert session_fraction(datetime(2026, 11, 4, 10, 30, tzinfo=tz)) == 0.25
    assert session_fraction(datetime(2026, 11, 4, 15, 30, tzinfo=tz)) == 1.0
    dist = list(np.linspace(0, 10, 101))
    assert abs(pct_from_dist(5.0, dist) - 0.5) < 1e-9
