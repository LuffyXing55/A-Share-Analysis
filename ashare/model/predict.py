"""加载训练好的 LightGBM 模型，对最新截面打分并给出可解释的贡献分解。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

FACTOR_LABELS = {
    "ep": "盈利收益率 E/P", "bp": "账面市值比 B/P", "roe": "ROE(PB/PE)", "size": "小市值",
    "vol_20": "20日低波动", "vol_60": "60日低波动", "ivol_20": "低特质波动", "maxret_20": "近20日无暴涨",
    "turn_20": "20日低换手", "turn_5": "5日低换手", "abn_turn": "换手未异常放大", "amihud_20": "非流动性溢价",
    "rev_1": "昨日回调", "rev_5": "5日回调", "rev_20": "20日回调", "mom_250_20": "长期超跌",
    "pv_corr_20": "量价背离", "ovn_20": "隔夜强势", "intra_20": "日内未追高", "gap_1": "今日跳空",
    "close_pos_1": "收盘未在高位", "dist_ma20": "低于20日均线", "high52": "远离52周高点",
    "vol_ratio_5": "量比不高", "limit_up_5": "近5日无涨停", "sp": "营收市值比 S/P", "dy": "股息率",
    "e_growth": "利润增长", "s_growth": "营收增长", "e_revision": "利润近季上修",
}


class Model:
    def __init__(self, path: str | Path):
        import lightgbm as lgb

        path = Path(path)
        self.booster = lgb.Booster(model_file=str(path))
        meta = json.loads(path.with_suffix(".json").read_text())
        self.features: list[str] = meta["features"]
        self.meta = meta

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X)

    def contrib(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X, pred_contrib=True)[:, :-1]


def matrix(ranks: dict[str, pd.Series], features: list[str], codes) -> np.ndarray:
    return np.column_stack([ranks[f].reindex(codes).values.astype("float64") if f in ranks
                            else np.full(len(codes), np.nan) for f in features])


def explain(model: Model, X: np.ndarray, codes, ranks: dict[str, pd.Series], top: int = 5) -> dict:
    c = model.contrib(X)
    out = {}
    for i, code in enumerate(codes):
        order = np.argsort(-np.abs(c[i]))[:top]
        items = []
        for j in order:
            f = model.features[j]
            pct = ranks[f].get(code) if f in ranks else None
            items.append({"factor": f, "name": FACTOR_LABELS.get(f, f), "contrib": round(float(c[i, j]), 4),
                          "pct": None if pct is None or not np.isfinite(pct) else round(float(pct) * 100, 1)})
        out[code] = items
    return out


def to_percentile(pred: np.ndarray) -> np.ndarray:
    s = pd.Series(pred)
    return (s.rank(pct=True) * 100).values
