"""规则 + 词典的中文财经新闻情绪打分（免费、无需 API key、可解释）。

词典为本项目自行整理（不含任何第三方受版权保护的词表），按 A 股新闻常见措辞分级：
强正面 +2、正面 +1、负面 -1、强负面 -2；遇到否定词（未/不/无/否认…）时翻转。
另外识别「事件类型」（业绩、回购增持、减持、监管处罚、订单中标、政策利好等），用于网站上的标签。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

POS2 = ["大幅增长", "大幅预增", "扭亏为盈", "超预期", "创历史新高", "涨停", "重大突破", "获批上市",
        "中标", "大单", "战略合作", "回购", "增持", "提高分红", "特别分红", "订单饱满", "量价齐升",
        "业绩预增", "首次覆盖买入", "上调评级", "降准", "降息", "超额完成", "国产替代突破"]
POS1 = ["增长", "上涨", "走强", "回暖", "改善", "利好", "受益", "提振", "扩产", "放量", "新高",
        "签约", "合作", "获批", "上调", "看好", "买入", "推荐", "盈利", "复苏", "加速", "落地",
        "支持", "补贴", "刺激", "宽松", "净流入", "反弹", "提价", "涨价", "突破", "领先", "景气",
        "增持评级", "护盘", "托底", "稳增长", "出口增长", "需求旺盛", "供不应求", "新订单"]
NEG1 = ["下跌", "走弱", "下滑", "下降", "放缓", "承压", "利空", "风险", "亏损", "减少", "收缩",
        "回落", "跌破", "净流出", "下调", "低于预期", "不及预期", "拖累", "疲软", "压力", "调查",
        "问询", "关注函", "诉讼", "延期", "终止", "取消", "降价", "过剩", "库存高企", "加息",
        "收紧", "制裁", "管制", "关税", "摩擦", "冲突", "停产", "召回", "违约", "警示", "减值"]
NEG2 = ["大幅下滑", "业绩预亏", "首亏", "巨亏", "暴跌", "跌停", "立案调查", "行政处罚", "退市",
        "ST", "财务造假", "重大违法", "减持计划", "清仓减持", "爆雷", "暴雷", "破产", "债务违约",
        "实体清单", "出口禁令", "停牌核查", "无法表示意见", "商誉减值", "大额计提"]
NEGATORS = ["未", "不", "无", "非", "否认", "并未", "没有", "暂无", "不会", "停止", "解除", "撤销"]

EVENT_RULES = [
    ("业绩", ["业绩", "净利润", "营收", "预增", "预亏", "快报", "季报", "年报", "中报"]),
    ("回购增持", ["回购", "增持"]),
    ("减持解禁", ["减持", "解禁"]),
    ("监管处罚", ["立案", "处罚", "问询", "关注函", "警示函", "监管"]),
    ("订单合作", ["中标", "订单", "签约", "合作", "合同"]),
    ("政策", ["政策", "国务院", "发改委", "工信部", "央行", "证监会", "财政部", "补贴", "规划"]),
    ("海外宏观", ["美联储", "加息", "降息", "非农", "CPI", "美债", "美元", "关税", "出口管制"]),
    ("大宗商品", ["原油", "黄金", "铜价", "锂", "大宗", "期货"]),
]

_LEX = [(w, 2) for w in POS2] + [(w, 1) for w in POS1] + [(w, -1) for w in NEG1] + [(w, -2) for w in NEG2]
# 长词优先，避免「大幅增长」被「增长」重复计分
_LEX.sort(key=lambda x: -len(x[0]))


@dataclass
class Sentiment:
    score: float          # -1 ~ 1
    hits: list[str]
    events: list[str]


def score_text(text: str) -> Sentiment:
    if not text:
        return Sentiment(0.0, [], [])
    used = [False] * len(text)
    total, hits = 0.0, []
    for word, w in _LEX:
        for m in re.finditer(re.escape(word), text):
            s, e = m.span()
            if any(used[s:e]):
                continue
            for k in range(s, e):
                used[k] = True
            window = text[max(0, s - 4): s]
            sign = -1 if any(n in window for n in NEGATORS) else 1
            total += w * sign
            hits.append(("非" if sign < 0 else "") + word)
    events = [name for name, kws in EVENT_RULES if any(k in text for k in kws)]
    norm = max(1.0, len(hits)) ** 0.5
    sc = max(-1.0, min(1.0, total / (2.0 * norm)))
    return Sentiment(round(sc, 3), hits[:8], events)


def match_news_to_stocks(news: list[dict], names: dict[str, str],
                         chain_keywords: dict[str, list[str]] | None = None) -> dict[str, list[dict]]:
    """把新闻按股票简称 / 产业链关键词归属到股票。names: code → 简称。"""
    out: dict[str, list[dict]] = {}
    clean = {c: n.replace("*", "").replace("ST", "").replace(" ", "") for c, n in names.items()}
    for item in news:
        text = (item.get("title", "") + " " + item.get("content", ""))
        sent = score_text(text)
        enriched = {**item, "sentiment": sent.score, "hits": sent.hits, "events": sent.events}
        for code, nm in clean.items():
            if len(nm) >= 2 and nm in text:
                out.setdefault(code, []).append({**enriched, "match": "公司"})
        if chain_keywords:
            for code, kws in chain_keywords.items():
                if any(k in text for k in kws):
                    out.setdefault(code, []).append({**enriched, "match": "产业链"})
    return out


def aggregate(items: list[dict], half_life_items: int = 10) -> float:
    """新闻情绪汇总（越新权重越高，公司新闻权重高于产业链新闻）。"""
    if not items:
        return 0.0
    tot, wsum = 0.0, 0.0
    for i, it in enumerate(items):
        w = 0.5 ** (i / half_life_items) * (1.0 if it.get("match") == "公司" else 0.4)
        tot += w * it.get("sentiment", 0.0)
        wsum += w
    return round(tot / wsum, 3) if wsum else 0.0
