"""本地 Mac 服务：网页 + 盘中实时刷新。

启动：  python -m ashare.server.app        然后打开 http://127.0.0.1:8000
（scripts/start_mac.sh 会自动完成安装、同步数据、启动与打开浏览器）
"""
from __future__ import annotations

import argparse
import copy
import json
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BJ = timezone(timedelta(hours=8))


class LiveEngine:
    def __init__(self, site_dir: Path, refresh: int = 120):
        self.site = site_dir
        self.refresh = refresh
        self.lock = threading.Lock()
        self.dash: dict | None = None
        self.hist: dict = {}
        self.hist_day = None
        self.quotes_ts = 0.0
        self.overseas_ts = 0.0
        self.news_ts = 0.0
        self.models = None
        self.models_stamp = None
        self.status = "starting"

    # ------------------------------------------------------------ 数据加载
    def load_base(self):
        dash = json.loads((self.site / "dashboard.json").read_text())
        latest = json.loads((self.site / "state" / "latest.json").read_text())
        fdist = json.loads((self.site / "state" / "factor_dist.json").read_text())
        return dash, latest, fdist

    def ensure_models(self):
        # git pull 可能带来重新训练的模型：模型文件变化时重新加载
        meta = ROOT / "models" / "short_model.json"
        stamp = meta.stat().st_mtime if meta.exists() else None
        if self.models is None or stamp != self.models_stamp:
            from ..model.predict import Model

            self.models = {"short": Model(ROOT / "models" / "short_model.txt")}
            self.models_stamp = stamp
        return self.models

    def is_trading_time(self, now: datetime) -> bool:
        if now.weekday() >= 5:
            return False
        t = now.hour * 60 + now.minute
        return (9 * 60 + 15) <= t <= (11 * 60 + 35) or (12 * 60 + 55) <= t <= (15 * 60 + 5)

    # ------------------------------------------------------------ 刷新
    def refresh_once(self) -> None:
        from ..live import realtime as RT
        from ..live import sources as S
        from ..macro import overseas as OV
        from ..macro import watch as W
        from ..news.sentiment import aggregate, match_news_to_stocks, score_text
        from ..pipeline.daily import INDEX_CODES

        now = datetime.now(BJ)
        dash, latest, fdist = self.load_base()
        live = copy.deepcopy(dash)
        live["mode"] = "live"
        tech = live.get("tech") or None
        tech_base = ((latest.get("tech") or {}).get("base")) or {}
        main_codes = [p["code"] for p in dash["picks"]]
        tech_codes = [p["code"] for p in (tech or {}).get("picks", [])]
        codes = list(dict.fromkeys(main_codes + tech_codes))
        all_base = {**tech_base, **latest["base"]}
        if self.hist_day != now.date() or not self.hist or any(c not in self.hist for c in codes):
            self.hist = RT.fetch_histories(codes)
            self.hist_day = now.date()
        quotes = {q["code"]: q for q in S.tencent_quotes(codes)}
        try:
            idx = S.tencent_index([c for c, _ in INDEX_CODES])
            nm = dict(INDEX_CODES)
            live["market"]["indices"] = [{"code": x["code"], "name": nm.get(x["code"], x["name"]),
                                          "price": x["price"], "pct": (x["pct"] or 0) / 100} for x in idx]
        except Exception:  # noqa: BLE001
            pass
        today_mkt = None
        idx_pct = {x.get("code"): x.get("pct") for x in live["market"].get("indices", [])}
        for code in ("sh000985", "sh000852"):          # 优先中证全指，其次中证 1000
            if idx_pct.get(code) is not None:
                today_mkt = idx_pct[code]
                break
        fb = {c: (all_base.get(c) or {}).get("feat_ranks") or {} for c in codes}
        scores, scores_tech = {}, {}
        # 收盘数据若是用另一个版本的短线模型算出的（模型刚更新、还没跑过收盘流水线），
        # 盘中就不重算分数，沿用收盘值，避免新旧模型的分布混用
        model_ok = fdist.get("short_model_trained_at") == self.ensure_models()["short"].meta.get("trained_at")
        try:   # 收盘数据仍含已剔除的因子（例如刚改了 exclude_factors）：同样视为过期
            excluded = set(json.loads((ROOT / "config" / "settings.json").read_text()).get("exclude_factors", []))
        except Exception:  # noqa: BLE001
            excluded = set()
        model_ok = model_ok and not (excluded & set(fdist.get("tail_weights") or {}))
        if not model_ok:
            self.status = "模型或因子设置已更新，等待下一次收盘流水线；盘中暂时显示收盘分数"
        try:
            if not model_ok:
                raise RuntimeError("model version mismatch")
            buckets = {c: (all_base.get(c) or {}).get("mv_bucket") for c in main_codes}
            scores = RT.live_scores(self.ensure_models(), fdist, {c: self.hist[c] for c in main_codes if c in self.hist},
                                    quotes, now, fallback_ranks=fb, today_mkt=today_mkt, buckets=buckets)
            if tech_codes and fdist.get("pred_dist_tech"):
                # 科技榜的分数是科技股池内的分位：用科技股池的模型输出分布来换算
                scores_tech = RT.live_scores(self.ensure_models(), fdist,
                                             {c: self.hist[c] for c in tech_codes if c in self.hist},
                                             quotes, now, fallback_ranks=fb, today_mkt=today_mkt,
                                             pred_dist_key="pred_dist_tech")
        except Exception as e:  # noqa: BLE001
            if model_ok:
                traceback.print_exc()
            else:
                _ = e
        # 海外行情每 10 分钟、新闻每 3 分钟刷新
        if time.time() - self.overseas_ts > 600 or not hasattr(self, "ov_quotes"):
            try:
                self.ov_quotes = OV.fetch_overseas(OV.needed_symbols(codes, [all_base.get(c, {}).get("industry")
                                                                             for c in codes]))
            except Exception:  # noqa: BLE001
                self.ov_quotes = getattr(self, "ov_quotes", {})
            self.overseas_ts = time.time()
        if time.time() - self.news_ts > 180 or not hasattr(self, "news"):
            try:
                self.news = S.all_news(80)
            except Exception:  # noqa: BLE001
                self.news = getattr(self, "news", [])
            self.news_ts = time.time()
        settings = json.loads((ROOT / "config" / "settings.json").read_text())
        ov_cfg = settings["overlay"]
        stats = OV.load_linkage_stats(ROOT / "config" / "linkage_stats.json")
        names = {c: (all_base.get(c) or {}).get("name", c) for c in codes}
        matched = match_news_to_stocks(self.news, names)

        def update(plist, base, sc_map, w):
            for p in plist:
                c = p["code"]
                q = quotes.get(c) or {}
                b = base.get(c) or {}
                if q.get("price"):
                    p["price"] = q["price"]
                    p["pct_today"] = (q["pct"] or 0) / 100
                    entry = b.get("price")
                    if b.get("ret_since_pick") is not None and entry:
                        p["ret_since_pick"] = (1 + b["ret_since_pick"]) * q["price"] / entry - 1
                    p["is_st"] = "ST" in (q.get("name") or "")
                sc = sc_map.get(c) or {}
                pred_ex, chains = OV.chain_signals(c, self.ov_quotes, stats, ov_cfg["overseas_min_t"],
                                                   p.get("industry"), min_date=latest["data_date"])
                ov_pts = float(np.clip(pred_ex * ov_cfg["overseas_points_per_pct"], -ov_cfg["overseas_cap"],
                                       ov_cfg["overseas_cap"]))
                items = matched.get(c, [])
                ns = aggregate(items)
                news_pts = float(np.clip(ns * ov_cfg["news_points"], -6, 6)) if items else 0.0
                value = sc.get("value") if sc.get("value") is not None else b.get("value")
                short_model = sc.get("short") if sc.get("short") is not None else b.get("short_model")
                short = None if short_model is None else round(float(np.clip(short_model + ov_pts + news_pts, 0, 100)), 1)
                p["value"], p["short"] = value, short
                p["total"] = None if value is None or short is None else round(w["value"] * value + w["short"] * short, 1)
                p["chains"] = chains
                p["signals"] = {"overseas": round(ov_pts / 4, 2), "news": ns, "overseas_points": round(ov_pts, 1),
                                "news_points": round(news_pts, 1)}
                if items:
                    p["news"] = [{k: x.get(k) for k in ("time", "source", "title", "url", "sentiment", "events", "match")}
                                 for x in items[:6]]
            prev_close = {p["code"]: p["rank"] for p in plist}
            ranked = sorted(plist, key=lambda x: (-(x["total"] if x["total"] is not None else -1), x["code"]))
            for i, p in enumerate(ranked, 1):
                p["prev_rank"] = prev_close.get(p["code"])   # 盘中「变化」= 相对上一次收盘后的排名
                p["rank"] = i
            return ranked

        tag = now.strftime("%Y-%m-%d") + "（盘中）"
        live["picks"] = update(live["picks"], latest["base"], scores, settings["score_weights"])
        for p in live["picks"]:
            live["history"].setdefault(p["code"], []).append([tag, p["total"], p["value"], p["short"]])
        if tech:
            tw = (settings.get("tech") or {}).get("score_weights", settings["score_weights"])
            tech["picks"] = update(tech["picks"], tech_base, scores_tech, tw)
            for p in tech["picks"]:
                tech["history"].setdefault(p["code"], []).append([tag, p["total"], p["value"], p["short"]])
        live["market"]["time"] = now.strftime("%Y-%m-%d %H:%M:%S")
        live["generated_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
        both = live["picks"] + [p for p in (tech or {}).get("picks", []) if p["code"] not in latest["base"]]
        live["watch"]["signals"] = W.live_signals(self.ov_quotes, both) + [
            s for s in dash["watch"].get("signals", []) if not s["title"].startswith("海外异动")]
        news_out = []
        for n in self.news[:60]:
            st = score_text(n.get("title", "") + n.get("content", ""))
            txt = n.get("title", "") + n.get("content", "")
            news_out.append({**n, "content": n.get("content", "")[:300], "sentiment": st.score,
                             "events": st.events, "picks": [c for c, nm in names.items() if nm and nm in txt]})
        live["news"] = news_out
        with self.lock:
            self.dash = live
            self.status = "ok" if model_ok else "模型已更新，等待下一次收盘流水线；盘中暂时显示收盘分数"

    def maybe_git_pull(self) -> None:
        """每 30 分钟 git pull 一次，拿到 GitHub Actions 每天算好的新结果（含月初换仓）。"""
        if time.time() - getattr(self, "pull_ts", 0) < 1800 or not (ROOT / ".git").exists():
            return
        self.pull_ts = time.time()
        import subprocess

        try:
            subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only", "-q"], timeout=120,
                           capture_output=True)
        except Exception:  # noqa: BLE001
            pass

    def loop(self) -> None:
        while True:
            try:
                self.maybe_git_pull()
                self.refresh_once()
            except Exception as e:  # noqa: BLE001
                self.status = f"刷新失败：{e}"
                traceback.print_exc()
            now = datetime.now(BJ)
            time.sleep(self.refresh if self.is_trading_time(now) else max(self.refresh, 900))


def create_app(site_dir: Path, refresh: int):
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    engine = LiveEngine(site_dir, refresh)
    threading.Thread(target=engine.loop, daemon=True).start()
    app = FastAPI(title="A股30强看好度")

    @app.get("/api/live")
    def api_live():
        with engine.lock:
            d = engine.dash
        if d is None:
            return JSONResponse(json.loads((site_dir / "dashboard.json").read_text()))
        return JSONResponse(d)

    @app.get("/api/status")
    def api_status():
        return {"status": engine.status}

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web" / "index.html")

    app.mount("/data", StaticFiles(directory=site_dir), name="data")
    app.mount("/", StaticFiles(directory=ROOT / "web"), name="web")
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(ROOT / "site_data"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    import uvicorn

    settings = json.loads((ROOT / "config" / "settings.json").read_text())
    app = create_app(Path(args.site), settings.get("intraday_refresh_seconds", 120))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
