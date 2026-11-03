"""带限速、重试、超时与本地缓存回退的 HTTP 工具。

设计原则（来自资料汇编 0.2 / 0.4）：
- 东方财富 push2 系按 IP 限流，被封后至少 5–40 分钟，期间不要再请求任何 push2 域名；
- 腾讯 / 新浪批量接口为盘中主力，串行 + 随机间隔；
- 每个请求失败都回退到上一次成功的缓存，而不是让网站整体出错。
"""
from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any

import requests

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# 各域名的最小请求间隔（秒）与熔断冷却时间（秒）
DOMAIN_POLICY = {
    "push2.eastmoney.com": (1.5, 900),
    "push2his.eastmoney.com": (1.5, 900),
    "push2ex.eastmoney.com": (1.5, 900),
    "datacenter-web.eastmoney.com": (0.6, 300),
    "np-weblist.eastmoney.com": (1.0, 300),
    "qt.gtimg.cn": (0.3, 120),
    "web.ifzq.gtimg.cn": (0.4, 120),
    "hq.sinajs.cn": (0.8, 300),
    "zhibo.sina.com.cn": (1.0, 300),
    "www.cls.cn": (1.5, 300),
    "query1.finance.yahoo.com": (0.8, 300),
    "query2.finance.yahoo.com": (0.8, 300),
    "fred.stlouisfed.org": (0.5, 300),
}
PUSH2_FAMILY = {"push2.eastmoney.com", "push2his.eastmoney.com", "push2ex.eastmoney.com"}


class SourceUnavailable(RuntimeError):
    pass


class Fetcher:
    def __init__(self, cache_dir: str | Path = "cache/http", timeout: float = 12.0):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA, "Accept": "*/*"})
        self.timeout = timeout
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last: dict[str, float] = {}
        self._blocked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _domain(url: str) -> str:
        return url.split("/")[2].split(":")[0]

    def _wait(self, domain: str) -> None:
        gap, _ = DOMAIN_POLICY.get(domain, (0.5, 300))
        with self._lock:
            now = time.time()
            last = self._last.get(domain, 0.0)
            sleep = last + gap * random.uniform(1.0, 1.6) - now
            if sleep > 0:
                time.sleep(sleep)
            self._last[domain] = time.time()

    def _block(self, domain: str) -> None:
        _, cool = DOMAIN_POLICY.get(domain, (0.5, 300))
        until = time.time() + cool
        targets = PUSH2_FAMILY if domain in PUSH2_FAMILY else {domain}
        for d in targets:
            self._blocked_until[d] = until

    def is_blocked(self, url_or_domain: str) -> bool:
        d = self._domain(url_or_domain) if "/" in url_or_domain else url_or_domain
        return time.time() < self._blocked_until.get(d, 0.0)

    def get(self, url: str, params: dict | None = None, headers: dict | None = None,
            encoding: str | None = None, retries: int = 2) -> requests.Response:
        domain = self._domain(url)
        if self.is_blocked(domain):
            raise SourceUnavailable(f"{domain} 处于冷却期")
        err: Exception | None = None
        for attempt in range(retries + 1):
            self._wait(domain)
            try:
                r = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if r.status_code in (403, 429, 456) or (500 <= r.status_code < 600):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                if encoding:
                    r.encoding = encoding
                return r
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                err = e
                if domain in PUSH2_FAMILY:
                    # push2 断连多半是限流，立刻熔断，不要重试（重试会延长封禁）
                    self._block(domain)
                    break
                time.sleep(1.5 * (attempt + 1))
        self._block(domain)
        raise SourceUnavailable(f"{domain}: {err}")

    # ---------- 简单的 JSON 缓存（失败回退） ----------
    def cache_put(self, key: str, obj: Any) -> None:
        path = self.cache_dir / f"{key}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"ts": time.time(), "data": obj}, ensure_ascii=False))
        tmp.replace(path)

    def cache_get(self, key: str, max_age: float | None = None) -> Any | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            obj = json.loads(path.read_text())
        except Exception:
            return None
        if max_age is not None and time.time() - obj.get("ts", 0) > max_age:
            return None
        return obj.get("data")
