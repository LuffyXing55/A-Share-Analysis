"""获取近期全市场历史日线（用于每日计算因子）。

主数据源：chenditc/investment_data 的 GitHub Release（每日更新，Apache-2.0）。在 GitHub Actions 中
GitHub→GitHub 下载很快；在国内网络下也可以下载（可能较慢）。下载后只在本地/运行器中使用，不提交进仓库。
"""
from __future__ import annotations

import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import pandas as pd

from ..data.build_panels import is_target_board
from ..data.qlib_reader import build_panel, read_calendar, read_instruments

RELEASE_BASE = "https://github.com/chenditc/investment_data/releases/latest/download"
PRICE_FIELDS = ["open", "high", "low", "close", "vwap", "volume", "amount", "factor"]


def _download(url: str, dest: Path, tries: int = 6, resume: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(tries):
        # 大文件用 curl 断点续传；小文件（manifest）每次重新下载
        cmd = ["curl", "-sSL", "--fail"] + (["-C", "-"] if resume else []) + ["-o", str(dest), url]
        rc = subprocess.call(cmd)
        if rc == 0:
            return
        time.sleep(2 ** (i + 1))
    raise RuntimeError(f"下载失败：{url}")


def ensure_qlib_data(cache_dir: str | Path = "data/qlib_cache") -> Path:
    """下载并解压最新 Release，按 manifest 的 release_tag 缓存，同一天不重复下载。"""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    man_path = cache / "manifest.json"
    man_path.with_suffix(".new").unlink(missing_ok=True)
    _download(f"{RELEASE_BASE}/qlib_bin.manifest.json", man_path.with_suffix(".new"), resume=False)
    new_man = json.loads(man_path.with_suffix(".new").read_text())
    root = cache / "qlib"
    if man_path.exists() and (root / "calendars" / "day.txt").exists():
        old = json.loads(man_path.read_text())
        if old.get("release_tag") == new_man.get("release_tag"):
            return root
    tar_path = cache / "qlib_bin.tar.gz"
    if tar_path.exists():
        tar_path.unlink()
    _download(f"{RELEASE_BASE}/qlib_bin.tar.gz", tar_path)
    if root.exists():
        subprocess.call(["rm", "-rf", str(root)])
    root.mkdir(parents=True)
    with tarfile.open(tar_path) as tar:
        members = []
        for m in tar.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2:
                continue
            m.name = parts[1]
            members.append(m)
        try:
            tar.extractall(root, members=members, filter="data")
        except TypeError:  # Python < 3.11.4 没有 filter 参数
            for m in members:
                if m.name.startswith("/") or ".." in m.name.split("/"):
                    raise RuntimeError(f"压缩包含有不安全路径：{m.name}")
            tar.extractall(root, members=members)
    tar_path.unlink()
    man_path.with_suffix(".new").replace(man_path)
    return root


def recent_panels(qlib_root: str | Path, days: int = 420, codes: list[str] | None = None) -> dict:
    root = Path(qlib_root)
    cal = read_calendar(root)
    inst = read_instruments(root, "all")
    last = cal[-1]
    alive = inst.groupby("code")["end"].max()
    alive = alive[alive >= last - pd.Timedelta(days=40)].index
    if codes is None:
        codes = sorted(c for c in alive if is_target_board(c))
    start = cal[max(0, len(cal) - days)]
    p = {}
    for f in PRICE_FIELDS:
        p[f] = build_panel(root / "features", f, cal, codes, str(start.date()))
    # 去掉尾部没有任何数据的日期（Release 的日历可能含当天但数据未到）
    valid = p["close"].notna().sum(axis=1) > 500
    last_valid = valid[valid].index.max()
    for f in p:
        p[f] = p[f].loc[:last_valid]
    return p


def default_cache_dir() -> Path:
    return Path(os.environ.get("ASHARE_CACHE", "data")) / "qlib_cache"
