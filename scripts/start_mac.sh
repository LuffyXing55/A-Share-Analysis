#!/bin/bash
# 一键启动（macOS）：安装依赖 → 同步最新数据 → 启动本地实时网站 → 打开浏览器
# 用法：在终端里进入项目文件夹，运行   bash scripts/start_mac.sh
set -e
cd "$(dirname "$0")/.."
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if command -v $c >/dev/null 2>&1; then
    v=$($c -c 'import sys; print(sys.version_info[0]*100+sys.version_info[1])')
    if [ "$v" -ge 311 ]; then PY=$c; break; fi
  fi
done
if [ -z "$PY" ]; then
  echo "需要 Python 3.11 或更高版本。建议先安装 Homebrew（https://brew.sh），然后运行：brew install python@3.12"
  exit 1
fi
# LightGBM 在 Mac 上需要 OpenMP 运行库
if [ "$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
  brew list libomp >/dev/null 2>&1 || brew install libomp
fi
if [ ! -d .venv ]; then
  echo "首次运行：创建虚拟环境…"
  $PY -m venv .venv
fi
source .venv/bin/activate
# 国内网络下 PyPI 可能较慢，可在运行前设置：export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
pip install -q --upgrade pip
pip install -q -r requirements.txt
# 同步 GitHub 上每天自动更新的结果（如果是 git clone 下来的）
if [ -d .git ]; then
  git pull --ff-only || echo "（git pull 失败，继续使用本地已有数据）"
fi
if [ ! -f site_data/dashboard.json ]; then
  echo "本地还没有数据，开始完整计算（首次需下载约 570MB 历史数据，可能需要几分钟到十几分钟）…"
  python -m ashare.pipeline.daily --mode close
fi
echo "启动本地网站：http://127.0.0.1:8000  （按 Ctrl+C 停止）"
( sleep 3; open "http://127.0.0.1:8000" 2>/dev/null || true ) &
python -m ashare.server.app
