#!/bin/bash
# 在 Mac 本地完整重算一次（收盘后运行），结果写入 site_data/
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m ashare.pipeline.daily --mode ${1:-close}
