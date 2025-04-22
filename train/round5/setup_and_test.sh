#!/bin/bash

# 设置PYTHONPATH确保可以正确导入模块
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 使用uv安装依赖
echo "Installing dependencies with uv..."
uv pip install matplotlib pandas numpy

# 运行测试脚本
echo "Running volatility scalping strategy test..."
python train/round5/test_volatility_scalping.py

# 如果需要特定日期测试，取消下面的注释
# echo "Running specific day tests..."
# python -m prosperity3bt --trader train/round5/2_7_volatility_scalping.py --day 2 --day 3 --day 4 