#!/bin/bash

# 设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# 运行策略测试
cd /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo
python -m prosperity3bt --trader train/round5/2_7_volatility_scalping.py --day 2 --day 3 --day 4 