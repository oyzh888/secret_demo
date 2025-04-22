#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# 测试所有火山岩期权策略
echo "测试 2_6_volcanic_options_arbitrage.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_6_volcanic_options_arbitrage.py 5 --print

echo "测试 2_7_volcanic_options_gamma_scalping.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_7_volcanic_options_gamma_scalping.py 5 --print

echo "测试 2_8_volcanic_options_volatility_trading.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_8_volcanic_options_volatility_trading.py 5 --print

echo "测试 2_9_volcanic_options_directional_bet.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_9_volcanic_options_directional_bet.py 5 --print
