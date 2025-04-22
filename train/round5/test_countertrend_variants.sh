#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# 测试原始反趋势策略
echo "测试原始反趋势策略 2_4_countertrend_strategy.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_4_countertrend_strategy.py 5 --print

# 测试所有反趋势策略变种
echo "测试 5_1_enhanced_countertrend.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/5_1_enhanced_countertrend.py 5 --print

echo "测试 5_2_multi_timeframe_countertrend.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/5_2_multi_timeframe_countertrend.py 5 --print

echo "测试 5_3_adaptive_countertrend.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/5_3_adaptive_countertrend.py 5 --print

echo "测试 5_4_countertrend_arbitrage.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/5_4_countertrend_arbitrage.py 5 --print

echo "测试 5_5_countertrend_volatility.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/5_5_countertrend_volatility.py 5 --print
