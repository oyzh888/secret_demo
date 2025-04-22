#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# 测试所有新策略
echo "测试 2_1_volcanic_rock_specialist.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_1_volcanic_rock_specialist.py 5 --print

echo "测试 2_2_reverse_fee_aware.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_2_reverse_fee_aware.py 5 --print

echo "测试 2_3_product_selective_mm.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_3_product_selective_mm.py 5 --print

echo "测试 2_4_countertrend_strategy.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_4_countertrend_strategy.py 5 --print

echo "测试 2_5_hybrid_risk_control.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_5_hybrid_risk_control.py 5 --print
