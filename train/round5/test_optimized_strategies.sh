#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# 测试所有优化策略
echo "测试 3_1_super_combined_strategy.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3_1_super_combined_strategy.py 5 --print

echo "测试 3_2_rainforest_specialist.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3_2_rainforest_specialist.py 5 --print

echo "测试 3_3_dynamic_product_selector.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3_3_dynamic_product_selector.py 5 --print

echo "测试 3_4_enhanced_directional_options.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3_4_enhanced_directional_options.py 5 --print

echo "测试 3_5_adaptive_risk_strategy.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3_5_adaptive_risk_strategy.py 5 --print
