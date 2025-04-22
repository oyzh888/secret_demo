#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# 测试表现最好的策略
echo "测试 1_5_combined_strategy.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/1_5_combined_strategy.py 5 --print

echo "测试 2_3_product_selective_mm.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_3_product_selective_mm.py 5 --print

echo "测试 2_9_volcanic_options_directional_bet.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_9_volcanic_options_directional_bet.py 5 --print

# 测试新的优化策略
echo "测试 3_1_super_combined_strategy.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3_1_super_combined_strategy.py 5 --print

echo "测试 3_2_rainforest_specialist.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3_2_rainforest_specialist.py 5 --print

echo "测试 3_3_dynamic_product_selector.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3_3_dynamic_product_selector.py 5 --print
