#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Test all strategies
# echo "Testing 1_2_adaptive_mm_arb_cursor.py..."
# prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/1_2_adaptive_mm_arb_cursor.py 5 --print

# echo "Testing 1_3_counterparty_mm_strategy.py..."
# prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/1_3_counterparty_mm_strategy.py 5 --print

echo "Testing 1_4_final_strategy.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/1_4_final_strategy.py 5 --print

# echo "Testing 1_5_combined_strategy.py..."
# prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/1_5_combined_strategy.py 5 --print

# echo "Testing 2_cursor_counterparty_mm.py..."
# prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_cursor_counterparty_mm.py 5 --print

# echo "Testing 2_cursor_volatility_strat.py..."
# prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_cursor_volatility_strat.py 5 --print

# echo "Testing 2_cursor_fee_aware_mm.py..."
# prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_cursor_fee_aware_mm.py 5 --print
