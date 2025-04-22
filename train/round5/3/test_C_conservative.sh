#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Test C-conservative strategy
echo "Testing C-conservative strategy (streak thr = 3，mean‑revert 关闭)..."
prosperity3bt --data /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/eval/imc-prosperity-3-backtester/prosperity3bt/resources /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3/3_C_conservative.py 5
