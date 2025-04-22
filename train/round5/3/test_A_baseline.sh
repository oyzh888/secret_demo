#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Test A-baseline strategy
echo "Testing A-baseline strategy (被动做市 + streak thr = 2，size_frac = 0.10)..."
prosperity3bt --data /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/eval/imc-prosperity-3-backtester/prosperity3bt/resources /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3/3_A_baseline.py 5
