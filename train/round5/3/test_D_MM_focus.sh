#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Test D-MM focus strategy
echo "Testing D-MM focus strategy (关掉 momentum，只测做市)..."
prosperity3bt --data /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/eval/imc-prosperity-3-backtester/prosperity3bt/resources /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3/3_D_MM_focus.py 5
