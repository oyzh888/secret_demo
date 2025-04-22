#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Test X-tight strategy
echo "Testing X-tight strategy (让买/卖价都跨1 tick)..."
prosperity3bt --data /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/eval/imc-prosperity-3-backtester/prosperity3bt/resources /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3/3_X_tight.py 5
