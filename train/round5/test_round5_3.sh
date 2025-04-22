#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Test adaptive market making strategy
echo "Testing adaptive_mm_arb.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/adaptive_mm_arb_ac.py 5

# You can add more test cases here