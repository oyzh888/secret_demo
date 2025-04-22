#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Test counterparty market making strategy
echo "Testing 2_cursor_counterparty_mm.py..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/2_cursor_counterparty_mm.py 5
