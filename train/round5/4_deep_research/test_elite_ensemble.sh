#!/bin/bash

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Test elite_ensemble.py
echo "Testing elite_ensemble.py..."
prosperity3bt train/round5/4_deep_research/elite_ensemble.py 5 