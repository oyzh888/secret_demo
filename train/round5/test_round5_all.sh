#!/bin/bash

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Test 1_2_adaptive_mm_arb_cursor.py
echo "Testing 1_2_adaptive_mm_arb_cursor.py..."
prosperity3bt train/round5/1_2_adaptive_mm_arb_cursor.py 1 2 3 4 5

# Test 1_1_adaptive_mm_arb_ac.py
echo "Testing 1_1_adaptive_mm_arb_ac.py..."
prosperity3bt train/round5/1_1_adaptive_mm_arb_ac.py 1 2 3 4 5

# Test 11_1_aggressive_mm.py
echo "Testing 11_1_aggressive_mm.py..."
prosperity3bt train/round5/11_1_aggressive_mm.py 1 2 3 4 5