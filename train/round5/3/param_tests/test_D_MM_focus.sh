#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Copy the parameter file to uploaded_params.json
cp /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3/param_tests/params_D_MM_focus.json /Users/zhihaoouyang/Desktop/code/imc_prosperity/uploaded_params.json

# Run the test
echo "Testing Parameter Set D (MM Focus)..."
prosperity3bt /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3/param_tests/trader_base.py 5-2 5-3 5-4
