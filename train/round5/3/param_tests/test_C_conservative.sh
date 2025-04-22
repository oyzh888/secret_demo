#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo

# Copy the parameter file to uploaded_params.json
cp /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3/param_tests/params_C_conservative.json /Users/zhihaoouyang/Desktop/code/imc_prosperity/uploaded_params.json

# Run the test
echo "Testing Parameter Set C (Conservative)..."
prosperity3bt --data /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/eval/imc-prosperity-3-backtester/prosperity3bt/resources /Users/zhihaoouyang/Desktop/code/imc_prosperity/secret_demo/train/round5/3/param_tests/trader_base.py 5-2 5-3 5-4
