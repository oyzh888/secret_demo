import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def run_strategy_test(days):
    """运行策略测试并收集结果"""
    day_args = " ".join([f"--day {day}" for day in days])
    cmd = f"python -m prosperity3bt --trader train/round5/2_7_volatility_scalping.py {day_args}"
    
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running strategy: {result.stderr}")
        return None
    
    print("Strategy test completed successfully")
    return result.stdout

def parse_results(output):
    """解析测试结果输出"""
    # 提取PnL和日期信息
    pnl_data = []
    for line in output.split('\n'):
        if "PnL" in line:
            parts = line.split()
            try:
                day = int(parts[1].replace("day", ""))
                pnl = float(parts[3])
                pnl_data.append((day, pnl))
            except (IndexError, ValueError):
                continue
    
    return pnl_data

def visualize_results(pnl_data):
    """可视化策略结果"""
    if not pnl_data:
        print("No PnL data to visualize")
        return
    
    days, pnls = zip(*pnl_data)
    
    plt.figure(figsize=(10, 6))
    plt.bar(days, pnls)
    plt.title('Volatility Scalping Strategy - Daily PnL')
    plt.xlabel('Trading Day')
    plt.ylabel('PnL')
    plt.grid(True, alpha=0.3)
    
    # 添加总PnL信息
    total_pnl = sum(pnls)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.text(0.02, 0.95, f'Total PnL: {total_pnl:.2f}', transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'vol_scalp_results_{timestamp}.png')
    plt.close()
    
    print(f"Results visualization saved as vol_scalp_results_{timestamp}.png")
    print(f"Total PnL across all days: {total_pnl:.2f}")
    
    # 每日PnL分析
    print("\nDaily PnL Analysis:")
    print("-" * 30)
    for day, pnl in pnl_data:
        print(f"Day {day}: {pnl:.2f}")

def analyze_strategy_performance():
    """分析策略表现"""
    days_to_test = [1, 2, 3, 4, 5]
    
    # 运行策略测试
    output = run_strategy_test(days_to_test)
    if not output:
        return
    
    # 解析并可视化结果
    pnl_data = parse_results(output)
    visualize_results(pnl_data)

if __name__ == "__main__":
    print("Starting Volatility Scalping Strategy Analysis")
    analyze_strategy_performance() 