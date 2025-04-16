from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import math
import numpy as np
from scipy.stats import norm

class Trader:
    def __init__(self):
        self.position_limits = {
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }
        self.size_factor = 0.5  # 激进参数，可调整到0.8
        self.TTE = 4/365  # 剩余时间
        self.risk_multiplier = 3  # 风险乘数

    def bs_call(self, S, K, T, sigma):
        d1 = (np.log(S/K) + (0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S * norm.cdf(d1) - K * norm.cdf(d2)

    def implied_vol(self, S, K, market_price):
        try:
            return self.find_iv(S, K, market_price, precision=0.0001)
        except:
            return None

    def find_iv(self, S, K, price, precision=0.0001):
        iv = 0.3
        max_iter = 100
        for _ in range(max_iter):
            p = self.bs_call(S, K, self.TTE, iv)
            vega = S * norm.pdf( (np.log(S/K) + (0.5*iv**2)*self.TTE) / (iv*np.sqrt(self.TTE)) ) * np.sqrt(self.TTE)
            diff = p - price
            if abs(diff) < precision:
                return iv
            iv -= diff/vega
        return iv

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = ""
        
        # 获取火山石现货价格
        rock_price = next((item.mid_price for item in state.order_depths.values() 
                          if "VOLCANIC_ROCK" in item.symbol), None)
        
        if not rock_price:
            return result, conversions, trader_data
        
        # 分析每个voucher
        iv_data = []
        for symbol, depth in state.order_depths.items():
            if "VOUCHER" not in symbol:
                continue
                
            strike = int(symbol.split("_")[-1])
            mid_price = (min(depth.sell_orders.keys()) + max(depth.buy_orders.keys())) / 2
            
            # 计算隐含波动率
            iv = self.implied_vol(rock_price, strike, mid_price)
            if iv is None:
                continue
                
            moneyness = np.log(strike/rock_price)/np.sqrt(self.TTE)
            iv_data.append((symbol, strike, mid_price, iv, moneyness))
        
        if len(iv_data) < 3:
            return result, conversions, trader_data
        
        # 二次拟合IV曲面
        X = np.array([x[4] for x in iv_data])
        y = np.array([x[3] for x in iv_data])
        coeffs = np.polyfit(X, y, 2)
        iv_fit = np.polyval(coeffs, X)
        
        # 生成交易信号
        signals = []
        for i, (symbol, strike, price, iv, m) in enumerate(iv_data):
            iv_diff = iv - iv_fit[i]
            position = state.position.get(symbol, 0)
            max_pos = self.position_limits[symbol]
            
            # 计算激进仓位
            if iv_diff < -0.05:  # 被低估
                qty = min(int((max_pos - position) * self.size_factor), 
                         int((max_pos - position) * self.risk_multiplier))
                if qty > 0:
                    best_ask = min(depth.sell_orders.keys())
                    signals.append((symbol, best_ask, qty))
                    
            elif iv_diff > 0.05:  # 被高估
                qty = max(int((-max_pos - position) * self.size_factor), 
                         int((-max_pos - position) * self.risk_multiplier))
                if qty < 0:
                    best_bid = max(depth.buy_orders.keys())
                    signals.append((symbol, best_bid, qty))
        
        # 生成订单
        for symbol, price, qty in signals:
            orders = []
            if qty > 0:
                orders.append(Order(symbol, int(price*1.001), qty))  # 激进买入
            else:
                orders.append(Order(symbol, int(price*0.999), qty))  # 激进卖出
            result[symbol] = orders
        
        return result, conversions, trader_data