from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List
import math
import numpy as np
from math import log, sqrt
from statistics import NormalDist

def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return NormalDist(mu=0, sigma=1).cdf(x)

class Trader:
    def __init__(self):
        # Round 3 剩余 4 天到期
        self.TTE = 4.0 / 365.0
        
        # 题目给的各期权持仓上限
        self.position_limits = {
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }
        
        # 每轮用多大比例仓位进行交易（可调）
        self.size_factor = 0.4

    def bs_call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Black–Scholes 欧式看涨期权定价(忽略利率与股息)。"""
        if T <= 0 or S <= 0 or K <= 0 or sigma < 1e-8:
            return max(S - K, 0)
        d1 = (log(S/K) + 0.5*sigma**2*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        return S*norm_cdf(d1) - K*norm_cdf(d2)

    def implied_volatility(self, S: float, K: float, T: float, market_price: float) -> float:
        """简单的二分搜索推算隐含波动率。"""
        intrinsic_val = max(S - K, 0)
        if market_price < intrinsic_val:
            return None  # 市价低于内在价值，说明无解
        
        low, high = 1e-5, 3.0
        for _ in range(25):
            mid = 0.5*(low+high)
            guess_price = self.bs_call_price(S, K, T, mid)
            if guess_price > market_price:
                high = mid
            else:
                low = mid
        return 0.5*(low+high)

    def run(self, state: TradingState):
        """
        每当交易环境调用 run() 时:
          1) 确定火山石VOLCANIC_ROCK的中间价S_mid
          2) 为每个voucher计算Implied Vol
          3) 对IV vs. m做二次拟合
          4) 对比iv_diff决定买卖
        """
        orders_to_return = {}
        
        # step1: 计算 VOLCANIC_ROCK 的中价
        rock_symbol = "VOLCANIC_ROCK"
        if rock_symbol not in state.order_depths:
            return {}, 0, state.traderData
        
        rock_depth = state.order_depths[rock_symbol]
        if not rock_depth.buy_orders or not rock_depth.sell_orders:
            return {}, 0, state.traderData
        
        best_bid = max(rock_depth.buy_orders.keys())
        best_ask = min(rock_depth.sell_orders.keys())
        S_mid = 0.5*(best_bid+best_ask)

        # step2: 收集voucher数据 & 计算IV
        vouchers = [
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000",
            "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        data_list = []
        for v in vouchers:
            if v not in state.order_depths: 
                continue
            od_v = state.order_depths[v]
            if not od_v.buy_orders or not od_v.sell_orders:
                continue
            # mid price of voucher
            voucher_bid = max(od_v.buy_orders.keys())
            voucher_ask = min(od_v.sell_orders.keys())
            premium_mid = 0.5*(voucher_bid+voucher_ask)

            # 提取strike
            try:
                strike_val = int(v.split('_')[-1])
            except:
                continue

            iv = self.implied_volatility(S_mid, strike_val, self.TTE, premium_mid)
            if iv is None:
                continue
            m_t = log(strike_val/S_mid)/sqrt(self.TTE)
            data_list.append((v, strike_val, premium_mid, m_t, iv))

        if len(data_list) < 2:
            return {}, 0, state.traderData

        # step3: 二次拟合
        xvals = np.array([x[3] for x in data_list])  # m_t
        yvals = np.array([x[4] for x in data_list])  # iv
        coef = np.polyfit(xvals, yvals, 2)  # [a,b,c]
        
        # step4: 计算 iv_diff
        diffs = []
        for (v, Kval, pmid, mt, iv_real) in data_list:
            iv_fit = np.polyval(coef, mt)
            diff = iv_real - iv_fit
            diffs.append((v, Kval, pmid, iv_real, iv_fit, diff))
        diffs.sort(key=lambda x: x[5])  # 按 diff 排序
        
        buy_cands = diffs[:2]    # 取IV diff最负的2个做多
        sell_cands = diffs[-2:]  # 取IV diff最正的2个做空
        
        result_orders = {}
        
        # 买入逻辑
        for cand in buy_cands:
            symbol, Kval, pmid, iv_r, iv_f, diff = cand
            if diff >= 0:
                break
            pos = state.position.get(symbol, 0)
            cap_long = self.position_limits[symbol] - pos
            if cap_long <= 0:
                continue
            trade_amt = int(cap_long * self.size_factor)
            if trade_amt <= 0:
                continue
            # buy at best ask
            dep_v = state.order_depths[symbol]
            best_ask_v = min(dep_v.sell_orders.keys())
            vol_at_ask = abs(dep_v.sell_orders[best_ask_v])
            qty = min(trade_amt, vol_at_ask)
            if qty > 0:
                if symbol not in result_orders:
                    result_orders[symbol] = []
                result_orders[symbol].append(Order(symbol, best_ask_v, qty))

        # 卖出逻辑
        for cand in reversed(sell_cands):
            symbol, Kval, pmid, iv_r, iv_f, diff = cand
            if diff <= 0:
                break
            pos = state.position.get(symbol, 0)
            cap_short = self.position_limits[symbol] + pos
            if cap_short <= 0:
                continue
            trade_amt = int(cap_short * self.size_factor)
            if trade_amt <= 0:
                continue
            # sell at best bid
            dep_v = state.order_depths[symbol]
            best_bid_v = max(dep_v.buy_orders.keys())
            vol_at_bid = dep_v.buy_orders[best_bid_v]
            qty = min(trade_amt, vol_at_bid)
            if qty > 0:
                if symbol not in result_orders:
                    result_orders[symbol] = []
                result_orders[symbol].append(Order(symbol, best_bid_v, -qty))

        return result_orders, 0, state.traderData
