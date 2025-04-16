from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import math
import numpy as np

class Trader:
    def __init__(self):
        self.TTE = 4 / 365  # 4 days to expiry from Round 3
        self.position_limits = {
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
        }
        self.size_factor = 0.2

    def bs_call_price(self, S, K, T, sigma):
        d1 = (math.log(S/K) + 0.5*sigma**2*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm_cdf(d1) - K * norm_cdf(d2)

    def implied_volatility(self, S, K, T, market_price):
        try:
            a, b = 0.01, 3.0
            for _ in range(20):
                mid = (a + b) / 2
                price = self.bs_call_price(S, K, T, mid)
                if price > market_price:
                    b = mid
                else:
                    a = mid
            return (a + b) / 2
        except:
            return None

    def run(self, state: TradingState):
        result = {}
        conversions = 0
        rock_price = None
        voucher_data = []

        if "VOLCANIC_ROCK" in state.order_depths:
            rock_depth = state.order_depths["VOLCANIC_ROCK"]
            if rock_depth.buy_orders and rock_depth.sell_orders:
                best_bid = max(rock_depth.buy_orders.keys())
                best_ask = min(rock_depth.sell_orders.keys())
                rock_price = (best_bid + best_ask) / 2

        if rock_price is None:
            return {}, conversions, "NO_ROCK_PRICE"

        for product in state.order_depths:
            if "VOUCHER" not in product:
                continue

            order_depth: OrderDepth = state.order_depths[product]
            if not order_depth.sell_orders:
                continue

            strike = int(product.split("_")[-1])
            best_ask = min(order_depth.sell_orders.keys())
            premium = best_ask

            m = math.log(strike / rock_price) / math.sqrt(self.TTE)
            iv = self.implied_volatility(rock_price, strike, self.TTE, premium)

            if iv is None:
                continue

            voucher_data.append((product, strike, premium, m, iv))

        if len(voucher_data) >= 3:
            m_arr = np.array([x[3] for x in voucher_data])
            iv_arr = np.array([x[4] for x in voucher_data])
            fit = np.polyfit(m_arr, iv_arr, 2)
            v_fitted = np.polyval(fit, m_arr)
            iv_diff = iv_arr - v_fitted

            # ⚠️ 反转原策略逻辑：
            # 买入原本“被高估”的 → 利用趋势继续拉高
            # 卖出原本“被低估”的 → 趋势性做空波动修复慢的票
            idx_buy = np.argmax(iv_diff)  # 原本是 np.argmin
            idx_sell = np.argmin(iv_diff)  # 原本是 np.argmax

            buy_prod = voucher_data[idx_buy][0]
            sell_prod = voucher_data[idx_sell][0]
            buy_price = voucher_data[idx_buy][2]
            sell_price = voucher_data[idx_sell][2]

            buy_limit = self.position_limits[buy_prod]
            sell_limit = self.position_limits[sell_prod]
            pos_buy = state.position.get(buy_prod, 0)
            pos_sell = state.position.get(sell_prod, 0)

            max_buy_qty = int((buy_limit - pos_buy) * self.size_factor)
            max_sell_qty = int((sell_limit + pos_sell) * self.size_factor)

            orders = []
            if max_buy_qty > 0:
                orders.append(Order(buy_prod, buy_price, max_buy_qty))
            if max_sell_qty > 0:
                orders.append(Order(sell_prod, sell_price, -max_sell_qty))

            if orders:
                result[buy_prod] = [o for o in orders if o.symbol == buy_prod]
                result[sell_prod] = [o for o in orders if o.symbol == sell_prod]

        return result, conversions, "vol_skew_reversed"

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2
