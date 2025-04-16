from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List
import math

class Trader:
    def __init__(self):
        self.TTE = 4.0 / 365.0  # Round 3 还剩 4天
        self.alpha = 0.1       # 时间价值系数，可再调
        self.position_limits = {
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }
        self.size_factor = 0.5  # 每次操作占可用仓位 50%

    def run(self, state: TradingState):
        result_orders = {}

        # 1) 获取 VOLCANIC_ROCK 的 mid price
        rock_sym = "VOLCANIC_ROCK"
        if rock_sym not in state.order_depths:
            return {}, 0, state.traderData
        
        od_rock = state.order_depths[rock_sym]
        if not od_rock.buy_orders or not od_rock.sell_orders:
            return {}, 0, state.traderData
        
        best_bid = max(od_rock.buy_orders.keys())
        best_ask = min(od_rock.sell_orders.keys())
        S_mid = 0.5*(best_bid + best_ask)

        # 2) 遍历每个voucher，根据 (S - K) + alpha * sqrt(TTE) 与其 mid price 比较
        vouchers = [
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000",
            "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]

        for v in vouchers:
            if v not in state.order_depths:
                continue
            od_v = state.order_depths[v]
            if not od_v.buy_orders or not od_v.sell_orders:
                continue
            
            # voucher mid price
            v_bid = max(od_v.buy_orders.keys())
            v_ask = min(od_v.sell_orders.keys())
            premium_mid = 0.5*(v_bid + v_ask)

            # 解析 strike
            try:
                strike_price = int(v.split('_')[-1])
            except:
                continue

            intrinsic_val = S_mid - strike_price
            time_val_est = self.alpha * math.sqrt(self.TTE) * S_mid  # 简单估计

            # 给定一个目标估值:
            fair_approx = max(intrinsic_val, 0) + time_val_est

            # 判断
            if premium_mid < (fair_approx - 20):
                # 明显低估 => BUY
                pos = state.position.get(v, 0)
                cap_long = self.position_limits[v] - pos
                if cap_long > 0:
                    amt_buy = int(cap_long * self.size_factor)
                    if amt_buy > 0:
                        best_ask_v = min(od_v.sell_orders.keys())
                        vol_ask = abs(od_v.sell_orders[best_ask_v])
                        qty = min(amt_buy, vol_ask)
                        if qty > 0:
                            result_orders.setdefault(v, []).append(Order(v, best_ask_v, qty))

            elif premium_mid > (fair_approx + 20):
                # 明显高估 => SELL
                pos = state.position.get(v, 0)
                cap_short = self.position_limits[v] + pos
                if cap_short > 0:
                    amt_sell = int(cap_short * self.size_factor)
                    if amt_sell > 0:
                        best_bid_v = max(od_v.buy_orders.keys())
                        vol_bid = od_v.buy_orders[best_bid_v]
                        qty = min(amt_sell, vol_bid)
                        if qty > 0:
                            result_orders.setdefault(v, []).append(Order(v, best_bid_v, -qty))
            
            # 否则不交易

        return result_orders, 0, state.traderData
