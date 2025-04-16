from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List
import math

class Trader:
    def __init__(self):
        self.TTE = 4/365
        # 每个产品持仓上限
        self.position_limits = {
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200
        }
        # 单次下多少手
        self.spread_lot = 10

    def run(self, state: TradingState):
        """
        Bull Call Spread: Buy VOUCHER_9500, Sell VOUCHER_10000
        条件: 仅在(9500合约 - 10000合约)价差低于某阈值时进行(看涨)
        """
        result = {}

        # 1) 获取 9500 voucher 和 10000 voucher 的 mid price
        buy_sym = "VOLCANIC_ROCK_VOUCHER_9500"
        sell_sym = "VOLCANIC_ROCK_VOUCHER_10000"
        
        if buy_sym not in state.order_depths or sell_sym not in state.order_depths:
            return {}, 0, state.traderData
        
        od_buy = state.order_depths[buy_sym]
        od_sell = state.order_depths[sell_sym]

        # 如果没买/卖报价, 无法成交
        if (not od_buy.buy_orders or not od_buy.sell_orders or
            not od_sell.buy_orders or not od_sell.sell_orders):
            return {}, 0, state.traderData
        
        # 计算 mid price
        buy_v_mid = 0.5*(max(od_buy.buy_orders.keys()) + min(od_buy.sell_orders.keys()))
        sell_v_mid = 0.5*(max(od_sell.buy_orders.keys()) + min(od_sell.sell_orders.keys()))
        
        # 实际当前 spread
        spread_price = buy_v_mid - sell_v_mid
        
        # 2) 根据你的定价逻辑, 估算spread的公允价值. 
        #    这里示例直接用 (S-K_low) - (S-K_high) = (K_high - K_low). 
        #    但真实情况需要更细致(Black-Scholes / 历史经验)
        #    K_low=9500, K_high=10000 => 行权价差=500
        #    naive = 500 * 某 factor
        fair_spread = 150  # 举例: 你觉得 bull call spread 公允价值是 150 Seashells
        
        # 3) 若当前spread 便宜 => 买入spread(= buy low strike + sell high strike)
        #    若不便宜 => pass
        threshold = 20  # 给定一个容忍度
        if spread_price < (fair_spread - threshold):
            # => 做多 bull spread
            # A) 买 voucher_9500
            can_buy_9500 = self.position_limits[buy_sym] - state.position.get(buy_sym, 0)
            can_sell_10000 = self.position_limits[sell_sym] + state.position.get(sell_sym, 0)

            # 每次建的组合数量
            combo_size = min(can_buy_9500, can_sell_10000, self.spread_lot)
            if combo_size > 0:
                # 买 9500 at best ask
                best_ask_9500 = min(od_buy.sell_orders.keys())
                vol_ask_9500 = abs(od_buy.sell_orders[best_ask_9500])
                buy_qty = min(combo_size, vol_ask_9500)

                # 卖 10000 at best bid
                best_bid_10000 = max(od_sell.buy_orders.keys())
                vol_bid_10000 = od_sell.buy_orders[best_bid_10000]
                sell_qty = min(combo_size, vol_bid_10000)

                # 只在都能成交时下单
                if buy_qty>0 and sell_qty>0:
                    result[buy_sym] = [Order(buy_sym, best_ask_9500, buy_qty)]
                    result[sell_sym] = [Order(sell_sym, best_bid_10000, -sell_qty)]
        
        # 注: 若你想做反向(过贵=>做空spread), 或多组合(9750,10250)等，都可类似处理

        return result, 0, state.traderData
