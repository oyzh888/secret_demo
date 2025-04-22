from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict

LIMIT = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75
}

PARAM = {
    "tight_spread": 1,        # 基础做市 ±1 tick
    "k_vol": 1.2,             # 波动越大, 报价外扩
    "panic_ratio": 0.8,       # |pos|≥limit×0.8 触发"跳价清仓"
    "panic_add": 4,           # 跳价距离 mid 的额外 ticks
    "mm_size_frac": 0.25,     # 每次挂单为 limit×该系数
    "aggr_take": True         # 是否吃掉对手价
}

def best_bid_ask(depth: OrderDepth) -> Tuple[int|None,int|None]:
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)   # 用于估波动

    def _vol(self, p:str) -> float:
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) or 1

    def _mid(self, depth):
        b,a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None

    def mm_product(self, p:str, depth:OrderDepth, pos:int) -> List[Order]:
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders

        self.prices[p].append(mid)
        spread = int(PARAM["tight_spread"] + PARAM["k_vol"] * self._vol(p))
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"]))

        # panic 强清
        if abs(pos) >= LIMIT[p] * PARAM["panic_ratio"]:
            buy_px = int(mid - PARAM["panic_add"] - spread)
            sell_px = int(mid + PARAM["panic_add"] + spread)
            size = max(size, abs(pos)//2)

        # 吃单
        b,a = best_bid_ask(depth)
        if PARAM["aggr_take"] and b is not None and a is not None:
            if a < mid - spread and pos < LIMIT[p]:
                qty = min(size, LIMIT[p] - pos, abs(depth.sell_orders[a]))
                if qty: orders.append(Order(p, a, qty))
            if b > mid + spread and pos > -LIMIT[p]:
                qty = min(size, LIMIT[p] + pos, depth.buy_orders[b])
                if qty: orders.append(Order(p, b, -qty))

        # 常规挂单
        if pos < LIMIT[p]:
            orders.append(Order(p, buy_px, min(size, LIMIT[p] - pos)))
        if pos > -LIMIT[p]:
            orders.append(Order(p, sell_px, -min(size, LIMIT[p] + pos)))

        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """Main trading logic"""
        result: Dict[str, List[Order]] = {}
        for p, depth in state.order_depths.items():
            if p in LIMIT:  # Only trade products we have limits for
                result[p] = self.mm_product(p, depth, state.position.get(p, 0))

        # No conversions in this strategy
        conversions = 0

        # Return the orders, conversions, and trader data
        return result, conversions, state.traderData