from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from datamodel import OrderDepth, Order, TradingState, Trade

# Shared utils
POSITION_LIMITS: Dict[str, int] = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75,
}

def clamp(qty: int, limit: int) -> int:
    """Ensure resulting position stays within ±limit."""
    return max(-limit, min(limit, qty))

def best_bid_ask(depth: OrderDepth) -> Tuple[int | None, int | None, int | None, int | None]:
    bid_p = bid_q = ask_p = ask_q = None
    if depth.buy_orders:
        bid_p, bid_q = max(depth.buy_orders.items(), key=lambda x: x[0])
    if depth.sell_orders:
        ask_p, ask_q = min(depth.sell_orders.items(), key=lambda x: x[0])
    return bid_p, bid_q, ask_p, ask_q

class GammaGorillaTrader:
    TAG = "GG"
    AGGRESSIVE_FACTOR = 1.0
    VOUCHERS = [
        "VOLCANIC_ROCK_VOUCHER_9500",
        "VOLCANIC_ROCK_VOUCHER_9750",
        "VOLCANIC_ROCK_VOUCHER_10000",
        "VOLCANIC_ROCK_VOUCHER_10250",
        "VOLCANIC_ROCK_VOUCHER_10500",
    ]

    def __init__(self):
        self.last_mid: float | None = None

    def strength(self, state: TradingState) -> float:
        rock_depth = state.order_depths["VOLCANIC_ROCK"]
        b, _, a, _ = best_bid_ask(rock_depth)
        if b is None or a is None:
            return 0.0
        mid = (a + b) / 2
        if self.last_mid is None:
            self.last_mid = mid
            return 0.0
        vol = abs(mid - self.last_mid) / self.last_mid
        self.last_mid = mid
        return min(1.0, vol * 50)  # scale: 0.02 move → strength 1

    def run(self, state: TradingState):
        res: Dict[str, List[Order]] = {}
        rock_depth = state.order_depths["VOLCANIC_ROCK"]
        bid_r, _, ask_r, _ = best_bid_ask(rock_depth)
        if bid_r is None or ask_r is None:
            return res, 0, self.TAG
        mid = (bid_r + ask_r) / 2
        for v in self.VOUCHERS:
            d = state.order_depths[v]
            bid, _, ask, _ = best_bid_ask(d)
            if bid is None or ask is None:
                continue
            strike = int(v.split("_")[-1])
            intrinsic = max(0, mid - strike)
            theo = intrinsic + 50
            pos = state.position.get(v, 0)
            limit = POSITION_LIMITS[v]
            lst: List[Order] = []
            if ask < theo - 20:
                q = clamp(int(0.4 * limit * self.AGGRESSIVE_FACTOR), limit - pos)
                if q:
                    lst.append(Order(v, ask, q))
            if bid > theo + 20:
                q = clamp(int(0.4 * limit * self.AGGRESSIVE_FACTOR), -limit - pos)
                if q:
                    lst.append(Order(v, bid, q))
            if lst:
                res[v] = lst
        return res, 0, self.TAG

class Trader:
    def __init__(self):
        self.trader = GammaGorillaTrader()

    def run(self, state: TradingState):
        return self.trader.run(state) 