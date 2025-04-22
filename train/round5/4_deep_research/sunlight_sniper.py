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

class SunlightSniperTrader:
    TAG = "SS"
    AGGRESSIVE_FACTOR = 1.0
    CSI = 750

    def strength(self, state: TradingState) -> float:
        obs = state.observations.conversionObservations.get("MAGNIFICENT_MACARONS")
        if not obs:
            return 0.0
        return min(1.0, abs(self.CSI - obs.sunlightIndex) / 200)

    def run(self, state: TradingState):
        res: Dict[str, List[Order]] = {}
        obs = state.observations.conversionObservations.get("MAGNIFICENT_MACARONS")
        if not obs:
            return res, 0, self.TAG
        sun = obs.sunlightIndex
        sugar = obs.sugarPrice
        depth = state.order_depths["MAGNIFICENT_MACARONS"]
        bid, _, ask, _ = best_bid_ask(depth)
        if bid is None or ask is None:
            return res, 0, self.TAG
        fair = 10000 + 2 * sugar + (self.CSI - sun) * 10
        pos = state.position.get("MAGNIFICENT_MACARONS", 0)
        limit = POSITION_LIMITS["MAGNIFICENT_MACARONS"]
        lst: List[Order] = []
        if sun < self.CSI and ask < fair - 100:
            q = clamp(int(0.6 * limit * self.AGGRESSIVE_FACTOR), limit - pos)
            if q:
                lst.append(Order("MAGNIFICENT_MACARONS", ask, q))
        if sun > self.CSI and bid > fair + 100:
            q = clamp(int(0.6 * limit * self.AGGRESSIVE_FACTOR), -limit - pos)
            if q:
                lst.append(Order("MAGNIFICENT_MACARONS", bid, q))
        if lst:
            res["MAGNIFICENT_MACARONS"] = lst
        return res, 0, self.TAG

class Trader:
    def __init__(self):
        self.trader = SunlightSniperTrader()

    def run(self, state: TradingState):
        return self.trader.run(state) 