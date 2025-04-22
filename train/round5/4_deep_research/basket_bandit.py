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

class BasketBanditTrader:
    TAG = "BB"
    AGGRESSIVE_FACTOR = 1.0
    _THRESH = 0.01

    COMP_MAP = {
        "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
        "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2},
    }

    def _fv(self, basket: str, depth: Dict[str, OrderDepth]):
        w = self.COMP_MAP[basket]
        total = 0
        for p, qty in w.items():
            b, _, a, _ = best_bid_ask(depth[p])
            if b is None or a is None:
                return None
            total += qty * (b + a) / 2
        return total

    def strength(self, state: TradingState) -> float:
        s = 0.0
        depth = state.order_depths
        for b in self.COMP_MAP:
            fv = self._fv(b, depth)
            if fv is None:
                continue
            bid, _, ask, _ = best_bid_ask(depth[b])
            if bid and abs(fv - bid) / fv > self._THRESH:
                s = max(s, min(1.0, abs(fv - bid) / (fv * 0.05)))
        return s

    def run(self, state: TradingState):
        res: Dict[str, List[Order]] = {}
        depth = state.order_depths
        for basket in self.COMP_MAP:
            fv = self._fv(basket, depth)
            if fv is None:
                continue
            bid, bid_q, ask, ask_q = best_bid_ask(depth[basket])
            pos = state.position.get(basket, 0)
            limit = POSITION_LIMITS[basket]
            lst: List[Order] = []
            if ask and (fv - ask) / fv > self._THRESH:  # discount
                q = clamp(int(0.5 * limit * self.AGGRESSIVE_FACTOR), limit - pos)
                if q:
                    lst.append(Order(basket, ask, q))
            if bid and (bid - fv) / fv > self._THRESH:  # premium
                q = clamp(int(0.5 * limit * self.AGGRESSIVE_FACTOR), -limit - pos)
                if q:
                    lst.append(Order(basket, bid, q))
            if lst:
                res[basket] = lst
        return res, 0, self.TAG 

class Trader:
    def __init__(self):
        self.trader = BasketBanditTrader()

    def run(self, state: TradingState):
        return self.trader.run(state) 