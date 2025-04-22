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

class MomentumRipperTrader:
    TAG = "MR"
    AGGRESSIVE_FACTOR = 1.0
    _STREAK_MIN = 3  # trigger length

    def __init__(self):
        self.streak: Dict[str, Tuple[str | None, int]] = {}

    def strength(self, state: TradingState) -> float:
        # Strength = max streak length across products (scaled 0‑1)
        s = 0.0
        for p, trades in state.market_trades.items():
            side, length = self.streak.get(p, (None, 0))
            if length > 0:
                s = max(s, min(1.0, length / 5))
        return s

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        for product, depth in state.order_depths.items():
            bid_p, bid_q, ask_p, ask_q = best_bid_ask(depth)
            if bid_p is None or ask_p is None:
                continue
            # update streak table
            recent = state.market_trades.get(product, [])
            side, length = self.streak.get(product, (None, 0))
            for t in recent:
                if t.buyer == "SUBMISSION" or t.seller == "SUBMISSION":
                    continue
                s = "BUY" if t.buyer in ("Caesar", "Camilla") else "SELL" if t.seller in ("Caesar", "Camilla") else None
                if not s:
                    continue
                if s == side:
                    length += 1
                else:
                    side, length = s, 1
            self.streak[product] = (side, length)
            pos = state.position.get(product, 0)
            limit = POSITION_LIMITS[product]
            basket: List[Order] = []
            if length >= self._STREAK_MIN:
                qty = clamp(int(0.3 * limit * self.AGGRESSIVE_FACTOR), limit - pos if side == "BUY" else -limit - pos)
                if qty:
                    px = ask_p + 1 if side == "BUY" else bid_p - 1
                    basket.append(Order(product, px, qty))
            if basket:
                orders[product] = basket
        return orders, 0, self.TAG

class Trader:
    def __init__(self):
        self.trader = MomentumRipperTrader()

    def run(self, state: TradingState):
        return self.trader.run(state) 