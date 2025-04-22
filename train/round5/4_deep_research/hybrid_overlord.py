from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from datamodel import OrderDepth, Order, TradingState, Trade
from momentum_ripper import MomentumRipperTrader
from basket_bandit import BasketBanditTrader
from gamma_gorilla import GammaGorillaTrader
from sunlight_sniper import SunlightSniperTrader

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

class HybridOverlordTrader:
    """Routes capital to the strongest sub‑agents each iteration."""

    def __init__(self):
        self.agents = [MomentumRipperTrader(), BasketBanditTrader(), GammaGorillaTrader(), SunlightSniperTrader()]
        self.last_weights = np.ones(len(self.agents)) / len(self.agents)

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        s = arr.sum()
        return arr / s if s > 0 else np.ones_like(arr) / len(arr)

    def run(self, state: TradingState):
        strengths = np.array([agent.strength(state) for agent in self.agents])
        # exponential moving average on strengths for smoother weights
        self.last_weights = 0.5 * self.last_weights + 0.5 * self._normalise(strengths + 1e-6)
        # pick orders, scale quantities by weight per agent
        combined: Dict[str, List[Order]] = {}
        for w, agent in zip(self.last_weights, self.agents):
            if w < 0.05:  # ignore very weak agent this step
                continue
            orders, _, _ = agent.run(state)
            for product, olist in orders.items():
                scaled: List[Order] = []
                for o in olist:
                    limit = POSITION_LIMITS[product]
                    scaled_qty = int(o.quantity * w)
                    if scaled_qty == 0:
                        continue
                    # respect position limits by naive Clamp
                    pos = state.position.get(product, 0)
                    scaled_qty = clamp(pos + scaled_qty, limit) - pos
                    if scaled_qty != 0:
                        scaled.append(Order(product, o.price, scaled_qty))
                if scaled:
                    combined.setdefault(product, []).extend(scaled)
        return combined, 0, "HO"

class Trader:
    def __init__(self):
        self.trader = HybridOverlordTrader()

    def run(self, state: TradingState):
        return self.trader.run(state) 