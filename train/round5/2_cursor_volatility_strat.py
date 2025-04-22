from typing import Dict, List, Tuple, Optional
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np

# Product limits (same as base)
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

# Volatility Tiers based on analysis (Standard Deviation)
VOLATILITY_TIERS = {
    # High Volatility (σ > 100)
    "PICNIC_BASKET1": "high",
    "VOLCANIC_ROCK": "high",
    "VOLCANIC_ROCK_VOUCHER_9500": "high",
    "VOLCANIC_ROCK_VOUCHER_9750": "high",
    # Medium Volatility (20 < σ <= 100)
    "MAGNIFICENT_MACARONS": "medium",
    "VOLCANIC_ROCK_VOUCHER_10000": "medium",
    "SQUID_INK": "medium",
    "PICNIC_BASKET2": "medium",
    "DJEMBES": "medium",
    "JAMS": "medium",
    "VOLCANIC_ROCK_VOUCHER_10250": "medium",
    # Low Volatility (σ <= 20)
    "CROISSANTS": "low",
    "KELP": "low",
    "VOLCANIC_ROCK_VOUCHER_10500": "low",
    "RAINFOREST_RESIN": "low",
}

# Parameters for different volatility tiers
PARAM = {
    "low": {
        "tight_spread": 1,
        "k_vol": 0.5, # Less sensitive to vol
        "mm_size_frac": 0.3, # Maybe slightly larger size for stable products
        "aggr_take": False # Less aggressive for low vol
    },
    "medium": {
        "tight_spread": 1,
        "k_vol": 1.0,
        "mm_size_frac": 0.25,
        "aggr_take": True,
        "ema_short": 5, # For simple trend filter
        "ema_long": 15
    },
    "high": {
        "tight_spread": 2, # Wider base spread
        "k_vol": 1.5,
        "mm_size_frac": 0.2, # Smaller size for high vol
        "aggr_take": True,
        "momentum_window": 5 # Look at recent price change
    },
    "panic_ratio": 0.8,
    "panic_add": 4,
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.ema_short = defaultdict(lambda: None)
        self.ema_long = defaultdict(lambda: None)

    def _update_indicators(self, p: str, mid: float):
        self.prices[p].append(mid)
        prices_arr = np.array(self.prices[p])
        if len(prices_arr) >= PARAM["medium"]["ema_short"]:
            self.ema_short[p] = np.mean(prices_arr[-PARAM["medium"]["ema_short"]:])
        if len(prices_arr) >= PARAM["medium"]["ema_long"]:
            self.ema_long[p] = np.mean(prices_arr[-PARAM["medium"]["ema_long"]:])

    def _vol(self, p:str) -> float:
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) or 1

    def _mid(self, depth):
        b,a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None

    def run_strategy(self, p: str, tier: str, depth: OrderDepth, pos: int) -> List[Order]:
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders

        self._update_indicators(p, mid)
        tier_params = PARAM[tier]
        vol = self._vol(p)

        spread = int(tier_params["tight_spread"] + tier_params["k_vol"] * vol)
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        size = max(1, int(LIMIT[p] * tier_params["mm_size_frac"]))

        # --- Strategy Logic based on Volatility Tier ---
        if tier == "low":
            # Pure Market Making (basic spread)
            pass # Prices already set

        elif tier == "medium":
            # MM + Simple Trend Filter (EMA Crossover)
            if self.ema_short[p] and self.ema_long[p]:
                if self.ema_short[p] > self.ema_long[p]: # Up-trend
                    buy_px = int(mid - spread + 1) # Slightly more aggressive buy
                elif self.ema_short[p] < self.ema_long[p]: # Down-trend
                    sell_px = int(mid + spread - 1) # Slightly more aggressive sell

        elif tier == "high":
            # Simple Momentum (trade in direction of recent price change)
            if len(self.prices[p]) >= tier_params["momentum_window"] + 1:
                price_change = self.prices[p][-1] - self.prices[p][-tier_params["momentum_window"]-1]
                if price_change > 0: # Price increased, lean towards buying
                    buy_px = int(mid - spread + 1)
                elif price_change < 0: # Price decreased, lean towards selling
                    sell_px = int(mid + spread - 1)

        # --- Common Logic (Panic, Aggression, Order Placement) ---
        if abs(pos) >= LIMIT[p] * PARAM["panic_ratio"]:
            buy_px = int(mid - PARAM["panic_add"] - spread)
            sell_px = int(mid + PARAM["panic_add"] + spread)
            size = max(size, abs(pos)//2)

        b,a = best_bid_ask(depth)
        if tier_params["aggr_take"] and b is not None and a is not None:
            if a < sell_px and pos < LIMIT[p]:
                qty = min(size, LIMIT[p] - pos, abs(depth.sell_orders[a]))
                if qty > 0: orders.append(Order(p, a, qty))
            if b > buy_px and pos > -LIMIT[p]:
                qty = min(size, LIMIT[p] + pos, depth.buy_orders[b])
                if qty > 0: orders.append(Order(p, b, -qty))

        if pos < LIMIT[p]:
            orders.append(Order(p, buy_px, min(size, LIMIT[p] - pos)))
        if pos > -LIMIT[p]:
            orders.append(Order(p, sell_px, -min(size, LIMIT[p] + pos)))

        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        for p, depth in state.order_depths.items():
            if p in LIMIT and p in VOLATILITY_TIERS:
                tier = VOLATILITY_TIERS[p]
                result[p] = self.run_strategy(p, tier, depth, state.position.get(p, 0))

        conversions = 0
        return result, conversions, state.traderData