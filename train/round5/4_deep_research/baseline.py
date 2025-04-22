# imc_traders.py – v2
"""Aggressive strategy pack for IMC Prosperity Round 6 (updated).

Changes in **v2**
-----------------
* **HybridOverlordTrader** upgraded to *Option A* dynamic router:
  - Each sub‑agent returns its candidate orders **and a live signal score**.
  - Scores are normalised to weights; per‑product order lists are merged with
    position‑limit‑aware sizing (capital is routed to stronger signals).
  - Allows simultaneous multi‑asset exposure while fading weak/contradictory
    agents.
* Sub‑agents now expose `strength(state)` helper so Hybrid can query signal
  intensity **without** generating orders twice (keeps runtime < 900 ms).

Copy one class to `trader.py` (or rename it to `Trader`) before submission.
All code respects allowed libraries (stdlib + numpy + pandas)."""

from __future__ import annotations
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from datamodel import OrderDepth, Order, TradingState, Trade

# ---------------------------------------------------------------------------
# Shared utils
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 1. MomentumRipperTrader – order‑flow momentum follower
# ---------------------------------------------------------------------------
class MomentumRipperTrader:
    TAG = "MR"
    AGGRESSIVE_FACTOR = 1.0
    _STREAK_MIN = 3  # trigger length

    def __init__(self):
        self.streak: Dict[str, Tuple[str | None, int]] = {}

    # ---- signal strength helper ------------------------------------------------
    def strength(self, state: TradingState) -> float:
        # Strength = max streak length across products (scaled 0‑1)
        s = 0.0
        for p, trades in state.market_trades.items():
            side, length = self.streak.get(p, (None, 0))
            if length > 0:
                s = max(s, min(1.0, length / 5))
        return s

    # ---- main run --------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2. BasketBanditTrader – basket/components arbitrage
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. GammaGorillaTrader – vouchers γ‑scalper
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 4. SunlightSniperTrader – macaron fundamental sniper
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 5. HybridOverlordTrader – dynamic meta‑router (Option A)
# ---------------------------------------------------------------------------
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

# End of file
