from typing import Dict, List, Tuple, Optional
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict

# Define OwnTrade class if not available in datamodel
class OwnTrade:
    def __init__(self, symbol: str, price: int, quantity: int, counter_party: Optional[str] = None):
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.counter_party = counter_party
        self.timestamp = 0

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

# Parameters including fee awareness
PARAM = {
    "tight_spread": 1,        # Base spread
    "k_vol": 1.2,
    "panic_ratio": 0.8,
    "panic_add": 4,
    "mm_size_frac": 0.25,
    "aggr_take": True,
    "fee_vol_tiers": [100, 200, 400], # Volume tiers for fee increase
    "fee_edge_add": [0, 1, 3, 5]      # Corresponding edge increase (0%, 0.2%, 0.5%, 1% fee)
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.daily_volume = defaultdict(int) # Track daily traded volume per product
        self.current_day = -1 # Track current day to reset volume

    def _update_daily_volume(self, state: TradingState):
        # Reset volume at the start of a new day
        day = state.timestamp // 1_000_000 # Assuming timestamp reflects day
        if day != self.current_day:
            self.daily_volume.clear()
            self.current_day = day

        # Add volume from own trades in this timestamp
        for product, trades in state.own_trades.items():
            for trade in trades:
                # Handle different trade formats
                if hasattr(trade, 'quantity'):
                    self.daily_volume[product] += abs(trade.quantity)
                # In backtester, the trade might be a tuple
                elif isinstance(trade, tuple) and len(trade) >= 4:
                    # Assuming format is (symbol, buyer/seller, price, quantity, ...)
                    self.daily_volume[product] += abs(trade[3])

    def _get_fee_tier_edge(self, p: str) -> int:
        volume = self.daily_volume[p]
        tier_index = 0
        for i, tier_vol in enumerate(PARAM["fee_vol_tiers"]):
            if volume >= tier_vol:
                tier_index = i + 1
            else:
                break
        return PARAM["fee_edge_add"][tier_index]

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

        # Calculate spread considering base, volatility, and fee tier
        fee_edge = self._get_fee_tier_edge(p)
        vol_spread = PARAM["k_vol"] * self._vol(p)
        # Minimum edge required increases with fee tier
        required_edge = PARAM["tight_spread"] + fee_edge
        # Total spread includes volatility adjustment but respects minimum edge
        spread = int(max(required_edge, PARAM["tight_spread"] + vol_spread))

        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"]))

        # panic strong clear position logic (same as base)
        if abs(pos) >= LIMIT[p] * PARAM["panic_ratio"]:
            buy_px = int(mid - PARAM["panic_add"] - spread) # Apply panic on top of adjusted spread
            sell_px = int(mid + PARAM["panic_add"] + spread)
            size = max(size, abs(pos)//2)

        # Aggressive order taking logic
        b,a = best_bid_ask(depth)
        if PARAM["aggr_take"] and b is not None and a is not None:
            # Only take if the price is better than our fee-adjusted spread
            if a < sell_px and pos < LIMIT[p]:
                qty = min(size, LIMIT[p] - pos, abs(depth.sell_orders[a]))
                if qty > 0: orders.append(Order(p, a, qty))
            if b > buy_px and pos > -LIMIT[p]:
                qty = min(size, LIMIT[p] + pos, depth.buy_orders[b])
                if qty > 0: orders.append(Order(p, b, -qty))

        # Standard limit order placement
        if pos < LIMIT[p]:
            orders.append(Order(p, buy_px, min(size, LIMIT[p] - pos)))
        if pos > -LIMIT[p]:
            orders.append(Order(p, sell_px, -min(size, LIMIT[p] + pos)))

        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        self._update_daily_volume(state) # Update volume before making decisions

        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.mm_product(p, depth, state.position.get(p, 0))

        conversions = 0
        return result, conversions, state.traderData