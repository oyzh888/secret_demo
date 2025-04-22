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

# Parameters (base + counterparty adjustments)
PARAM = {
    "tight_spread": 1,        # Base spread
    "k_vol": 1.2,             # Volatility multiplier
    "panic_ratio": 0.8,
    "panic_add": 4,
    "mm_size_frac": 0.25,
    "aggr_take": True,
    "cp_aggr_add": 1,        # Extra spread for aggressive CPs (Caesar buy)
    "cp_passive_sub": 0,     # Spread reduction for passive CPs (Charlie) - currently none
    "cp_trend_follow_add": 1 # Extra spread when trading against trend follower (Camilla buy)
}

# Key Counterparties identified in analysis
KEY_COUNTERPARTIES = {"Caesar", "Camilla", "Charlie"}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.last_counterparty = defaultdict(lambda: None) # Track last counterparty per product

    def _vol(self, p:str) -> float:
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) or 1

    def _mid(self, depth):
        b,a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None

    def update_last_counterparty(self, own_trades: Dict[str, List]):
        for symbol, trades in own_trades.items():
            if trades:
                # Use the counterparty from the most recent trade for this symbol
                # In the backtester, trades might not have counter_party attribute
                trade = trades[-1]
                if hasattr(trade, 'counter_party'):
                    self.last_counterparty[symbol] = trade.counter_party
                # Try to get buyer/seller in backtester environment
                elif hasattr(trade, 'buyer') and hasattr(trade, 'seller'):
                    # If we're the buyer, the counterparty is the seller
                    if trade.buyer == 'SUBMISSION':
                        self.last_counterparty[symbol] = trade.seller
                    # If we're the seller, the counterparty is the buyer
                    elif trade.seller == 'SUBMISSION':
                        self.last_counterparty[symbol] = trade.buyer

    def mm_product(self, p:str, depth:OrderDepth, pos:int, last_cp: Optional[str]) -> List[Order]:
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders

        self.prices[p].append(mid)
        base_spread = int(PARAM["tight_spread"] + PARAM["k_vol"] * self._vol(p))
        buy_spread = base_spread
        sell_spread = base_spread

        # Adjust spread based on last counterparty
        if last_cp == "Caesar":
            # Caesar buys aggressively, sell higher
            sell_spread += PARAM["cp_aggr_add"]
            # Caesar sells passively, can buy tighter? (or keep base)
            # buy_spread -= PARAM["cp_passive_sub"] # Example, currently 0
        elif last_cp == "Camilla":
            # Camilla buys biased, sell higher (trend following)
            sell_spread += PARAM["cp_trend_follow_add"]
        elif last_cp == "Charlie":
            # Charlie is neutral, maybe slightly tighter spread?
            # buy_spread -= PARAM["cp_passive_sub"] # Example, currently 0
            # sell_spread -= PARAM["cp_passive_sub"] # Example, currently 0
            pass # Keep base spread for now

        buy_px = int(mid - buy_spread)
        sell_px = int(mid + sell_spread)
        size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"]))

        # panic strong clear position logic (same as base)
        if abs(pos) >= LIMIT[p] * PARAM["panic_ratio"]:
            buy_px = int(mid - PARAM["panic_add"] - buy_spread)
            sell_px = int(mid + PARAM["panic_add"] + sell_spread)
            size = max(size, abs(pos)//2)

        # Aggressive order taking logic (same as base, uses adjusted prices implicitly)
        b,a = best_bid_ask(depth)
        if PARAM["aggr_take"] and b is not None and a is not None:
            if a < sell_px and pos < LIMIT[p]: # Check against our adjusted sell price
                qty = min(size, LIMIT[p] - pos, abs(depth.sell_orders[a]))
                if qty > 0: orders.append(Order(p, a, qty))
            if b > buy_px and pos > -LIMIT[p]: # Check against our adjusted buy price
                qty = min(size, LIMIT[p] + pos, depth.buy_orders[b])
                if qty > 0: orders.append(Order(p, b, -qty))

        # Standard limit order placement (uses adjusted prices)
        if pos < LIMIT[p]:
            orders.append(Order(p, buy_px, min(size, LIMIT[p] - pos)))
        if pos > -LIMIT[p]:
            orders.append(Order(p, sell_px, -min(size, LIMIT[p] + pos)))

        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        self.update_last_counterparty(state.own_trades) # Update counterparty info

        for p, depth in state.order_depths.items():
            if p in LIMIT:
                last_cp = self.last_counterparty[p]
                result[p] = self.mm_product(p, depth, state.position.get(p, 0), last_cp)

        conversions = 0
        return result, conversions, state.traderData