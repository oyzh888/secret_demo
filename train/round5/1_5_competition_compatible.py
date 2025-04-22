from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics

# Position limits for each product
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

# Strategy parameters
PARAM = {
    "tight_spread": 1,        # Base market making spread (±1 tick)
    "k_vol": 1.2,             # Volatility factor for spread widening
    "panic_ratio": 0.8,       # Panic threshold: |pos|≥limit×0.8 triggers aggressive pricing
    "panic_add": 4,           # Additional ticks from mid price in panic mode
    "mm_size_frac": 0.25,     # Order size as fraction of position limit
    "aggr_take": True,        # Whether to take opponent's orders
    "arb_threshold": 1.0,     # Threshold for arbitrage opportunities
    "arb_size_limit": 0.1,    # Limit size of arbitrage trades as fraction of position limit
    "trend_window": 20,       # Window for trend detection
    "trend_threshold": 0.6    # Threshold for trend detection
}

def best_bid_ask(depth: OrderDepth):
    """Get best bid and ask prices from order depth"""
    best_bid = max(depth.buy_orders.keys()) if depth.buy_orders else None
    best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None
    return best_bid, best_ask

class Trader:
    def __init__(self):
        # Initialize state variables
        self.prices = {}
        self.last_mid_price = {}
        self.trends = {}
        self.cp_trades = {}
        self.cp_score = {}
        self.profitable_cps = set()
        self.unprofitable_cps = set()

    def _vol(self, product, prices):
        """Calculate volatility for a product"""
        if product not in prices or len(prices[product]) < 15:
            return 1
        return statistics.stdev(prices[product][-15:]) or 1

    def _mid(self, depth):
        """Calculate mid price from order book"""
        bid, ask = best_bid_ask(depth)
        return (bid + ask) / 2 if bid is not None and ask is not None else None

    def detect_trend(self, product, prices):
        """Detect market trend based on recent price movements"""
        if product not in prices or len(prices[product]) < PARAM["trend_window"]:
            return 0  # Not enough data

        recent_prices = prices[product][-PARAM["trend_window"]:]
        up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])

        up_ratio = up_moves / (len(recent_prices) - 1)
        down_ratio = down_moves / (len(recent_prices) - 1)

        if up_ratio > PARAM["trend_threshold"]:
            return 1  # Uptrend
        elif down_ratio > PARAM["trend_threshold"]:
            return -1  # Downtrend
        return 0  # Neutral

    def check_voucher_arbitrage(self, state):
        """Check for arbitrage opportunities between VOLCANIC_ROCK and its vouchers"""
        # Disable arbitrage strategy completely for now
        # This is to avoid issues in the competition environment
        return {}

    def mm_product(self, product, depth, pos, own_trades=None, market_trades=None, prices=None, trends=None):
        """Market making strategy for a single product"""
        orders = []
        mid = self._mid(depth)
        if mid is None:
            return orders

        # Record mid price
        if product not in prices:
            prices[product] = []
        prices[product].append(mid)

        # Analyze counterparty information
        active_cps = set()
        if own_trades:
            active_cps = self.analyze_counterparty(product, own_trades, mid, prices)

        # Detect market trend
        trend = self.detect_trend(product, prices)
        trends[product] = trend

        # Calculate base spread
        spread = int(PARAM["tight_spread"] + PARAM["k_vol"] * self._vol(product, prices))
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)

        # Adjust prices based on trend
        if trend == 1:  # Uptrend
            buy_px += 1  # More aggressive buying
            sell_px += 1  # Less aggressive selling
        elif trend == -1:  # Downtrend
            buy_px -= 1  # Less aggressive buying
            sell_px -= 1  # More aggressive selling

        # Adjust prices based on counterparty analysis
        profitable_count = sum(1 for cp in active_cps if cp in getattr(self, 'profitable_cps', set()))
        unprofitable_count = sum(1 for cp in active_cps if cp in getattr(self, 'unprofitable_cps', set()))

        # If more profitable than unprofitable counterparties are active
        if profitable_count > unprofitable_count:
            # Be more aggressive with pricing
            buy_px += 1  # Willing to pay more when buying
            sell_px -= 1  # Willing to accept less when selling
        # If more unprofitable than profitable counterparties are active
        elif unprofitable_count > profitable_count and unprofitable_count > 0:
            # Be more conservative with pricing
            buy_px -= 1  # Less willing to pay when buying
            sell_px += 1  # Demand more when selling

        size = max(1, int(LIMIT[product] * PARAM["mm_size_frac"]))

        # Panic mode for large positions
        if abs(pos) >= LIMIT[product] * PARAM["panic_ratio"]:
            buy_px = int(mid - PARAM["panic_add"] - spread)
            sell_px = int(mid + PARAM["panic_add"] + spread)
            size = max(size, abs(pos)//2)

        # Take orders if they're favorable
        bid, ask = best_bid_ask(depth)
        if PARAM["aggr_take"] and bid is not None and ask is not None:
            if ask < mid - spread and pos < LIMIT[product]:
                qty = min(size, LIMIT[product] - pos, abs(depth.sell_orders[ask]))
                if qty:
                    orders.append(Order(product, ask, qty))
            if bid > mid + spread and pos > -LIMIT[product]:
                qty = min(size, LIMIT[product] + pos, depth.buy_orders[bid])
                if qty:
                    orders.append(Order(product, bid, -qty))

        # Regular market making orders
        if pos < LIMIT[product]:
            orders.append(Order(product, buy_px, min(size, LIMIT[product] - pos)))
        if pos > -LIMIT[product]:
            orders.append(Order(product, sell_px, -min(size, LIMIT[product] + pos)))

        return orders

    def analyze_counterparty(self, product, own_trades, mid_price, prices):
        """Analyze counterparty information from trades"""
        # In Round 5, the OwnTrade object now includes a counter_party property
        # We can use this to track which counterparties tend to trade in which direction

        # Initialize counterparty tracking if needed
        if not hasattr(self, 'cp_trades'):
            self.cp_trades = {}
        if not hasattr(self, 'cp_price_impact'):
            self.cp_price_impact = {}
        if not hasattr(self, 'cp_score'):
            self.cp_score = {}
        if not hasattr(self, 'profitable_cps'):
            self.profitable_cps = set()
        if not hasattr(self, 'unprofitable_cps'):
            self.unprofitable_cps = set()

        # Track active counterparties in this iteration
        active_cps = set()

        for trade in own_trades:
            # Get counterparty (only available in actual competition)
            cp = getattr(trade, 'counter_party', None)
            if not cp:
                continue

            # Record counterparty activity
            active_cps.add(cp)

            # Initialize tracking for this counterparty if needed
            if cp not in self.cp_trades:
                self.cp_trades[cp] = []
            if cp not in self.cp_price_impact:
                self.cp_price_impact[cp] = []

            # Record trade
            self.cp_trades[cp].append((product, trade.price, trade.quantity))

            # Keep only recent trades
            if len(self.cp_trades[cp]) > 50:  # Remember last 50 trades
                self.cp_trades[cp].pop(0)

            # Record price impact if we have previous price data
            if product in prices and len(prices[product]) > 1:
                prev_price = prices[product][-2]  # Price before this trade
                self.cp_price_impact[cp].append((prev_price, mid_price))

                # Keep only recent price impacts
                if len(self.cp_price_impact[cp]) > 50:
                    self.cp_price_impact[cp].pop(0)

        # Update counterparty scores
        for cp in active_cps:
            if cp not in self.cp_price_impact or not self.cp_price_impact[cp]:
                continue

            # Calculate how often prices move favorably after trading with this counterparty
            favorable_moves = 0
            total_moves = len(self.cp_price_impact[cp])

            for before, after in self.cp_price_impact[cp]:
                # Get the most recent trade with this counterparty
                if not self.cp_trades[cp]:
                    continue

                _, _, qty = self.cp_trades[cp][-1]

                # For buys (positive quantity), favorable if price goes up
                # For sells (negative quantity), favorable if price goes down
                if qty > 0 and after > before:  # We bought, price went up (good for us)
                    favorable_moves += 1
                elif qty < 0 and after < before:  # We sold, price went down (good for us)
                    favorable_moves += 1
                elif qty > 0 and after < before:  # We bought, price went down (bad for us)
                    favorable_moves -= 1
                elif qty < 0 and after > before:  # We sold, price went up (bad for us)
                    favorable_moves -= 1

            # Update score with exponential decay
            if total_moves > 0:
                new_score = favorable_moves / total_moves
                old_score = self.cp_score.get(cp, 0)
                self.cp_score[cp] = 0.7 * old_score + 0.3 * new_score

            # Classify counterparty
            if self.cp_score.get(cp, 0) > 0.6:  # Threshold for considering profitable
                self.profitable_cps.add(cp)
                if cp in self.unprofitable_cps:
                    self.unprofitable_cps.remove(cp)
            elif self.cp_score.get(cp, 0) < -0.6:  # Threshold for considering unprofitable
                self.unprofitable_cps.add(cp)
                if cp in self.profitable_cps:
                    self.profitable_cps.remove(cp)

        return active_cps

    def run(self, state: TradingState):
        """Main trading logic"""
        # Initialize or update state from trader data if available
        if state.traderData:
            try:
                import json
                data = json.loads(state.traderData)
                self.prices = data.get("prices", {})
                self.trends = data.get("trends", {})
            except:
                # If there's an error parsing the trader data, initialize empty
                self.prices = {}
                self.trends = {}
        else:
            # Initialize empty state
            self.prices = {}
            self.trends = {}

        # Convert prices from string keys to proper types if needed
        prices = {}
        for product, price_list in self.prices.items():
            prices[product] = price_list

        trends = {}
        for product, trend in self.trends.items():
            trends[product] = trend

        result = {}

        # First check for arbitrage opportunities
        arb_orders = self.check_voucher_arbitrage(state)
        result.update(arb_orders)

        # Then apply market making strategy to all products
        for product, depth in state.order_depths.items():
            if product not in LIMIT:
                continue

            # Skip if we already have orders for this product from arbitrage
            if product in result:
                continue

            own_trades = state.own_trades.get(product, [])
            market_trades = state.market_trades.get(product, [])

            result[product] = self.mm_product(
                product,
                depth,
                state.position.get(product, 0),
                own_trades,
                market_trades,
                prices,
                trends
            )

        # Save state for next iteration
        try:
            import json
            traderData = json.dumps({
                "prices": prices,
                "trends": trends
            })
        except:
            traderData = ""

        # No conversions in this strategy
        conversions = 0

        return result, conversions, traderData
