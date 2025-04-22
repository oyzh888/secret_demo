from typing import Dict, List, Tuple, Set, Optional
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict

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

PARAM = {
    "tight_spread": 1,        # Base market making spread (±1 tick)
    "k_vol": 1.2,             # Volatility factor for spread widening
    "panic_ratio": 0.8,       # Panic threshold: |pos|≥limit×0.8 triggers aggressive pricing
    "panic_add": 4,           # Additional ticks from mid price in panic mode
    "mm_size_frac": 0.25,     # Order size as fraction of position limit
    "aggr_take": True,        # Whether to take opponent's orders
    "cp_memory": 50,          # Number of trades to remember per counterparty
    "cp_alpha": 0.7,          # Counterparty profitability score decay factor
    "cp_threshold": 0.6,      # Threshold for considering a counterparty profitable
    "arb_threshold": 1.0,     # Threshold for arbitrage opportunities (increased to be more conservative)
    "arb_size_limit": 0.1,    # Limit size of arbitrage trades as fraction of position limit
    "trend_window": 20,       # Window for trend detection
    "trend_threshold": 0.6    # Threshold for trend detection
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class CounterpartyAnalyzer:
    """Analyzes counterparty behavior to identify profitable trading opportunities"""

    def __init__(self):
        # Track trades with each counterparty
        self.cp_trades = defaultdict(list)  # {counterparty_id: [(price, quantity, timestamp), ...]}
        # Track profitability score for each counterparty
        self.cp_score = defaultdict(float)  # {counterparty_id: score}
        # Track price movements after trading with each counterparty
        self.cp_price_impact = defaultdict(list)  # {counterparty_id: [(price_before, price_after), ...]}
        # Set of counterparties that tend to be profitable to trade against
        self.profitable_cps = set()
        # Set of counterparties that tend to be unprofitable to trade against
        self.unprofitable_cps = set()

    def record_trade(self, product: str, trade_price: int, trade_qty: int,
                    counterparty: str, timestamp: int, mid_price: float):
        """Record a trade with a counterparty"""
        if not counterparty:
            return

        # Store trade info
        self.cp_trades[counterparty].append((product, trade_price, trade_qty, timestamp))

        # Keep only the most recent trades
        if len(self.cp_trades[counterparty]) > PARAM["cp_memory"]:
            self.cp_trades[counterparty].pop(0)

        # Update price impact data
        if product in self.last_mid_price:
            price_before = self.last_mid_price[product]
            self.cp_price_impact[counterparty].append((price_before, mid_price))

            # Keep only recent price impacts
            if len(self.cp_price_impact[counterparty]) > PARAM["cp_memory"]:
                self.cp_price_impact[counterparty].pop(0)

    def update_counterparty_scores(self):
        """Update profitability scores for all counterparties"""
        for cp, price_impacts in self.cp_price_impact.items():
            if not price_impacts:
                continue

            # Calculate how often prices move favorably after trading with this counterparty
            favorable_moves = 0
            total_moves = len(price_impacts)

            for before, after in price_impacts:
                # For buys (negative quantity), favorable if price goes up
                # For sells (positive quantity), favorable if price goes down
                last_trade = self.cp_trades[cp][-1] if self.cp_trades[cp] else None
                if not last_trade:
                    continue

                _, _, qty, _ = last_trade  # Unpack only the quantity
                if qty < 0 and after > before:  # We sold, price went up (bad for us)
                    favorable_moves -= 1
                elif qty > 0 and after < before:  # We bought, price went down (bad for us)
                    favorable_moves -= 1
                elif qty < 0 and after < before:  # We sold, price went down (good for us)
                    favorable_moves += 1
                elif qty > 0 and after > before:  # We bought, price went up (good for us)
                    favorable_moves += 1

            # Update score with exponential decay
            if total_moves > 0:
                new_score = favorable_moves / total_moves
                old_score = self.cp_score[cp]
                self.cp_score[cp] = PARAM["cp_alpha"] * old_score + (1 - PARAM["cp_alpha"]) * new_score

            # Classify counterparty
            if self.cp_score[cp] > PARAM["cp_threshold"]:
                self.profitable_cps.add(cp)
                if cp in self.unprofitable_cps:
                    self.unprofitable_cps.remove(cp)
            elif self.cp_score[cp] < -PARAM["cp_threshold"]:
                self.unprofitable_cps.add(cp)
                if cp in self.profitable_cps:
                    self.profitable_cps.remove(cp)

    def should_trade_with(self, counterparty: str) -> bool:
        """Determine if we should trade with a specific counterparty"""
        if counterparty in self.profitable_cps:
            return True
        if counterparty in self.unprofitable_cps:
            return False
        return True  # Default to trading if we don't have enough data

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)   # For volatility estimation
        self.last_mid_price = {}          # Last observed mid price per product
        self.cp_analyzer = CounterpartyAnalyzer()
        self.cp_analyzer.last_mid_price = self.last_mid_price  # Share reference
        self.trends = defaultdict(int)    # Track market trends (1=up, -1=down, 0=neutral)
        self.voucher_arb = {}             # Track arbitrage opportunities for vouchers

    def _vol(self, p: str) -> float:
        """Calculate volatility for a product"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) or 1

    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """Calculate mid price from order book"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None

    def analyze_recent_trades(self, product: str, trades, mid_price: float):
        """Analyze recent trades to identify counterparty patterns"""
        if not trades:
            return

        for trade in trades:
            # Check if trade has counter_party attribute (only in actual competition, not in backtester)
            counterparty = getattr(trade, 'counter_party', None)
            if not counterparty:
                # Try to get buyer/seller in backtester environment
                counterparty = getattr(trade, 'buyer', None) or getattr(trade, 'seller', None)

            self.cp_analyzer.record_trade(
                product,
                trade.price,
                trade.quantity,
                counterparty,
                getattr(trade, 'timestamp', 0),
                mid_price
            )

        # Update counterparty scores
        self.cp_analyzer.update_counterparty_scores()

    def adjust_price_for_counterparty(self, base_price: int, is_buy: bool, counterparties: Set[str]) -> int:
        """Adjust price based on counterparty analysis"""
        # Default to base price if no counterparty info
        if not counterparties:
            return base_price

        # Count profitable vs unprofitable counterparties
        profitable_count = sum(1 for cp in counterparties if cp in self.cp_analyzer.profitable_cps)
        unprofitable_count = sum(1 for cp in counterparties if cp in self.cp_analyzer.unprofitable_cps)

        # If more profitable than unprofitable counterparties are active
        if profitable_count > unprofitable_count:
            # Be more aggressive with pricing
            return base_price + (1 if is_buy else -1)
        # If more unprofitable than profitable counterparties are active
        elif unprofitable_count > profitable_count:
            # Be more conservative with pricing
            return base_price + (-1 if is_buy else 1)

        return base_price

    def detect_trend(self, product: str) -> int:
        """Detect market trend based on recent price movements"""
        prices = self.prices[product]
        if len(prices) < PARAM["trend_window"]:
            return 0  # Not enough data

        recent_prices = prices[-PARAM["trend_window"]:]
        up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])

        up_ratio = up_moves / (len(recent_prices) - 1)
        down_ratio = down_moves / (len(recent_prices) - 1)

        if up_ratio > PARAM["trend_threshold"]:
            return 1  # Uptrend
        elif down_ratio > PARAM["trend_threshold"]:
            return -1  # Downtrend
        return 0  # Neutral

    def check_voucher_arbitrage(self, state: TradingState) -> Dict[str, List[Order]]:
        """Check for arbitrage opportunities between VOLCANIC_ROCK and its vouchers"""
        result = {}

        # Check if VOLCANIC_ROCK is available
        if "VOLCANIC_ROCK" not in state.order_depths:
            return result

        rock_depth = state.order_depths["VOLCANIC_ROCK"]
        rock_mid = self._mid(rock_depth)
        if rock_mid is None:
            return result

        # Check each voucher
        voucher_products = [p for p in state.order_depths.keys() if p.startswith("VOLCANIC_ROCK_VOUCHER_")]
        for voucher in voucher_products:
            if voucher not in state.order_depths:
                continue

            # Extract voucher value from name (e.g., VOLCANIC_ROCK_VOUCHER_10000 -> 10000)
            try:
                voucher_value = int(voucher.split("_")[-1])
            except ValueError:
                continue

            voucher_depth = state.order_depths[voucher]
            voucher_mid = self._mid(voucher_depth)
            if voucher_mid is None:
                continue

            # Calculate theoretical fair value ratio
            fair_ratio = voucher_value / 10000  # Normalize to base of 10000

            # Calculate actual ratio
            actual_ratio = voucher_mid / rock_mid

            # Check for arbitrage opportunity
            arb_diff = actual_ratio - fair_ratio

            if abs(arb_diff) > PARAM["arb_threshold"] / 100:  # Convert threshold to percentage
                # Store arbitrage opportunity
                self.voucher_arb[voucher] = arb_diff

                # If voucher is underpriced relative to rock
                if arb_diff < 0:
                    # Buy voucher, sell rock
                    if voucher_depth.sell_orders and rock_depth.buy_orders:
                        best_ask_voucher = min(voucher_depth.sell_orders.keys())
                        best_bid_rock = max(rock_depth.buy_orders.keys())

                        # Calculate quantities
                        rock_pos = state.position.get("VOLCANIC_ROCK", 0)
                        voucher_pos = state.position.get(voucher, 0)

                        # Limit arbitrage size to reduce risk
                        arb_size_limit = int(LIMIT[voucher] * PARAM["arb_size_limit"])

                        max_buy_voucher = min(
                            abs(voucher_depth.sell_orders[best_ask_voucher]),
                            LIMIT[voucher] - voucher_pos,
                            arb_size_limit
                        )

                        if max_buy_voucher > 0:
                            # Calculate profit potential
                            voucher_cost = best_ask_voucher * max_buy_voucher
                            rock_qty = int(max_buy_voucher * (voucher_value / 10000))
                            rock_revenue = best_bid_rock * rock_qty
                            profit_potential = rock_revenue - voucher_cost

                            # Only proceed if profitable
                            if profit_potential > 0:
                                # Buy voucher
                                if voucher not in result:
                                    result[voucher] = []
                                result[voucher].append(Order(voucher, best_ask_voucher, max_buy_voucher))

                                # Sell equivalent amount of rock
                                max_sell_rock = min(
                                    rock_depth.buy_orders[best_bid_rock],
                                    rock_pos + LIMIT["VOLCANIC_ROCK"]
                                )

                                if rock_qty > 0 and rock_qty <= max_sell_rock:
                                    if "VOLCANIC_ROCK" not in result:
                                        result["VOLCANIC_ROCK"] = []
                                    result["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", best_bid_rock, -rock_qty))

                # If voucher is overpriced relative to rock
                else:
                    # Sell voucher, buy rock
                    if voucher_depth.buy_orders and rock_depth.sell_orders:
                        best_bid_voucher = max(voucher_depth.buy_orders.keys())
                        best_ask_rock = min(rock_depth.sell_orders.keys())

                        # Calculate quantities
                        rock_pos = state.position.get("VOLCANIC_ROCK", 0)
                        voucher_pos = state.position.get(voucher, 0)

                        # Limit arbitrage size to reduce risk
                        arb_size_limit = int(LIMIT[voucher] * PARAM["arb_size_limit"])

                        max_sell_voucher = min(
                            voucher_depth.buy_orders[best_bid_voucher],
                            voucher_pos + LIMIT[voucher],
                            arb_size_limit
                        )

                        if max_sell_voucher > 0:
                            # Calculate profit potential
                            voucher_revenue = best_bid_voucher * max_sell_voucher
                            rock_qty = int(max_sell_voucher * (voucher_value / 10000))
                            rock_cost = best_ask_rock * rock_qty
                            profit_potential = voucher_revenue - rock_cost

                            # Only proceed if profitable
                            if profit_potential > 0:
                                # Sell voucher
                                if voucher not in result:
                                    result[voucher] = []
                                result[voucher].append(Order(voucher, best_bid_voucher, -max_sell_voucher))

                                # Buy equivalent amount of rock
                                max_buy_rock = min(
                                    abs(rock_depth.sell_orders[best_ask_rock]),
                                    LIMIT["VOLCANIC_ROCK"] - rock_pos
                                )

                                if rock_qty > 0 and rock_qty <= max_buy_rock:
                                    if "VOLCANIC_ROCK" not in result:
                                        result["VOLCANIC_ROCK"] = []
                                    result["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", best_ask_rock, rock_qty))

        return result

    def mm_product(self, p: str, depth: OrderDepth, pos: int,
                  own_trades: List = None, market_trades: List = None) -> List[Order]:
        """Market making strategy for a single product"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders

        # Record mid price
        self.last_mid_price[p] = mid
        self.prices[p].append(mid)

        # Analyze recent trades
        active_counterparties = set()
        if own_trades:
            self.analyze_recent_trades(p, own_trades, mid)
            # Get counterparties safely
            for trade in own_trades:
                cp = getattr(trade, 'counter_party', None)
                if cp:
                    active_counterparties.add(cp)
                else:
                    # Try to get buyer/seller in backtester environment
                    buyer = getattr(trade, 'buyer', None)
                    seller = getattr(trade, 'seller', None)
                    if buyer:
                        active_counterparties.add(buyer)
                    if seller:
                        active_counterparties.add(seller)

        if market_trades:
            for trade in market_trades:
                buyer = getattr(trade, 'buyer', None)
                seller = getattr(trade, 'seller', None)
                if buyer:
                    active_counterparties.add(buyer)
                if seller:
                    active_counterparties.add(seller)

        # Detect market trend
        trend = self.detect_trend(p)
        self.trends[p] = trend

        # Calculate base spread
        spread = int(PARAM["tight_spread"] + PARAM["k_vol"] * self._vol(p))
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)

        # Adjust prices based on counterparty analysis
        buy_px = self.adjust_price_for_counterparty(buy_px, True, active_counterparties)
        sell_px = self.adjust_price_for_counterparty(sell_px, False, active_counterparties)

        # Adjust prices based on trend
        if trend == 1:  # Uptrend
            buy_px += 1  # More aggressive buying
            sell_px += 1  # Less aggressive selling
        elif trend == -1:  # Downtrend
            buy_px -= 1  # Less aggressive buying
            sell_px -= 1  # More aggressive selling

        size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"]))

        # Panic mode for large positions
        if abs(pos) >= LIMIT[p] * PARAM["panic_ratio"]:
            buy_px = int(mid - PARAM["panic_add"] - spread)
            sell_px = int(mid + PARAM["panic_add"] + spread)
            size = max(size, abs(pos)//2)

        # Take orders if they're favorable
        b, a = best_bid_ask(depth)
        if PARAM["aggr_take"] and b is not None and a is not None:
            if a < mid - spread and pos < LIMIT[p]:
                qty = min(size, LIMIT[p] - pos, abs(depth.sell_orders[a]))
                if qty: orders.append(Order(p, a, qty))
            if b > mid + spread and pos > -LIMIT[p]:
                qty = min(size, LIMIT[p] + pos, depth.buy_orders[b])
                if qty: orders.append(Order(p, b, -qty))

        # Regular market making orders
        if pos < LIMIT[p]:
            orders.append(Order(p, buy_px, min(size, LIMIT[p] - pos)))
        if pos > -LIMIT[p]:
            orders.append(Order(p, sell_px, -min(size, LIMIT[p] + pos)))

        return orders

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """Main trading logic"""
        result: Dict[str, List[Order]] = {}

        # In the competition environment, we can't use environment variables
        # Instead, we'll use a simpler approach to disable arbitrage for problematic data

        # Disable arbitrage for now to avoid issues in the competition environment
        # arb_orders = self.check_voucher_arbitrage(state)
        # result.update(arb_orders)

        # Then apply market making strategy to all products
        for p, depth in state.order_depths.items():
            if p not in LIMIT:
                continue

            # Skip if we already have orders for this product from arbitrage
            if p in result:
                continue

            own_trades = state.own_trades.get(p, [])
            market_trades = state.market_trades.get(p, [])

            result[p] = self.mm_product(
                p,
                depth,
                state.position.get(p, 0),
                own_trades,
                market_trades
            )

        return result, 0, state.traderData
