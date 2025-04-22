from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order
import numpy as np
from collections import deque

class Trader:
    def __init__(self):
        # Configuration
        self.position_limits = {
            "CROISSANTS": 250,
            "DJEMBES": 60,
            "JAMS": 350,
            "KELP": 300,
            "MAGNIFICENT_MACARONS": 70,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 60,
            "RAINFOREST_RESIN": 300,
            "SQUID_INK": 300,
            "VOLCANIC_ROCK": 100,
            "VOLCANIC_ROCK_VOUCHER_10000": 50,
            "VOLCANIC_ROCK_VOUCHER_10250": 50,
            "VOLCANIC_ROCK_VOUCHER_10500": 50,
            "VOLCANIC_ROCK_VOUCHER_9500": 50,
            "VOLCANIC_ROCK_VOUCHER_9750": 50
        }

        # Trading state
        self.position_history = {}
        self.price_history = {}
        self.spread_history = {}
        self.trades_today = {}

        # Market making parameters
        self.min_spread = 2
        self.max_position_util = 0.8
        self.history_size = 50

        for product in self.position_limits.keys():
            self.position_history[product] = deque(maxlen=self.history_size)
            self.price_history[product] = deque(maxlen=self.history_size)
            self.spread_history[product] = deque(maxlen=self.history_size)
            self.trades_today[product] = 0

    def calculate_fair_price(self, product: str, order_depth: OrderDepth) -> float:
        """Calculate fair price based on order book"""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')

        # If we have a valid bid-ask
        if best_bid > 0 and best_ask < float('inf'):
            fair_price = (best_bid + best_ask) / 2
        # If we only have bids
        elif best_bid > 0:
            fair_price = best_bid + 1
        # If we only have asks
        elif best_ask < float('inf'):
            fair_price = best_ask - 1
        # If we have no orders, use last price or default
        else:
            fair_price = self.price_history[product][-1] if len(self.price_history[product]) > 0 else 0

        return fair_price

    def adjust_prices(self, product: str, fair_price: float, position: int) -> tuple[float, float]:
        """Adjust prices based on position and trading volume"""
        if fair_price == 0:  # Don't trade if we can't determine a fair price
            return 0, 0

        position_util = abs(position) / self.position_limits[product]
        base_spread = self.min_spread * (1 + position_util)

        # Increase spread if high trading volume
        if self.trades_today[product] > 100:
            base_spread *= 1.5
        if self.trades_today[product] > 200:
            base_spread *= 2

        # Skew prices based on position
        skew = position / self.position_limits[product] * base_spread

        bid_price = fair_price - base_spread/2 - skew
        ask_price = fair_price + base_spread/2 - skew

        return int(bid_price), int(ask_price)

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """Main trading logic"""
        result = {}
        conversions = 0  # No conversions in this strategy
        trader_data = ""  # No trader data to persist

        for product in state.order_depths.keys():
            if product not in self.position_limits:
                continue

            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)

            # Update history
            self.position_history[product].append(position)

            # Calculate fair price
            fair_price = self.calculate_fair_price(product, order_depth)
            if fair_price > 0:
                self.price_history[product].append(fair_price)

            # Calculate bid/ask prices
            bid_price, ask_price = self.adjust_prices(product, fair_price, position)

            # Only proceed if we have valid prices
            if bid_price <= 0 or ask_price <= 0:
                continue

            # Calculate quantities
            available_buy = self.position_limits[product] - position
            available_sell = self.position_limits[product] + position

            # Place orders
            orders: List[Order] = []

            if available_buy > 0 and bid_price > 0:
                orders.append(Order(product, bid_price, min(10, available_buy)))

            if available_sell > 0 and ask_price > 0:
                orders.append(Order(product, ask_price, -min(10, available_sell)))

            if orders:
                result[product] = orders

            # Update trade count
            self.trades_today[product] += len(orders)

        return result, conversions, trader_data
