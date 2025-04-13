import json
from typing import Any, Dict, List
import statistics
import jsonpickle
from collections import deque

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

#############################################
# HYPERPARAMETERS - Adjust these as needed
#############################################

# Global parameters
POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50
}

# History parameters
MAX_HISTORY_LENGTH = 30  # Keep last 30 points for calculations
MM_HISTORY_SIZE = 10     # History size for market making liquidation logic

# RAINFOREST_RESIN parameters (Market Making)
RESIN_DEFAULT_PRICE = 10000
RESIN_POSITION_SKEW_THRESHOLD = 0.5  # Position ratio at which to skew prices
RESIN_PRICE_ADJUSTMENT = 1  # Price adjustment when position is skewed

# KELP parameters (Market Making)
KELP_DEFAULT_PRICE = 10000  # Fallback default price
KELP_POSITION_SKEW_THRESHOLD = 0.5  # Position ratio at which to skew prices
KELP_PRICE_ADJUSTMENT = 1  # Price adjustment when position is skewed

# SQUID_INK parameters (Mean Reversion)
SQUID_PARAMS = {
    "window": 15,           # MA window for recent average
    "order_size": 10,       # Smaller order size due to risk hint
    "z_entry_threshold": 2.5,  # Z-score to consider entry
    "z_exit_threshold": 2.0    # Z-score to confirm entry after crossing back
}

# Market Making parameters
MM_SOFT_LIQUIDATION_THRESHOLD = 0.5  # Trigger soft liquidation when 50% of history exceeds limit
MM_HARD_LIQUIDATION_THRESHOLD = 1.0  # Trigger hard liquidation when 100% of history exceeds limit
MM_PRICE_ADJUSTMENT = 2  # Price adjustment for soft liquidation orders

#############################################

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        if len(self.logs) < 3000:  # Limit log size
            self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        try:
            base_length = len(
                self.to_json(
                    [
                        self.compress_state(state, ""),
                        self.compress_orders(orders),
                        conversions,
                        "",
                        "",
                    ]
                )
            )

            # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
            max_item_length = (self.max_log_length - base_length) // 3

            print(
                self.to_json(
                    [
                        self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                        self.compress_orders(orders),
                        conversions,
                        self.truncate(trader_data, max_item_length),
                        self.truncate(self.logs, max_item_length),
                    ]
                )
            )

            self.logs = ""
        except Exception as e:
            print(f"Error in logger flush: {str(e)}")

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        self.products = ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]
        self.position_limits = POSITION_LIMITS
        self.max_history_length = MAX_HISTORY_LENGTH
        self.mm_history_size = MM_HISTORY_SIZE
        self.squid_params = SQUID_PARAMS

    def _calculate_moving_average(self, prices: List[float], window: int) -> float | None:
        if len(prices) < window:
            return None
        return statistics.mean(prices[-window:])

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        try:
            result = {}
            conversions = 0

            # Restore price history & saved state from traderData
            saved_state = jsonpickle.decode(state.traderData) if state.traderData else {}
            # Ensure basic structure exists for all products
            for p in self.products:
                if p not in saved_state:
                    saved_state[p] = {"prices": [], "prev_z": 0.0, "mm_history": []}
                else:
                    if "prices" not in saved_state[p]:
                        saved_state[p]["prices"] = []
                    # Ensure prev_z exists for squid, default to 0 if missing
                    if p == "SQUID_INK" and "prev_z" not in saved_state[p]:
                        saved_state[p]["prev_z"] = 0.0
                    # Ensure mm_history exists for resin/kelp, default to empty list
                    if p in ["RAINFOREST_RESIN", "KELP"] and "mm_history" not in saved_state[p]:
                        saved_state[p]["mm_history"] = []

            logger.print(f"Timestamp: {state.timestamp}")
            logger.print(f"Current positions: {state.position}")

            for product in self.products:
                try:
                    if product not in state.order_depths:
                        result[product] = []
                        continue

                    order_depth = state.order_depths[product]
                    position = state.position.get(product, 0)
                    product_state = saved_state[product] # Use the loaded state dict for this product
                    orders: List[Order] = []

                    # Calculate best bid/ask and mid_price (needed for history, but not all strategies use it directly)
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

                    if best_bid is not None and best_ask is not None:
                        mid_price = (best_bid + best_ask) / 2
                        product_state["prices"].append(mid_price)
                        # Trim price history
                        if len(product_state["prices"]) > self.max_history_length:
                            product_state["prices"] = product_state["prices"][-self.max_history_length:]
                        logger.print(f"Product: {product}, Mid Price: {mid_price:.2f}, Position: {position}")
                    else:
                        # Cannot calculate mid_price, might affect MA based strategies if history is short
                        logger.print(f"Product: {product}, Position: {position}, Error: Missing best bid or ask.")

                    # Apply strategy based on product
                    if product == "RAINFOREST_RESIN":
                        orders = self._rainforest_resin_strategy(product, order_depth, position, product_state)
                    elif product == "KELP":
                        orders = self._kelp_strategy(product, order_depth, position, product_state)
                    elif product == "SQUID_INK":
                        # Pass the specific product's state dict which contains 'prev_z' and prices
                        # Squid strategy returns tuple: (orders, current_z)
                        orders, new_z = self._squid_ink_strategy(product, order_depth, position, product_state)
                        # Update prev_z in the state dict after the strategy runs
                        product_state["prev_z"] = new_z

                    result[product] = orders
                    if orders:
                        logger.print(f"{product} Orders: {orders}")
                except Exception as e:
                    logger.print(f"Error processing {product}: {str(e)}")
                    result[product] = []

            # Save price history & state back to traderData
            trader_data = jsonpickle.encode(saved_state)

            logger.flush(state, result, conversions, trader_data)
            return result, conversions, trader_data
        except Exception as e:
            logger.print(f"Critical error in trader: {str(e)}")
            return {}, 0, "{}"

    def _market_making_logic(self, product: str, order_depth: OrderDepth, position: int, product_state: Dict, default_price: int) -> list[Order]:
        """Core market making logic strictly adapted from 12good.py's act method."""
        # Reset internal orders list for this call
        self.orders = []
        limit = self.position_limits[product]

        # Load and manage history deque
        history_list = product_state.get("mm_history", [])
        mm_history_deque = deque(history_list, maxlen=self.mm_history_size)

        # Get sorted books
        buy_orders_book = sorted(order_depth.buy_orders.items(), reverse = True)
        sell_orders_book = sorted(order_depth.sell_orders.items())

        # Initial capacity - these will be updated sequentially
        to_buy = limit - position
        to_sell = limit + position

        # Update history
        hit_limit = abs(position) >= limit
        mm_history_deque.append(hit_limit)

        # Calculate liquidation flags
        soft_liquidate = len(mm_history_deque) == self.mm_history_size and sum(mm_history_deque) >= self.mm_history_size * MM_SOFT_LIQUIDATION_THRESHOLD and mm_history_deque[-1]
        hard_liquidate = len(mm_history_deque) == self.mm_history_size and all(mm_history_deque)

        # Adjust target prices
        max_buy_price = default_price - RESIN_PRICE_ADJUSTMENT if position > limit * RESIN_POSITION_SKEW_THRESHOLD else default_price
        min_sell_price = default_price + RESIN_PRICE_ADJUSTMENT if position < limit * -RESIN_POSITION_SKEW_THRESHOLD else default_price

        # --- Process Buys Sequentially ---

        # 1. Aggressive Buys
        if sell_orders_book: # Check if sell book exists
            for price, volume in sell_orders_book:
                if price <= max_buy_price and to_buy > 0:
                    quantity = min(-volume, to_buy)
                    self.buy(product, price, quantity)
                    to_buy -= quantity

        # 2. Hard Liquidate Buys
        if hard_liquidate and to_buy > 0:
            quantity = to_buy // 2
            if quantity > 0: # Avoid placing zero quantity orders
                self.buy(product, default_price, quantity)
                to_buy -= quantity

        # 3. Soft Liquidate Buys
        if soft_liquidate and to_buy > 0:
            quantity = to_buy // 2
            if quantity > 0: # Avoid placing zero quantity orders
                self.buy(product, default_price - MM_PRICE_ADJUSTMENT, quantity)
                to_buy -= quantity

        # 4. Quoting Buys (if capacity remains)
        if to_buy > 0 and buy_orders_book: # Check if buy book exists for quoting reference
            try:
                # Find buy price with largest volume
                most_popular_price = max(buy_orders_book, key=lambda item: item[1])[0]
                quote_buy_price = min(max_buy_price, most_popular_price + 1)
                self.buy(product, quote_buy_price, to_buy) # Buy remaining capacity
            except ValueError:
                logger.print(f"[{product} MM Quote] Error: Buy book empty for quoting popular price.")
                # Fallback: Quote at max_buy_price if book is empty but we still want to quote
                self.buy(product, max_buy_price, to_buy)

        # --- Process Sells Sequentially ---

        # 5. Aggressive Sells
        if buy_orders_book: # Check if buy book exists
            for price, volume in buy_orders_book:
                if price >= min_sell_price and to_sell > 0:
                    quantity = min(volume, to_sell)
                    self.sell(product, price, quantity)
                    to_sell -= quantity

        # 6. Hard Liquidate Sells
        if hard_liquidate and to_sell > 0:
            quantity = to_sell // 2
            if quantity > 0: # Avoid placing zero quantity orders
                self.sell(product, default_price, quantity)
                to_sell -= quantity

        # 7. Soft Liquidate Sells
        if soft_liquidate and to_sell > 0:
            quantity = to_sell // 2
            if quantity > 0: # Avoid placing zero quantity orders
                self.sell(product, default_price + MM_PRICE_ADJUSTMENT, quantity)
                to_sell -= quantity

        # 8. Quoting Sells (if capacity remains)
        if to_sell > 0 and sell_orders_book: # Check if sell book exists for quoting reference
            try:
                # Find sell price with largest absolute volume (volume is negative)
                most_popular_price = min(sell_orders_book, key=lambda item: item[1])[0] # min price has largest abs volume
                quote_sell_price = max(min_sell_price, most_popular_price - 1)
                self.sell(product, quote_sell_price, to_sell) # Sell remaining capacity
            except ValueError:
                logger.print(f"[{product} MM Quote] Error: Sell book empty for quoting popular price.")
                # Fallback: Quote at min_sell_price if book is empty but we still want to quote
                self.sell(product, min_sell_price, to_sell)

        # Save updated deque back to product state dict
        product_state["mm_history"] = list(mm_history_deque)

        # Return the populated self.orders list
        final_orders = self.orders
        self.orders = [] # Reset for next product/call
        return final_orders

    # Helper methods to add orders to a temporary list within the class instance
    def buy(self, product: str, price: int, quantity: int) -> None:
        if not hasattr(self, 'orders'): self.orders = []
        if quantity > 0:
            self.orders.append(Order(product, price, quantity))

    def sell(self, product: str, price: int, quantity: int) -> None:
        if not hasattr(self, 'orders'): self.orders = []
        if quantity > 0:
            self.orders.append(Order(product, price, -quantity))

    def _rainforest_resin_strategy(self, product, order_depth, position, product_state):
        """RAINFOREST_RESIN: Market making strategy with fixed default price."""
        return self._market_making_logic(product, order_depth, position, product_state, RESIN_DEFAULT_PRICE)

    def _kelp_strategy(self, product, order_depth, position, product_state):
        """KELP: Market making strategy with default price based on most popular orders."""
        sell_orders = order_depth.sell_orders.items()
        buy_orders = order_depth.buy_orders.items()

        # Calculate default price (handle empty books)
        default_price = KELP_DEFAULT_PRICE # Fallback default
        try:
            if sell_orders and buy_orders:
                most_popular_sell_price = min(sell_orders, key = lambda item : -item[1])[0] # Min price with max volume
                most_popular_buy_price = max(buy_orders, key = lambda item : item[1])[0] # Max price with max volume
                default_price = (most_popular_buy_price + most_popular_sell_price) // 2
            elif buy_orders:
                # Only buy orders exist, estimate based on best bid
                default_price = max(buy_orders, key = lambda item : item[1])[0]
            elif sell_orders:
                # Only sell orders exist, estimate based on best ask
                default_price = min(sell_orders, key = lambda item : -item[1])[0]
        except ValueError:
            logger.print(f"[Kelp MM Default] Error calculating default price, falling back.")

        return self._market_making_logic(product, order_depth, position, product_state, default_price)

    def _squid_ink_strategy(self, product, order_depth, position, history):
        """
        SQUID_INK: Mean reversion based on deviation from recent average (Hint-based).
        Uses Z-score and requires crossing back from extremes.
        Returns tuple: (list[Order], current_z_score)
        """
        params = self.squid_params
        limit = self.position_limits[product]
        orders = []
        current_z = 0.0 # Default z-score if calculation fails

        if len(history["prices"]) < params["window"]:
            return orders, current_z # Return default z if not enough data

        recent_prices = history["prices"][-params["window"]:]
        moving_average = statistics.mean(recent_prices)
        stdev = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 1e-6 # Avoid division by zero
        if stdev < 1e-6: # Handle case of zero stdev
            stdev = 1e-6

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_ask is None or best_bid is None:
            # If market disappears, return previous z to avoid large jumps in state
            return orders, history.get("prev_z", 0.0)

        current_price = (best_bid + best_ask) / 2
        current_z = (current_price - moving_average) / stdev

        available_to_buy = limit - position
        available_to_sell = limit + position

        # Load previous z-score from history dict
        prev_z = history.get("prev_z", 0.0)

        entry_z = params["z_entry_threshold"]
        exit_z = params["z_exit_threshold"]

        logger.print(f"[Squid] mid={current_price:.2f} avg={moving_average:.2f} std={stdev:.2f} z={current_z:.2f} prev_z={prev_z:.2f} pos={position}")

        # Buy condition: Was extremely low (prev_z < -entry_z) AND has started reverting (current_z > -exit_z)
        if prev_z < -entry_z and current_z > -exit_z and available_to_buy > 0:
            if best_ask in order_depth.sell_orders and order_depth.sell_orders[best_ask] < 0:
                volume = min(params["order_size"], available_to_buy, -order_depth.sell_orders[best_ask])
                if volume > 0:
                    orders.append(Order(product, best_ask, volume))
                    logger.print(f"SQUID BUY triggered @ {best_ask} vol {volume} (z crossed back from {prev_z:.2f} to {current_z:.2f})")
            else:
                logger.print(f"Squid: Buy signal but no volume at best ask {best_ask}")

        # Sell condition: Was extremely high (prev_z > entry_z) AND has started reverting (current_z < exit_z)
        elif prev_z > entry_z and current_z < exit_z and available_to_sell > 0:
            if best_bid in order_depth.buy_orders and order_depth.buy_orders[best_bid] > 0:
                volume = min(params["order_size"], available_to_sell, order_depth.buy_orders[best_bid])
                if volume > 0:
                    orders.append(Order(product, best_bid, -volume))
                    logger.print(f"SQUID SELL triggered @ {best_bid} vol {volume} (z crossed back from {prev_z:.2f} to {current_z:.2f})")
            else:
                logger.print(f"Squid: Sell signal but no volume at best bid {best_bid}")

        # Return orders and the *current* z-score to be saved as prev_z for the next tick
        return orders, current_z 