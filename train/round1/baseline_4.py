import json
from typing import Any, Dict, List, Deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque
import numpy as np
import pandas as pd
from typing import Any, TypeAlias

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
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

class Strategy:
    def __init__(self, product: str, limit: int):
        self.product = product
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.product, price, quantity))

    def sell(self, price: int, quantity: int) -> None:
        self.orders.append(Order(self.product, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class SquidInkStrategy(Strategy):
    def __init__(self, product: str, limit: int):
        super().__init__(product, limit)
        self.price_history: Deque[int] = deque(maxlen=100)  # Store last 100 prices
        self.volume_history: Deque[int] = deque(maxlen=100)  # Store last 100 volumes
        self.position_history: Deque[int] = deque(maxlen=100)  # Store last 100 positions
        self.bb_window = 20  # Bollinger Bands window
        self.bb_std = 2  # Number of standard deviations
        self.dynamic_spread = 1.0  # Initial spread multiplier
        self.min_spread = 0.5
        self.max_spread = 2.0
        self.volume_threshold = 10  # Volume threshold for significant trades
        self.position_limit = limit

    def update_parameters(self, state: TradingState) -> None:
        # Update price and volume history
        current_price = self.get_mid_price(state)
        current_volume = self.get_total_volume(state)
        current_position = state.position.get(self.product, 0)
        
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        self.position_history.append(current_position)

        if len(self.price_history) >= self.bb_window:
            # Calculate Bollinger Bands
            prices = np.array(self.price_history)
            sma = np.mean(prices)
            std = np.std(prices)
            
            # Calculate price volatility
            volatility = std / sma if sma != 0 else 0
            
            # Adjust spread based on volatility
            self.dynamic_spread = max(self.min_spread, min(self.max_spread, 1.0 + volatility * 2))
            
            # Adjust volume threshold based on recent volume patterns
            recent_volumes = np.array(self.volume_history)
            self.volume_threshold = int(np.percentile(recent_volumes, 75))

    def get_mid_price(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.product]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else 0
        return (best_bid + best_ask) // 2 if best_bid and best_ask else 0

    def get_total_volume(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.product]
        total_volume = sum(abs(vol) for vol in order_depth.buy_orders.values())
        total_volume += sum(abs(vol) for vol in order_depth.sell_orders.values())
        return total_volume

    def calculate_bollinger_bands(self) -> tuple[float, float, float]:
        if len(self.price_history) < self.bb_window:
            return 0, 0, 0
            
        prices = np.array(self.price_history)
        sma = np.mean(prices)
        std = np.std(prices)
        upper_band = sma + self.bb_std * std
        lower_band = sma - self.bb_std * std
        return sma, upper_band, lower_band

    def act(self, state: TradingState) -> None:
        self.update_parameters(state)
        
        if len(self.price_history) < self.bb_window:
            # Not enough data yet, use simple market making
            self.simple_market_making(state)
            return

        current_price = self.get_mid_price(state)
        current_volume = self.get_total_volume(state)
        current_position = state.position.get(self.product, 0)
        
        sma, upper_band, lower_band = self.calculate_bollinger_bands()
        
        # Calculate position limits
        max_buy = self.position_limit - current_position
        max_sell = self.position_limit + current_position
        
        # Calculate spread
        spread = int(self.dynamic_spread * (upper_band - lower_band) / 4)
        
        # Identify trading opportunities
        if current_price > upper_band and current_volume > self.volume_threshold:
            # Price is above upper band with high volume - potential sell opportunity
            sell_price = int(upper_band + spread)
            quantity = min(max_sell, 5)  # Conservative position sizing
            self.sell(sell_price, quantity)
            
        elif current_price < lower_band and current_volume > self.volume_threshold:
            # Price is below lower band with high volume - potential buy opportunity
            buy_price = int(lower_band - spread)
            quantity = min(max_buy, 5)  # Conservative position sizing
            self.buy(buy_price, quantity)
        
        # Regular market making
        buy_price = int(sma - spread)
        sell_price = int(sma + spread)
        
        # Adjust quantities based on position
        buy_quantity = min(max_buy, 3)
        sell_quantity = min(max_sell, 3)
        
        self.buy(buy_price, buy_quantity)
        self.sell(sell_price, sell_quantity)

    def simple_market_making(self, state: TradingState) -> None:
        # Simple market making when we don't have enough data
        current_price = self.get_mid_price(state)
        current_position = state.position.get(self.product, 0)
        
        max_buy = self.position_limit - current_position
        max_sell = self.position_limit + current_position
        
        spread = 2  # Fixed spread for initial market making
        xw
        buy_price = current_price - spread
        sell_price = current_price + spread
        
        buy_quantity = min(max_buy, 3)
        sell_quantity = min(max_sell, 3)
        
        self.buy(buy_price, buy_quantity)
        self.sell(sell_price, sell_quantity)

    def save(self) -> JSON:
        return {
            "price_history": list(self.price_history),
            "volume_history": list(self.volume_history),
            "position_history": list(self.position_history),
            "dynamic_spread": self.dynamic_spread,
            "volume_threshold": self.volume_threshold
        }

    def load(self, data: JSON) -> None:
        if data:
            self.price_history = deque(data["price_history"], maxlen=100)
            self.volume_history = deque(data["volume_history"], maxlen=100)
            self.position_history = deque(data["position_history"], maxlen=100)
            self.dynamic_spread = data["dynamic_spread"]
            self.volume_threshold = data["volume_threshold"]

class Trader:
    def __init__(self):
        self.strategies = {
            "SQUID_INK": SquidInkStrategy("SQUID_INK", 20)
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""

        for product, strategy in self.strategies.items():
            if product in state.order_depths:
                orders = strategy.run(state)
                if orders:
                    result[product] = orders

        return result, conversions, trader_data 