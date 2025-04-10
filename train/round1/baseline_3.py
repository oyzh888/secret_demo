import json
from typing import Any, Dict, List, Optional
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque
import numpy as np
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

class MarketStatistics:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.prices = deque(maxlen=window_size)
        self.volumes = deque(maxlen=window_size)
        self.price_volume_map = {}  # Maps price to total volume
        
    def update(self, price: int, volume: int) -> None:
        self.prices.append(price)
        self.volumes.append(volume)
        self.price_volume_map[price] = self.price_volume_map.get(price, 0) + volume
        
    def get_mean(self) -> float:
        return np.mean(self.prices) if self.prices else 0
        
    def get_std(self) -> float:
        return np.std(self.prices) if self.prices else 0
        
    def get_volume_weighted_price(self) -> float:
        if not self.prices or not self.volumes or sum(self.volumes) == 0:
            return self.get_mean() if self.prices else 0
        return np.average(self.prices, weights=self.volumes)
        
    def get_high_volume_prices(self, threshold: float = 0.7) -> List[int]:
        """Returns prices that have above threshold percentile volume"""
        if not self.price_volume_map:
            return []
        max_volume = max(self.price_volume_map.values())
        return [p for p, v in self.price_volume_map.items() 
                if v >= threshold * max_volume]

class SquidInkStrategy:
    def __init__(self, product: str, limit: int):
        self.product = product
        self.limit = limit
        self.orders = []
        self.stats = MarketStatistics()
        self.position = 0
        self.last_trade_price = None
        
    def run(self, state: TradingState) -> List[Order]:
        self.orders = []
        self.position = state.position.get(self.product, 0)
        
        # Update statistics with current market data
        self._update_statistics(state)
        
        # Calculate dynamic parameters
        mean_price = self.stats.get_mean()
        std_dev = self.stats.get_std()
        vwap = self.stats.get_volume_weighted_price()
        
        # Get high volume price levels
        high_volume_prices = self.stats.get_high_volume_prices()
        
        # Calculate spread parameters
        base_spread = max(1, int(std_dev * 0.1))  # Dynamic spread based on volatility
        position_adjustment = abs(self.position) / self.limit
        
        # Get current order book
        order_depth = state.order_depths[self.product]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())
        
        # Calculate fair price
        fair_price = self._calculate_fair_price(mean_price, vwap, high_volume_prices)
        
        # Place orders
        self._place_orders(fair_price, base_spread, position_adjustment, 
                          buy_orders, sell_orders, high_volume_prices)
        
        return self.orders
    
    def _update_statistics(self, state: TradingState) -> None:
        # Update with current trades
        for trade in state.market_trades.get(self.product, []):
            self.stats.update(trade.price, abs(trade.quantity))
            self.last_trade_price = trade.price
            
        # Update with current order book
        order_depth = state.order_depths[self.product]
        for price, volume in order_depth.buy_orders.items():
            self.stats.update(price, volume)
        for price, volume in order_depth.sell_orders.items():
            self.stats.update(price, volume)
    
    def _calculate_fair_price(self, mean_price: float, vwap: float, 
                            high_volume_prices: List[int]) -> int:
        """Calculate fair price using multiple factors"""
        if not high_volume_prices:
            return int((mean_price + vwap) / 2)
            
        # Weight high volume prices more heavily
        high_volume_weight = 0.4
        mean_weight = 0.3
        vwap_weight = 0.3
        
        high_volume_avg = np.mean(high_volume_prices)
        return int(high_volume_weight * high_volume_avg + 
                  mean_weight * mean_price + 
                  vwap_weight * vwap)
    
    def _place_orders(self, fair_price: int, base_spread: int, 
                     position_adjustment: float, buy_orders: List[tuple], 
                     sell_orders: List[tuple], high_volume_prices: List[int]) -> None:
        """Place orders with dynamic spread and position-based adjustments"""
        # Calculate position-based spread adjustment
        spread_adjustment = 1 + position_adjustment
        
        # Calculate buy and sell prices
        buy_price = fair_price - int(base_spread * spread_adjustment)
        sell_price = fair_price + int(base_spread * spread_adjustment)
        
        # Adjust prices to align with high volume levels if close
        for price in high_volume_prices:
            if abs(price - buy_price) <= 2:
                buy_price = price - 1
            if abs(price - sell_price) <= 2:
                sell_price = price + 1
        
        # Calculate quantities based on position
        max_buy = self.limit - self.position
        max_sell = self.limit + self.position
        
        # Place buy orders
        if max_buy > 0:
            quantity = min(max_buy, 10)  # Limit order size
            self.orders.append(Order(self.product, buy_price, quantity))
            
        # Place sell orders
        if max_sell > 0:
            quantity = min(max_sell, 10)  # Limit order size
            self.orders.append(Order(self.product, sell_price, -quantity))

class Trader:
    def __init__(self) -> None:
        self.strategies = {
            "SQUID_INK": SquidInkStrategy("SQUID_INK", 50)
        }
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        
        for symbol, strategy in self.strategies.items():
            if symbol in state.order_depths:
                result[symbol] = strategy.run(state)
        
        logger.flush(state, result, conversions, "")
        return result, conversions, "" 