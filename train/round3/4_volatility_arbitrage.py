import json
from typing import Any, List, Dict
import string
import jsonpickle
import numpy as np
import math
from scipy.stats import norm

from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation, Listing, Observation, Symbol, Trade, ProsperityEncoder

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

class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

# Position limits from the rules
POSITION_LIMITS = {
    Product.VOLCANIC_ROCK: 400,
    Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
    Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
    Product.VOLCANIC_ROCK_VOUCHER_10500: 200
}

# Strike prices from the rules
STRIKE_PRICES = {
    Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
    Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
    Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
    Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
    Product.VOLCANIC_ROCK_VOUCHER_10500: 10500
}

PARAMS = {
    Product.VOLCANIC_ROCK: {
        "volatility_threshold": 0.1,
        "window": 20,
        "initial_capital": 500000,
        "max_trades_per_strike": 5,
        "iv_smile_threshold": 0.05,  # Threshold for IV smile arbitrage
        "min_ttm": 0.1  # Minimum time to maturity
    }
}

class Trader:
    def __init__(self):
        self.position_limits = {
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }
        self.strike_prices = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500
        }
        self.initial_capital = 500000
        self.current_capital = self.initial_capital
        self.round = 1

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        return None

    def calculate_volatility_arbitrage(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        
        # Get current positions
        current_positions = state.position
        
        # Check if we have volcanic rock data
        if "VOLCANIC_ROCK" not in state.order_depths:
            return result
            
        # Get mid price of volcanic rock
        volcanic_rock = state.order_depths["VOLCANIC_ROCK"]
        volcanic_rock_mid = self.get_mid_price(volcanic_rock)
        
        if volcanic_rock_mid is None:
            return result
            
        # Calculate time to expiration (7 days - current round)
        tte = max(7 - self.round, 0.1)
        
        # Evaluate each strike price
        for strike in self.strike_prices:
            if strike not in state.order_depths:
                continue
                
            order_depth = state.order_depths[strike]
            if not order_depth.sell_orders or not order_depth.buy_orders:
                continue
                
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            
            # Calculate moneyness
            moneyness = (self.strike_prices[strike] - volcanic_rock_mid) / volcanic_rock_mid
            
            # Position limit check
            position_limit = self.position_limits[strike] - current_positions.get(strike, 0)
            
            if position_limit <= 0:
                continue
                
            # Implement volatility arbitrage based on moneyness
            if moneyness > 0.02:  # Overpriced
                if strike not in result:
                    result[strike] = []
                result[strike].append(Order(strike, best_bid, -min(position_limit, 5)))
                print(f"SELL {strike} at {best_bid} for volatility arbitrage")
                
            elif moneyness < -0.02:  # Underpriced
                if strike not in result:
                    result[strike] = []
                result[strike].append(Order(strike, best_ask, min(position_limit, 5)))
                print(f"BUY {strike} at {best_ask} for volatility arbitrage")
                
        return result

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        result = {}
        conversions = 0
        
        # Initialize trader data
        traderObject = {}
        if state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
            
        # Execute volatility arbitrage strategy
        result = self.calculate_volatility_arbitrage(state)
        
        # Increment round counter
        self.round += 1
        
        return result, conversions, jsonpickle.encode(traderObject) 