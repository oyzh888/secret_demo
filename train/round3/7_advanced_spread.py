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
        "window": 20,
        "initial_capital": 500000,
        "max_trades_per_strike": 5,
        "spread_threshold": 0.02,  # Minimum spread profit threshold
        "min_ttm": 0.1,  # Minimum time to maturity
        "volatility_threshold": 0.1,  # Threshold for volatility-based spreads
        "iv_smile_threshold": 0.05,  # Threshold for IV smile arbitrage
        "calendar_spread_threshold": 0.03  # Threshold for calendar spread profit
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

    def calculate_butterfly_profit(self, lower_strike: int, middle_strike: int, upper_strike: int, state: TradingState) -> float:
        """Calculate butterfly spread profit potential"""
        lower_voucher = f"VOLCANIC_ROCK_VOUCHER_{lower_strike}"
        middle_voucher = f"VOLCANIC_ROCK_VOUCHER_{middle_strike}"
        upper_voucher = f"VOLCANIC_ROCK_VOUCHER_{upper_strike}"
        
        if (lower_voucher not in state.order_depths or 
            middle_voucher not in state.order_depths or 
            upper_voucher not in state.order_depths):
            return 0.0
            
        lower_depth = state.order_depths[lower_voucher]
        middle_depth = state.order_depths[middle_voucher]
        upper_depth = state.order_depths[upper_voucher]
        
        if (not lower_depth.sell_orders or 
            not middle_depth.buy_orders or 
            not middle_depth.sell_orders or 
            not upper_depth.sell_orders):
            return 0.0
            
        lower_ask = min(lower_depth.sell_orders.keys())
        middle_bid = max(middle_depth.buy_orders.keys())
        middle_ask = min(middle_depth.sell_orders.keys())
        upper_ask = min(upper_depth.sell_orders.keys())
        
        # Butterfly spread cost
        spread_cost = lower_ask - 2 * middle_bid + upper_ask
        
        # Maximum profit is the distance between strikes
        max_profit = upper_strike - lower_strike
        
        # Net profit
        net_profit = max_profit - spread_cost
        
        return net_profit

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        result = {}
        conversions = 0
        
        # Initialize trader data
        traderObject = {}
        if state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        # Get current positions
        current_positions = state.position
        
        # Check if we have volcanic rock data
        if "VOLCANIC_ROCK" not in state.order_depths:
            return result, conversions, jsonpickle.encode(traderObject)
            
        # Get mid price of volcanic rock
        volcanic_rock = state.order_depths["VOLCANIC_ROCK"]
        volcanic_rock_mid = self.get_mid_price(volcanic_rock)
        
        if volcanic_rock_mid is None:
            return result, conversions, jsonpickle.encode(traderObject)
            
        # Calculate time to expiration (7 days - current round)
        tte = max(7 - self.round, 0.1)
        
        # Evaluate butterfly spreads
        strikes = [9500, 9750, 10000, 10250, 10500]
        for i in range(len(strikes) - 2):
            lower_strike = strikes[i]
            middle_strike = strikes[i + 1]
            upper_strike = strikes[i + 2]
            
            butterfly_profit = self.calculate_butterfly_profit(lower_strike, middle_strike, upper_strike, state)
            
            # Check if we have enough capital and position limits
            lower_voucher = f"VOLCANIC_ROCK_VOUCHER_{lower_strike}"
            middle_voucher = f"VOLCANIC_ROCK_VOUCHER_{middle_strike}"
            upper_voucher = f"VOLCANIC_ROCK_VOUCHER_{upper_strike}"
            
            position_limit = min(
                self.position_limits[lower_voucher] - current_positions.get(lower_voucher, 0),
                self.position_limits[middle_voucher] - current_positions.get(middle_voucher, 0),
                self.position_limits[upper_voucher] - current_positions.get(upper_voucher, 0)
            )
            
            if butterfly_profit > 0 and position_limit > 0:
                # Implement butterfly spread
                if lower_voucher not in result:
                    result[lower_voucher] = []
                if middle_voucher not in result:
                    result[middle_voucher] = []
                if upper_voucher not in result:
                    result[upper_voucher] = []
                    
                # Buy lower strike, sell 2x middle strike, buy upper strike
                result[lower_voucher].append(Order(lower_voucher, min(state.order_depths[lower_voucher].sell_orders.keys()), position_limit))
                result[middle_voucher].append(Order(middle_voucher, max(state.order_depths[middle_voucher].buy_orders.keys()), -2 * position_limit))
                result[upper_voucher].append(Order(upper_voucher, min(state.order_depths[upper_voucher].sell_orders.keys()), position_limit))
                print(f"BUTTERFLY: Buy {lower_voucher}, Sell 2x {middle_voucher}, Buy {upper_voucher}, Profit: {butterfly_profit:.2f}")

        # Increment round counter
        self.round += 1

        return result, conversions, jsonpickle.encode(traderObject) 