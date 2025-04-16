import json
from typing import Any, List, Dict
import string
import jsonpickle
import numpy as np
import math

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
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = "KELP"

PARAMS = {
    "volatility": {
        "window": 20,
        "threshold": 0.02,
        "max_leverage": 3
    },
    "momentum": {
        "window": 10,
        "threshold": 0.05
    },
    "straddle": {
        "volatility_multiplier": 1.5
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        
        self.LIMIT = {
            Product.VOLCANIC_ROCK: 100,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 100,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 100,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 100,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 100,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 100,
            Product.PICNIC_BASKET1: 70,
            Product.PICNIC_BASKET2: 70,
            Product.CROISSANTS: 250,
            Product.JAMS: 250,
            Product.DJEMBES: 250,
            Product.RAINFOREST_RESIN: 100,
            Product.SQUID_INK: 100,
            Product.KELP: 100
        }
        
        self.voucher_strikes = [9500, 9750, 10000, 10250, 10500]
        
    def calculate_volatility(self, price_history: List[float]) -> float:
        if len(price_history) < 2:
            return 0.0
        returns = np.diff(np.log(price_history))
        return np.std(returns) * math.sqrt(252)
        
    def calculate_momentum(self, price_history: List[float]) -> float:
        if len(price_history) < self.params["momentum"]["window"]:
            return 0.0
        return (price_history[-1] - price_history[0]) / price_history[0]
        
    def get_swmid(self, order_depth: OrderDepth) -> float:
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        return None
        
    def strategy_volatility_arbitrage(self, state: TradingState, trader_data: dict) -> Dict[Symbol, List[Order]]:
        result = {}
        
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return result
            
        volcanic_rock = state.order_depths[Product.VOLCANIC_ROCK]
        volcanic_rock_mid = self.get_swmid(volcanic_rock)
        
        if "price_history" not in trader_data:
            trader_data["price_history"] = {}
        if Product.VOLCANIC_ROCK not in trader_data["price_history"]:
            trader_data["price_history"][Product.VOLCANIC_ROCK] = []
            
        trader_data["price_history"][Product.VOLCANIC_ROCK].append(volcanic_rock_mid)
        if len(trader_data["price_history"][Product.VOLCANIC_ROCK]) > self.params["volatility"]["window"]:
            trader_data["price_history"][Product.VOLCANIC_ROCK].pop(0)
            
        vol_volatility = self.calculate_volatility(trader_data["price_history"][Product.VOLCANIC_ROCK])
        
        for strike in self.voucher_strikes:
            voucher_name = f"VOLCANIC_ROCK_VOUCHER_{strike}"
            if voucher_name not in state.order_depths:
                continue
                
            voucher = state.order_depths[voucher_name]
            voucher_mid = self.get_swmid(voucher)
            intrinsic_value = max(0, volcanic_rock_mid - strike)
            time_value = voucher_mid - intrinsic_value
            
            if vol_volatility > self.params["volatility"]["threshold"]:
                # High volatility strategy
                if strike > volcanic_rock_mid * 1.05 and time_value < vol_volatility * strike * 0.1:
                    if voucher_name not in result:
                        result[voucher_name] = []
                    result[voucher_name].append(Order(voucher_name, max(voucher.buy_orders.keys()), 2))
                    logger.print(f"VOLATILITY ARBITRAGE: Buy {voucher_name} at {max(voucher.buy_orders.keys())}")
                elif strike < volcanic_rock_mid * 0.95 and time_value > vol_volatility * strike * 0.15:
                    if voucher_name not in result:
                        result[voucher_name] = []
                    result[voucher_name].append(Order(voucher_name, min(voucher.sell_orders.keys()), -2))
                    logger.print(f"VOLATILITY ARBITRAGE: Sell {voucher_name} at {min(voucher.sell_orders.keys())}")
                    
        return result
        
    def strategy_momentum_trading(self, state: TradingState, trader_data: dict) -> Dict[Symbol, List[Order]]:
        result = {}
        
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return result
            
        volcanic_rock = state.order_depths[Product.VOLCANIC_ROCK]
        volcanic_rock_mid = self.get_swmid(volcanic_rock)
        
        if "price_history" not in trader_data:
            trader_data["price_history"] = {}
        if Product.VOLCANIC_ROCK not in trader_data["price_history"]:
            trader_data["price_history"][Product.VOLCANIC_ROCK] = []
            
        momentum = self.calculate_momentum(trader_data["price_history"][Product.VOLCANIC_ROCK])
        
        for strike in self.voucher_strikes:
            voucher_name = f"VOLCANIC_ROCK_VOUCHER_{strike}"
            if voucher_name not in state.order_depths:
                continue
                
            voucher = state.order_depths[voucher_name]
            
            if momentum > self.params["momentum"]["threshold"]:
                # Strong upward momentum
                if abs(strike - volcanic_rock_mid) / volcanic_rock_mid < 0.02:
                    if voucher_name not in result:
                        result[voucher_name] = []
                    result[voucher_name].append(Order(voucher_name, max(voucher.buy_orders.keys()), 3))
                    logger.print(f"MOMENTUM TRADING: Buy {voucher_name} at {max(voucher.buy_orders.keys())}")
                elif strike < volcanic_rock_mid * 0.95:
                    if voucher_name not in result:
                        result[voucher_name] = []
                    result[voucher_name].append(Order(voucher_name, min(voucher.sell_orders.keys()), -2))
                    logger.print(f"MOMENTUM TRADING: Sell {voucher_name} at {min(voucher.sell_orders.keys())}")
                    
            elif momentum < -self.params["momentum"]["threshold"]:
                # Strong downward momentum
                if strike < volcanic_rock_mid * 0.9:
                    if voucher_name not in result:
                        result[voucher_name] = []
                    result[voucher_name].append(Order(voucher_name, max(voucher.buy_orders.keys()), 2))
                    logger.print(f"MOMENTUM TRADING: Buy {voucher_name} at {max(voucher.buy_orders.keys())}")
                elif abs(strike - volcanic_rock_mid) / volcanic_rock_mid < 0.02:
                    if voucher_name not in result:
                        result[voucher_name] = []
                    result[voucher_name].append(Order(voucher_name, min(voucher.sell_orders.keys()), -3))
                    logger.print(f"MOMENTUM TRADING: Sell {voucher_name} at {min(voucher.sell_orders.keys())}")
                    
        return result
        
    def strategy_straddle(self, state: TradingState, trader_data: dict) -> Dict[Symbol, List[Order]]:
        result = {}
        
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return result
            
        volcanic_rock = state.order_depths[Product.VOLCANIC_ROCK]
        volcanic_rock_mid = self.get_swmid(volcanic_rock)
        
        if "price_history" not in trader_data:
            trader_data["price_history"] = {}
        if Product.VOLCANIC_ROCK not in trader_data["price_history"]:
            trader_data["price_history"][Product.VOLCANIC_ROCK] = []
            
        vol_volatility = self.calculate_volatility(trader_data["price_history"][Product.VOLCANIC_ROCK])
        
        closest_strike = min(self.voucher_strikes, key=lambda x: abs(x - volcanic_rock_mid))
        voucher_name = f"VOLCANIC_ROCK_VOUCHER_{closest_strike}"
        
        if vol_volatility > self.params["volatility"]["threshold"] * self.params["straddle"]["volatility_multiplier"]:
            if voucher_name not in result:
                result[voucher_name] = []
            result[voucher_name].append(Order(voucher_name, max(state.order_depths[voucher_name].buy_orders.keys()), 2))
            logger.print(f"STRADDLE: Buy {voucher_name} at {max(state.order_depths[voucher_name].buy_orders.keys())}")
            
        return result
        
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
            
        result = {}
        conversions = 0
        
        logger.print(f"===== TIMESTAMP: {state.timestamp} =====")
        
        # Initialize trader data if needed
        if "price_history" not in traderObject:
            traderObject["price_history"] = {}
            
        # Execute all strategies
        strategies = [
            self.strategy_volatility_arbitrage,
            self.strategy_momentum_trading,
            self.strategy_straddle
        ]
        
        for strategy in strategies:
            strategy_orders = strategy(state, traderObject)
            for symbol, orders in strategy_orders.items():
                if symbol not in result:
                    result[symbol] = []
                result[symbol].extend(orders)
                
        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
