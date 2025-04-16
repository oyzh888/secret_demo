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
        "max_trades_per_strike": 5
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.voucher_strikes = [9500, 9750, 10000, 10250, 10500]
        self.initial_capital = self.params[Product.VOLCANIC_ROCK]["initial_capital"]
        self.current_capital = self.initial_capital
        self.round = 1  # Track current round for expiration calculation

    def get_swmid(self, order_depth):
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        return None

    def calculate_volatility(self, price_history: List[float]) -> float:
        if len(price_history) < 2:
            return 0.0
        returns = np.diff(price_history) / price_history[:-1]
        return np.std(returns)

    def calculate_moneyness(self, spot_price: float, strike: float, tte: float) -> float:
        """Calculate moneyness as per the hint: m_t = log(K/St)/sqrt(TTE)"""
        return math.log(strike / spot_price) / math.sqrt(tte)

    def calculate_implied_volatility(self, spot_price: float, option_price: float, strike: float, tte: float) -> float:
        """Calculate Black-Scholes implied volatility"""
        def black_scholes_price(vol):
            d1 = (math.log(spot_price/strike) + (vol**2/2)*tte) / (vol*math.sqrt(tte))
            d2 = d1 - vol*math.sqrt(tte)
            return spot_price * norm.cdf(d1) - strike * math.exp(-0.05*tte) * norm.cdf(d2) - option_price

        # Use binary search to find implied volatility
        low = 0.001
        high = 5.0
        while high - low > 0.0001:
            mid = (low + high) / 2
            if black_scholes_price(mid) > 0:
                high = mid
            else:
                low = mid
        return (low + high) / 2

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0
        
        logger.print(f"===== TIMESTAMP: {state.timestamp} =====")
        logger.print(f"Current capital: {self.current_capital}")
        
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return result, conversions, jsonpickle.encode(traderObject)
            
        volcanic_rock = state.order_depths[Product.VOLCANIC_ROCK]
        volcanic_rock_mid = self.get_swmid(volcanic_rock)
        
        if "price_history" not in traderObject:
            traderObject["price_history"] = {}
        if Product.VOLCANIC_ROCK not in traderObject["price_history"]:
            traderObject["price_history"][Product.VOLCANIC_ROCK] = []
            
        traderObject["price_history"][Product.VOLCANIC_ROCK].append(volcanic_rock_mid)
        if len(traderObject["price_history"][Product.VOLCANIC_ROCK]) > self.params[Product.VOLCANIC_ROCK]["window"]:
            traderObject["price_history"][Product.VOLCANIC_ROCK].pop(0)
            
        volatility = self.calculate_volatility(traderObject["price_history"][Product.VOLCANIC_ROCK])
        
        # Calculate time to expiration (7 days - current round)
        tte = max(7 - self.round, 0.1)  # Minimum 0.1 to avoid division by zero
        
        # Calculate moneyness and implied volatility for each strike
        for strike in self.voucher_strikes:
            voucher_name = f"VOLCANIC_ROCK_VOUCHER_{strike}"
            if voucher_name not in state.order_depths:
                continue
                
            voucher = state.order_depths[voucher_name]
            voucher_mid = self.get_swmid(voucher)
            
            if voucher_mid is None:
                continue
                
            moneyness = self.calculate_moneyness(volcanic_rock_mid, strike, tte)
            implied_vol = self.calculate_implied_volatility(volcanic_rock_mid, voucher_mid, strike, tte)
            
            logger.print(f"Voucher {voucher_name}: moneyness={moneyness:.4f}, implied_vol={implied_vol:.4f}")
            
            if volatility > self.params[Product.VOLCANIC_ROCK]["volatility_threshold"]:
                # High volatility - implement straddle
                if abs(moneyness) < 0.1:  # Close to at-the-money
                    if voucher_name not in result:
                        result[voucher_name] = []
                    position_limit = POSITION_LIMITS[voucher_name]
                    max_trades = min(position_limit, self.params[Product.VOLCANIC_ROCK]["max_trades_per_strike"])
                    
                    result[voucher_name].append(Order(voucher_name, max(voucher.buy_orders.keys()), max_trades))
                    result[voucher_name].append(Order(voucher_name, min(voucher.sell_orders.keys()), -max_trades))
                    logger.print(f"STRADDLE: Buy straddle on {voucher_name} with {max_trades} contracts")

        # Increment round counter
        self.round += 1

        traderData = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData 