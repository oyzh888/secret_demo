      
import json
from typing import Any
from abc import abstractmethod
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import deque
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

class MarketMakingStrategy(Strategy):
    def __init__(self, product: str, limit: int):
        super().__init__(product, limit)

        self.history = deque()
        self.history_size = 10

    def act(self, state: TradingState) -> None:
        buy_orders = sorted(state.order_depths[self.product].buy_orders.items(), reverse = True)
        sell_orders = sorted(state.order_depths[self.product].sell_orders.items())

        position = state.position.get(self.product, 0)

        to_buy = self.limit - position
        to_sell = self.limit + position

        default_price = self.get_default_price(state)

        self.history.append(abs(position) == self.limit)
        
        if len(self.history) > self.history_size:
            self.history.popleft()

        soft_liquidate = len(self.history) == self.history_size and sum(self.history) >= self.history_size / 2 and self.history[-1]
        hard_liquidate = len(self.history) == self.history_size and all(self.history)

        max_buy_price = default_price - 1 if position > self.limit * 0.5 else default_price

        min_sell_price = default_price + 1 if position < self.limit * -0.5 else default_price  

        for price, volume in sell_orders:
            if price <= max_buy_price and to_buy > 0:
                quantity = min(-volume, to_buy)
                self.buy(price, quantity)
                to_buy -= quantity
                
        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(default_price, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy //2
            self.buy(default_price - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            most_popular_price = max(buy_orders, key = lambda item: item[1])[0]
            price = min(max_buy_price, most_popular_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if price >= min_sell_price and to_sell > 0:
                quantity = min(volume, to_buy)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:            
            quantity = to_sell // 2
            self.sell(default_price, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell (default_price + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            most_popular_price = min(sell_orders, key = lambda item: item[1])[0]
            price = max(min_sell_price, most_popular_price - 1)
            self.sell(price, to_sell)

    @abstractmethod
    def get_default_price(self, state: TradingState) -> int:
        raise NotImplementedError()
    
    def save(self) -> JSON:
        return list(self.history)
    
    def load(self, data : JSON) -> None:
        self.history = deque(data)


class RainForestResinStrategy(MarketMakingStrategy):
    def get_default_price(self, state: TradingState) -> int:
        return 10_000


class KelpStrategy(MarketMakingStrategy):
    def get_default_price(self, state: TradingState) -> int:
        order_depths = state.order_depths[self.product]
        sell_orders = order_depths.sell_orders.items()
        buy_orders = order_depths.buy_orders.items()

        most_popular_sell_price = min(sell_orders, key = lambda item : item[1])[0]
        most_popular_buy_price = max(buy_orders, key = lambda item : item[1])[0]
        
        return (most_popular_buy_price + most_popular_sell_price)//2


class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP" : 50
        }

        self.strategies = { symbol : strategyClass(symbol, limits[symbol]) for symbol, strategyClass in {
            "RAINFOREST_RESIN" : RainForestResinStrategy,
            "KELP" : KelpStrategy
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load(old_trader_data.get(symbol, None))

            if symbol in state.order_depths:
                result[symbol] = strategy.run(state)

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data

    