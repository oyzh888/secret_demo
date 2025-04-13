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
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD = "SPREAD"


PARAMS = {
    Product.SPREAD: {
        "default_spread_mean": 30, 
        "default_spread_std": 70,
        "spread_std_window": 50,
        "zscore_threshold": 3, 
        "target_position": 50,
    },
}

# Define BASKET_WEIGHTS here to fix the error
BASKET_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.PICNIC_BASKET1: 60,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
        }

    # 返回买单量和卖单量
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # 最大可买入金额
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]
                    logger.print(f"TAKE BUY: {product}, price={best_ask}, quantity={quantity}, fair_value={fair_value}")

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # 我们能够卖出的最大量
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
                    logger.print(f"TAKE SELL: {product}, price={best_bid}, quantity={quantity}, fair_value={fair_value}")
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # 最大可购买金额
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
                        logger.print(f"TAKE BUY WITH ADVERSE: {product}, price={best_ask}, quantity={quantity}, fair_value={fair_value}")

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # 我们可以卖出的最大量
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
                        logger.print(f"TAKE SELL WITH ADVERSE: {product}, price={best_bid}, quantity={quantity}, fair_value={fair_value}")

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # 买单
            logger.print(f"MAKE BUY: {product}, price={round(bid)}, quantity={buy_quantity}")

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # 卖单
            logger.print(f"MAKE SELL: {product}, price={round(ask)}, quantity={sell_quantity}")
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # 所有价格高于平均卖价的买单的累积交易量
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
                logger.print(f"CLEAR SELL: {product}, price={fair_for_ask}, quantity={sent_quantity}, position={position_after_take}")

        if position_after_take < 0:
            # 所有价格低于平均买价的卖单的累计交易量
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
                logger.print(f"CLEAR BUY: {product}, price={fair_for_bid}, quantity={sent_quantity}, position={position_after_take}")

        return buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth):
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        return None

    def get_synthetic_basket_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        synthetic_order_depth = OrderDepth()

        if (
            Product.CROISSANTS not in order_depths
            or Product.JAMS not in order_depths
            or Product.DJEMBES not in order_depths
        ):
            return synthetic_order_depth

        croissants_order_depth = order_depths[Product.CROISSANTS]
        jams_order_depth = order_depths[Product.JAMS]
        djembes_order_depth = order_depths[Product.DJEMBES]

        # Calculate synthetic buy orders
        for croissants_sell_price, croissants_sell_volume in croissants_order_depth.sell_orders.items():
            for jams_sell_price, jams_sell_volume in jams_order_depth.sell_orders.items():
                for djembes_sell_price, djembes_sell_volume in djembes_order_depth.sell_orders.items():
                    synthetic_price = (
                        croissants_sell_price * BASKET_WEIGHTS[Product.CROISSANTS]
                        + jams_sell_price * BASKET_WEIGHTS[Product.JAMS]
                        + djembes_sell_price * BASKET_WEIGHTS[Product.DJEMBES]
                    )
                    synthetic_volume = min(
                        abs(croissants_sell_volume) // BASKET_WEIGHTS[Product.CROISSANTS],
                        abs(jams_sell_volume) // BASKET_WEIGHTS[Product.JAMS],
                        abs(djembes_sell_volume) // BASKET_WEIGHTS[Product.DJEMBES],
                    )
                    if synthetic_price in synthetic_order_depth.sell_orders:
                        synthetic_order_depth.sell_orders[synthetic_price] += -synthetic_volume
                    else:
                        synthetic_order_depth.sell_orders[synthetic_price] = -synthetic_volume

        # Calculate synthetic sell orders
        for croissants_buy_price, croissants_buy_volume in croissants_order_depth.buy_orders.items():
            for jams_buy_price, jams_buy_volume in jams_order_depth.buy_orders.items():
                for djembes_buy_price, djembes_buy_volume in djembes_order_depth.buy_orders.items():
                    synthetic_price = (
                        croissants_buy_price * BASKET_WEIGHTS[Product.CROISSANTS]
                        + jams_buy_price * BASKET_WEIGHTS[Product.JAMS]
                        + djembes_buy_price * BASKET_WEIGHTS[Product.DJEMBES]
                    )
                    synthetic_volume = min(
                        croissants_buy_volume // BASKET_WEIGHTS[Product.CROISSANTS],
                        jams_buy_volume // BASKET_WEIGHTS[Product.JAMS],
                        djembes_buy_volume // BASKET_WEIGHTS[Product.DJEMBES],
                    )
                    if synthetic_price in synthetic_order_depth.buy_orders:
                        synthetic_order_depth.buy_orders[synthetic_price] += synthetic_volume
                    else:
                        synthetic_order_depth.buy_orders[synthetic_price] = synthetic_volume

        return synthetic_order_depth

    def convert_synthetic_basket_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        result = {}
        result[Product.CROISSANTS] = []
        result[Product.JAMS] = []
        result[Product.DJEMBES] = []

        for order in synthetic_orders:
            if order.quantity > 0:  # Buy synthetic basket
                result[Product.CROISSANTS].append(
                    Order(
                        Product.CROISSANTS,
                        min(order_depths[Product.CROISSANTS].sell_orders.keys()),
                        order.quantity * BASKET_WEIGHTS[Product.CROISSANTS],
                    )
                )
                result[Product.JAMS].append(
                    Order(
                        Product.JAMS,
                        min(order_depths[Product.JAMS].sell_orders.keys()),
                        order.quantity * BASKET_WEIGHTS[Product.JAMS],
                    )
                )
                result[Product.DJEMBES].append(
                    Order(
                        Product.DJEMBES,
                        min(order_depths[Product.DJEMBES].sell_orders.keys()),
                        order.quantity * BASKET_WEIGHTS[Product.DJEMBES],
                    )
                )
                logger.print(f"BUY SYNTHETIC BASKET: quantity={order.quantity}, price={order.price}")
            else:  # Sell synthetic basket
                result[Product.CROISSANTS].append(
                    Order(
                        Product.CROISSANTS,
                        max(order_depths[Product.CROISSANTS].buy_orders.keys()),
                        order.quantity * BASKET_WEIGHTS[Product.CROISSANTS],
                    )
                )
                result[Product.JAMS].append(
                    Order(
                        Product.JAMS,
                        max(order_depths[Product.JAMS].buy_orders.keys()),
                        order.quantity * BASKET_WEIGHTS[Product.JAMS],
                    )
                )
                result[Product.DJEMBES].append(
                    Order(
                        Product.DJEMBES,
                        max(order_depths[Product.DJEMBES].buy_orders.keys()),
                        order.quantity * BASKET_WEIGHTS[Product.DJEMBES],
                    )
                )
                logger.print(f"SELL SYNTHETIC BASKET: quantity={abs(order.quantity)}, price={order.price}")

        return result

    def execute_spread_orders(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):
        if Product.PICNIC_BASKET1 not in order_depths:
            return None

        target_quantity = target_position - basket_position
        if target_quantity == 0:
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)

        if len(basket_order_depth.buy_orders) == 0 or len(basket_order_depth.sell_orders) == 0:
            return None
        if len(synthetic_order_depth.buy_orders) == 0 or len(synthetic_order_depth.sell_orders) == 0:
            return None

        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        logger.print(f"SPREAD: basket_swmid={basket_swmid}, synthetic_swmid={synthetic_swmid}, spread={spread}")

        aggregate_orders = {}
        if target_quantity > 0:  # Buy basket, sell synthetic
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            logger.print(f"EXECUTE SPREAD BUY BASKET: quantity={execute_volume}, basket_price={basket_ask_price}, synthetic_price={synthetic_bid_price}")
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, abs(target_quantity))

            basket_orders = [
                Order(Product.PICNIC_BASKET1, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.PICNIC_BASKET1] = basket_orders
            logger.print(f"EXECUTE SPREAD SELL BASKET: quantity={execute_volume}, basket_price={basket_bid_price}, synthetic_price={synthetic_ask_price}")
            return aggregate_orders

    def spread_orders(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.PICNIC_BASKET1 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.PICNIC_BASKET1]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)
        logger.print(f"SPREAD DATA: spread={spread}, basket_position={basket_position}")

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD]["default_spread_mean"]
        ) / spread_std
        logger.print(f"SPREAD ZSCORE: zscore={zscore}, spread_std={spread_std}")

        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                logger.print(f"SPREAD SIGNAL: HIGH ZSCORE - SELL BASKET")
                return self.execute_spread_orders(
                    -self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                logger.print(f"SPREAD SIGNAL: LOW ZSCORE - BUY BASKET")
                return self.execute_spread_orders(
                    self.params[Product.SPREAD]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0
        
        logger.print(f"===== TIMESTAMP: {state.timestamp} =====")
        
        if Product.SPREAD not in traderObject:
            traderObject[Product.SPREAD] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }
            logger.print("Initializing SPREAD data in traderObject")

        basket_position = (
            state.position[Product.PICNIC_BASKET1]
            if Product.PICNIC_BASKET1 in state.position
            else 0
        )
        logger.print(f"Current basket position: {basket_position}")
        
        # Log available products and their order depths
        logger.print(f"Available products: {list(state.order_depths.keys())}")
        for product, order_depth in state.order_depths.items():
            if len(order_depth.buy_orders) > 0 or len(order_depth.sell_orders) > 0:
                logger.print(f"Order depth for {product}:")
                if len(order_depth.buy_orders) > 0:
                    logger.print(f"  Buy orders: {order_depth.buy_orders}")
                if len(order_depth.sell_orders) > 0:
                    logger.print(f"  Sell orders: {order_depth.sell_orders}")
        
        # Log current positions
        logger.print(f"Current positions: {state.position}")
        
        spread_orders = self.spread_orders(
            state.order_depths,
            Product.PICNIC_BASKET1,
            basket_position,
            traderObject[Product.SPREAD],
        )
        
        if spread_orders != None:
            logger.print("Executing spread orders")
            result[Product.CROISSANTS] = spread_orders[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders[Product.JAMS]
            result[Product.DJEMBES] = spread_orders[Product.DJEMBES]
            result[Product.PICNIC_BASKET1] = spread_orders[Product.PICNIC_BASKET1]
        
        # Log final orders
        logger.print(f"Final orders: {result}")

        traderData = jsonpickle.encode(traderObject)
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData