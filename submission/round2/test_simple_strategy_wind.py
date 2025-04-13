from datamodel import OrderDepth, TradingState, Order, Listing, Observation, Symbol, Trade, ProsperityEncoder
from typing import List, Dict, Any
import numpy as np
import jsonpickle
import json

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
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    SQUID_INK = "SQUID_INK"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    DJEMBES = "DJEMBES"
    KELP = "KELP"

# Position limits for each product
POSITION_LIMITS = {
    Product.PICNIC_BASKET1: 60,
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.SQUID_INK: 300,
    Product.PICNIC_BASKET2: 70,
    Product.RAINFOREST_RESIN: 300,
    Product.DJEMBES: 60,
    Product.KELP: 300,
}

# Basket composition
BASKET1_COMPONENTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.SQUID_INK: 1,
}

BASKET2_COMPONENTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}

class Trader:
    def __init__(self):
        self.position_limits = POSITION_LIMITS
        self.product_prices = {}  # Store historical prices for each product
        self.ema_short = {}  # Short-term exponential moving average
        self.ema_long = {}  # Long-term exponential moving average
        
        # EMA parameters
        self.alpha_short = 0.2  # Higher alpha means more weight to recent prices
        self.alpha_long = 0.05
        
        # Market making parameters
        self.spread_multiplier = 1.5  # Wider spreads for more volatile products
        self.min_spread = 2  # Minimum spread to maintain
        
        # Initialize price history for all products
        for product in POSITION_LIMITS.keys():
            self.product_prices[product] = []
            self.ema_short[product] = None
            self.ema_long[product] = None

    def get_fair_value(self, product: str, order_depth: OrderDepth) -> float:
        """Calculate fair value based on order book and historical prices"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            # If either side of the book is empty, use historical data if available
            if self.ema_short[product] is not None:
                return self.ema_short[product]
            return None
        
        # Calculate mid price from order book
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # Update price history and EMAs
        self.product_prices[product].append(mid_price)
        if len(self.product_prices[product]) > 1000:  # Limit history size
            self.product_prices[product].pop(0)
        
        # Update EMAs
        if self.ema_short[product] is None:
            self.ema_short[product] = mid_price
            self.ema_long[product] = mid_price
        else:
            self.ema_short[product] = (self.alpha_short * mid_price) + ((1 - self.alpha_short) * self.ema_short[product])
            self.ema_long[product] = (self.alpha_long * mid_price) + ((1 - self.alpha_long) * self.ema_long[product])
        
        # Calculate fair value as a weighted average of mid price and EMAs
        fair_value = 0.7 * mid_price + 0.2 * self.ema_short[product] + 0.1 * self.ema_long[product]
        
        # Log fair value calculation
        logger.print(f"Product: {product}")
        logger.print(f"Best Bid: {best_bid}, Best Ask: {best_ask}")
        logger.print(f"Mid Price: {mid_price}")
        logger.print(f"EMA Short: {self.ema_short[product]}, EMA Long: {self.ema_long[product]}")
        logger.print(f"Calculated Fair Value: {fair_value}")
        
        return fair_value

    def calculate_spread(self, product: str) -> int:
        """Calculate dynamic spread based on price volatility"""
        if len(self.product_prices[product]) < 10:
            return self.min_spread
        
        # Calculate volatility as standard deviation of recent prices
        recent_prices = self.product_prices[product][-20:]
        volatility = np.std(recent_prices) if len(recent_prices) > 1 else 1
        
        # Higher volatility = wider spread
        spread = max(self.min_spread, int(volatility * self.spread_multiplier))
        
        # Log spread calculation
        logger.print(f"Product: {product}")
        logger.print(f"Volatility: {volatility}")
        logger.print(f"Calculated Spread: {spread}")
        
        return spread

    def take_profitable_orders(self, product: str, fair_value: float, orders: List[Order], 
                               order_depth: OrderDepth, position: int) -> List[Order]:
        """Take orders that are priced favorably compared to our fair value"""
        position_limit = self.position_limits[product]
        
        # Check for profitable buy opportunities (when ask price is below our fair value)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = abs(order_depth.sell_orders[best_ask])
            
            # If the best ask is below our fair value minus a threshold, buy
            if best_ask < fair_value - 1:
                # Calculate how much we can buy based on position limits
                buy_volume = min(best_ask_volume, position_limit - position)
                if buy_volume > 0:
                    orders.append(Order(product, best_ask, buy_volume))
                    logger.print(f"Taking profitable buy order for {product}")
                    logger.print(f"Price: {best_ask}, Volume: {buy_volume}")
        
        # Check for profitable sell opportunities (when bid price is above our fair value)
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            
            # If the best bid is above our fair value plus a threshold, sell
            if best_bid > fair_value + 1:
                # Calculate how much we can sell based on position limits
                sell_volume = min(best_bid_volume, position_limit + position)
                if sell_volume > 0:
                    orders.append(Order(product, best_bid, -sell_volume))
                    logger.print(f"Taking profitable sell order for {product}")
                    logger.print(f"Price: {best_bid}, Volume: {sell_volume}")
        
        return orders

    def market_make(self, product: str, fair_value: float, orders: List[Order], position: int) -> List[Order]:
        """Place market making orders around the fair value"""
        if fair_value is None:
            return orders
        
        # Calculate spread based on product volatility
        spread = self.calculate_spread(product)
        
        # Calculate bid and ask prices
        bid_price = int(fair_value - spread/2)
        ask_price = int(fair_value + spread/2)
        
        # Calculate order sizes based on current position
        position_limit = self.position_limits[product]
        
        # Adjust order sizes based on current position
        position_ratio = position / position_limit if position_limit > 0 else 0
        
        # Base size for orders
        base_size = position_limit // 5
        
        # Adjust buy size - smaller if we're long, larger if we're short
        buy_size = int(base_size * (1 - position_ratio))
        buy_size = max(1, min(buy_size, position_limit - position))
        
        # Adjust sell size - larger if we're long, smaller if we're short
        sell_size = int(base_size * (1 + position_ratio))
        sell_size = max(1, min(sell_size, position_limit + position))
        
        # Place orders
        if buy_size > 0:
            orders.append(Order(product, bid_price, buy_size))
            logger.print(f"Placing market making buy order for {product}")
            logger.print(f"Price: {bid_price}, Volume: {buy_size}")
        
        if sell_size > 0:
            orders.append(Order(product, ask_price, -sell_size))
            logger.print(f"Placing market making sell order for {product}")
            logger.print(f"Price: {ask_price}, Volume: {sell_size}")
        
        return orders

    def basket_arbitrage(self, state: TradingState, orders: Dict[str, List[Order]]) -> Dict[str, List[Order]]:
        """Look for arbitrage opportunities between baskets and their components"""
        
        # Check if all required products are in the order depths
        if (Product.PICNIC_BASKET1 not in state.order_depths or
            Product.CROISSANTS not in state.order_depths or
            Product.JAMS not in state.order_depths or
            Product.SQUID_INK not in state.order_depths):
            return orders
        
        # Get order depths
        basket_depth = state.order_depths[Product.PICNIC_BASKET1]
        croissants_depth = state.order_depths[Product.CROISSANTS]
        jams_depth = state.order_depths[Product.JAMS]
        squid_ink_depth = state.order_depths[Product.SQUID_INK]
        
        # Check if we have all necessary prices
        if (not basket_depth.buy_orders or not basket_depth.sell_orders or
            not croissants_depth.buy_orders or not croissants_depth.sell_orders or
            not jams_depth.buy_orders or not jams_depth.sell_orders or
            not squid_ink_depth.buy_orders or not squid_ink_depth.sell_orders):
            return orders
        
        # Get best prices
        basket_bid = max(basket_depth.buy_orders.keys())
        basket_ask = min(basket_depth.sell_orders.keys())
        
        croissants_bid = max(croissants_depth.buy_orders.keys())
        croissants_ask = min(croissants_depth.sell_orders.keys())
        
        jams_bid = max(jams_depth.buy_orders.keys())
        jams_ask = min(jams_depth.sell_orders.keys())
        
        squid_ink_bid = max(squid_ink_depth.buy_orders.keys())
        squid_ink_ask = min(squid_ink_depth.sell_orders.keys())
        
        # Calculate component costs
        components_buy_cost = (
            6 * croissants_ask +
            3 * jams_ask +
            1 * squid_ink_ask
        )
        
        components_sell_value = (
            6 * croissants_bid +
            3 * jams_bid +
            1 * squid_ink_bid
        )
        
        # Log arbitrage calculations
        logger.print("Basket Arbitrage Analysis:")
        logger.print(f"Basket Bid: {basket_bid}, Basket Ask: {basket_ask}")
        logger.print(f"Components Buy Cost: {components_buy_cost}")
        logger.print(f"Components Sell Value: {components_sell_value}")
        
        # Check for arbitrage opportunities
        # 1. Buy basket, sell components
        if basket_ask < components_sell_value:
            logger.print("Found arbitrage opportunity: Buy basket, sell components")
            # Calculate max trade size based on position limits and available volumes
            basket_volume = abs(basket_depth.sell_orders[basket_ask])
            croissants_volume = croissants_depth.buy_orders[croissants_bid] // 6
            jams_volume = jams_depth.buy_orders[jams_bid] // 3
            squid_ink_volume = squid_ink_depth.buy_orders[squid_ink_bid]
            
            # Get current positions
            basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
            croissants_position = state.position.get(Product.CROISSANTS, 0)
            jams_position = state.position.get(Product.JAMS, 0)
            squid_ink_position = state.position.get(Product.SQUID_INK, 0)
            
            # Calculate max trade size considering position limits
            max_basket_buy = min(
                basket_volume,
                (self.position_limits[Product.PICNIC_BASKET1] - basket_position),
                croissants_volume,
                jams_volume,
                squid_ink_volume,
                (self.position_limits[Product.CROISSANTS] + croissants_position) // 6,
                (self.position_limits[Product.JAMS] + jams_position) // 3,
                (self.position_limits[Product.SQUID_INK] + squid_ink_position)
            )
            
            if max_basket_buy > 0:
                logger.print(f"Executing arbitrage trade size: {max_basket_buy}")
                # Add orders to buy basket and sell components
                if Product.PICNIC_BASKET1 not in orders:
                    orders[Product.PICNIC_BASKET1] = []
                orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, basket_ask, max_basket_buy))
                
                if Product.CROISSANTS not in orders:
                    orders[Product.CROISSANTS] = []
                orders[Product.CROISSANTS].append(Order(Product.CROISSANTS, croissants_bid, -6 * max_basket_buy))
                
                if Product.JAMS not in orders:
                    orders[Product.JAMS] = []
                orders[Product.JAMS].append(Order(Product.JAMS, jams_bid, -3 * max_basket_buy))
                
                if Product.SQUID_INK not in orders:
                    orders[Product.SQUID_INK] = []
                orders[Product.SQUID_INK].append(Order(Product.SQUID_INK, squid_ink_bid, -1 * max_basket_buy))
        
        # 2. Buy components, sell basket
        elif basket_bid > components_buy_cost:
            logger.print("Found arbitrage opportunity: Buy components, sell basket")
            # Calculate max trade size based on position limits and available volumes
            basket_volume = basket_depth.buy_orders[basket_bid]
            croissants_volume = abs(croissants_depth.sell_orders[croissants_ask]) // 6
            jams_volume = abs(jams_depth.sell_orders[jams_ask]) // 3
            squid_ink_volume = abs(squid_ink_depth.sell_orders[squid_ink_ask])
            
            # Get current positions
            basket_position = state.position.get(Product.PICNIC_BASKET1, 0)
            croissants_position = state.position.get(Product.CROISSANTS, 0)
            jams_position = state.position.get(Product.JAMS, 0)
            squid_ink_position = state.position.get(Product.SQUID_INK, 0)
            
            # Calculate max trade size considering position limits
            max_basket_sell = min(
                basket_volume,
                (self.position_limits[Product.PICNIC_BASKET1] + basket_position),
                croissants_volume,
                jams_volume,
                squid_ink_volume,
                (self.position_limits[Product.CROISSANTS] - croissants_position) // 6,
                (self.position_limits[Product.JAMS] - jams_position) // 3,
                (self.position_limits[Product.SQUID_INK] - squid_ink_position)
            )
            
            if max_basket_sell > 0:
                logger.print(f"Executing arbitrage trade size: {max_basket_sell}")
                # Add orders to sell basket and buy components
                if Product.PICNIC_BASKET1 not in orders:
                    orders[Product.PICNIC_BASKET1] = []
                orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, basket_bid, -max_basket_sell))
                
                if Product.CROISSANTS not in orders:
                    orders[Product.CROISSANTS] = []
                orders[Product.CROISSANTS].append(Order(Product.CROISSANTS, croissants_ask, 6 * max_basket_sell))
                
                if Product.JAMS not in orders:
                    orders[Product.JAMS] = []
                orders[Product.JAMS].append(Order(Product.JAMS, jams_ask, 3 * max_basket_sell))
                
                if Product.SQUID_INK not in orders:
                    orders[Product.SQUID_INK] = []
                orders[Product.SQUID_INK].append(Order(Product.SQUID_INK, squid_ink_ask, 1 * max_basket_sell))
        
        return orders

    def run(self, state: TradingState):
        """
        Main method called by the game engine.
        
        Args:
            state: The current state of the game
            
        Returns:
            Dict[str, List[Order]]: Orders to be placed on the market
        """
        # Initialize the result dict
        result = {}
        
        # Load trader data if available
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        
        # Log current state
        logger.print(f"===== TIMESTAMP: {state.timestamp} =====")
        logger.print(f"Current positions: {state.position}")
        
        # Process each product
        logger.print(f"Available products: {list(state.order_depths.keys())}")
        for product in state.order_depths.keys():
            # Skip products we don't want to trade
            if product not in self.position_limits:
                continue
            
            # Get order depth and current position
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            
            # Log order depth information
            if len(order_depth.buy_orders) > 0 or len(order_depth.sell_orders) > 0:
                logger.print(f"Order depth for {product}:")
                if len(order_depth.buy_orders) > 0:
                    logger.print(f"  Buy orders: {order_depth.buy_orders}")
                if len(order_depth.sell_orders) > 0:
                    logger.print(f"  Sell orders: {order_depth.sell_orders}")
            
            # Initialize orders list for this product
            if product not in result:
                result[product] = []
            
            # Calculate fair value
            fair_value = self.get_fair_value(product, order_depth)
            if fair_value is None:
                continue
            
            # Take profitable orders first
            result[product] = self.take_profitable_orders(
                product, fair_value, result[product], order_depth, position
            )
            
            # Then place market making orders
            result[product] = self.market_make(
                product, fair_value, result[product], position
            )
        
        # Look for basket arbitrage opportunities
        result = self.basket_arbitrage(state, result)
        
        # Save trader data for next round
        trader_data_string = jsonpickle.encode(trader_data)
        
        # Log final orders
        logger.print(f"Final orders: {result}")
        
        # Flush logs
        logger.flush(state, result, 0, trader_data_string)
        
        return result, 0, trader_data_string 