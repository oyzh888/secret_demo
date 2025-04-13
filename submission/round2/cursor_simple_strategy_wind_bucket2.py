from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any
import numpy as np
import jsonpickle

class Product:
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    SQUID_INK = "SQUID_INK"
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    DJEMBES = "DJEMBES"
    KELP = "KELP"

# Position limits for each product
POSITION_LIMITS = {
    Product.PICNIC_BASKET1: 60,
    Product.PICNIC_BASKET2: 70,
    Product.CROISSANTS: 250,
    Product.JAMS: 350,
    Product.SQUID_INK: 300,
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

# Correlation coefficients
CORRELATIONS = {
    (Product.CROISSANTS, Product.PICNIC_BASKET1): 0.83,
    (Product.CROISSANTS, Product.PICNIC_BASKET2): 0.82,
    (Product.JAMS, Product.PICNIC_BASKET1): 0.10,
    (Product.JAMS, Product.PICNIC_BASKET2): 0.36,
}

class Trader:
    def __init__(self):
        self.position_limits = POSITION_LIMITS
        self.product_prices = {}  # Store historical prices for each product
        self.ema_short = {}  # Short-term exponential moving average
        self.ema_long = {}  # Long-term exponential moving average
        
        # EMA parameters
        self.alpha_short = 0.2
        self.alpha_long = 0.05
        
        # Market making parameters
        self.spread_multiplier = 1.5
        self.min_spread = 2
        
        # Correlation-based trading parameters
        self.correlation_threshold = 0.7  # Minimum correlation to consider for spread trading
        self.spread_threshold = 0.02  # Minimum spread to consider for arbitrage
        
        # Initialize price history for all products
        for product in POSITION_LIMITS.keys():
            self.product_prices[product] = []
            self.ema_short[product] = None
            self.ema_long[product] = None

    def get_fair_value(self, product: str, order_depth: OrderDepth) -> float:
        """Calculate fair value based on order book and historical prices"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            if self.ema_short[product] is not None:
                return self.ema_short[product]
            return None
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        self.product_prices[product].append(mid_price)
        if len(self.product_prices[product]) > 1000:
            self.product_prices[product].pop(0)
        
        if self.ema_short[product] is None:
            self.ema_short[product] = mid_price
            self.ema_long[product] = mid_price
        else:
            self.ema_short[product] = (self.alpha_short * mid_price) + ((1 - self.alpha_short) * self.ema_short[product])
            self.ema_long[product] = (self.alpha_long * mid_price) + ((1 - self.alpha_long) * self.ema_long[product])
        
        fair_value = 0.7 * mid_price + 0.2 * self.ema_short[product] + 0.1 * self.ema_long[product]
        return fair_value

    def calculate_spread(self, product: str) -> int:
        """Calculate dynamic spread based on price volatility"""
        if len(self.product_prices[product]) < 10:
            return self.min_spread
        
        recent_prices = self.product_prices[product][-20:]
        volatility = np.std(recent_prices) if len(recent_prices) > 1 else 1
        
        spread = max(self.min_spread, int(volatility * self.spread_multiplier))
        return spread

    def correlation_arbitrage(self, state: TradingState, orders: Dict[str, List[Order]]) -> Dict[str, List[Order]]:
        """Implement correlation-based arbitrage between CROISSANTS and both baskets"""
        
        # Check if we have all necessary order depths
        required_products = [Product.CROISSANTS, Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]
        if not all(product in state.order_depths for product in required_products):
            return orders
        
        # Get order depths
        croissants_depth = state.order_depths[Product.CROISSANTS]
        basket1_depth = state.order_depths[Product.PICNIC_BASKET1]
        basket2_depth = state.order_depths[Product.PICNIC_BASKET2]
        
        # Check if we have prices on both sides
        if not all(depth.buy_orders and depth.sell_orders for depth in [croissants_depth, basket1_depth, basket2_depth]):
            return orders
        
        # Get best prices
        croissants_bid = max(croissants_depth.buy_orders.keys())
        croissants_ask = min(croissants_depth.sell_orders.keys())
        
        basket1_bid = max(basket1_depth.buy_orders.keys())
        basket1_ask = min(basket1_depth.sell_orders.keys())
        
        basket2_bid = max(basket2_depth.buy_orders.keys())
        basket2_ask = min(basket2_depth.sell_orders.keys())
        
        # Calculate spreads
        basket1_spread = (basket1_ask - basket1_bid) / basket1_bid
        basket2_spread = (basket2_ask - basket2_bid) / basket2_bid
        
        # Get current positions
        croissants_position = state.position.get(Product.CROISSANTS, 0)
        basket1_position = state.position.get(Product.PICNIC_BASKET1, 0)
        basket2_position = state.position.get(Product.PICNIC_BASKET2, 0)
        
        # Calculate fair value ratios based on correlations
        basket1_ratio = CORRELATIONS[(Product.CROISSANTS, Product.PICNIC_BASKET1)]
        basket2_ratio = CORRELATIONS[(Product.CROISSANTS, Product.PICNIC_BASKET2)]
        
        # Calculate expected basket prices based on croissants price
        expected_basket1_price = croissants_bid * 6 * basket1_ratio
        expected_basket2_price = croissants_bid * 4 * basket2_ratio
        
        # Check for arbitrage opportunities
        if basket1_spread > self.spread_threshold and basket2_spread > self.spread_threshold:
            # If basket1 is relatively expensive compared to basket2
            if basket1_ask / expected_basket1_price > basket2_ask / expected_basket2_price:
                # Sell basket1, buy basket2
                max_basket1_sell = min(
                    abs(basket1_depth.sell_orders[basket1_ask]),
                    self.position_limits[Product.PICNIC_BASKET1] + basket1_position
                )
                max_basket2_buy = min(
                    basket2_depth.buy_orders[basket2_bid],
                    self.position_limits[Product.PICNIC_BASKET2] - basket2_position
                )
                
                trade_size = min(max_basket1_sell, max_basket2_buy)
                
                if trade_size > 0:
                    if Product.PICNIC_BASKET1 not in orders:
                        orders[Product.PICNIC_BASKET1] = []
                    orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, basket1_ask, -trade_size))
                    
                    if Product.PICNIC_BASKET2 not in orders:
                        orders[Product.PICNIC_BASKET2] = []
                    orders[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, basket2_bid, trade_size))
            
            # If basket2 is relatively expensive compared to basket1
            elif basket2_ask / expected_basket2_price > basket1_ask / expected_basket1_price:
                # Sell basket2, buy basket1
                max_basket2_sell = min(
                    abs(basket2_depth.sell_orders[basket2_ask]),
                    self.position_limits[Product.PICNIC_BASKET2] + basket2_position
                )
                max_basket1_buy = min(
                    basket1_depth.buy_orders[basket1_bid],
                    self.position_limits[Product.PICNIC_BASKET1] - basket1_position
                )
                
                trade_size = min(max_basket2_sell, max_basket1_buy)
                
                if trade_size > 0:
                    if Product.PICNIC_BASKET2 not in orders:
                        orders[Product.PICNIC_BASKET2] = []
                    orders[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, basket2_ask, -trade_size))
                    
                    if Product.PICNIC_BASKET1 not in orders:
                        orders[Product.PICNIC_BASKET1] = []
                    orders[Product.PICNIC_BASKET1].append(Order(Product.PICNIC_BASKET1, basket1_bid, trade_size))
        
        return orders

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
        # If we're long, place larger sell orders and smaller buy orders
        # If we're short, place larger buy orders and smaller sell orders
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
        
        if sell_size > 0:
            orders.append(Order(product, ask_price, -sell_size))
        
        return orders

    def run(self, state: TradingState):
        """
        Main method called by the game engine.
        """
        result = {}
        
        # Load trader data if available
        trader_data = {}
        if state.traderData and state.traderData != "":
            trader_data = jsonpickle.decode(state.traderData)
        
        # Process each product
        for product in state.order_depths.keys():
            if product not in self.position_limits:
                continue
            
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            
            if product not in result:
                result[product] = []
            
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
        
        # Look for correlation-based arbitrage opportunities
        result = self.correlation_arbitrage(state, result)
        
        # Save trader data for next round
        trader_data_string = jsonpickle.encode(trader_data)
        
        return result, 0, trader_data_string 