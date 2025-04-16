from datamodel import OrderDepth, Order, TradingState
from typing import List, Dict

class Trader:

    def __init__(self):
        # Parameters for the bear spread strategy
        self.lower_strike = 9750
        self.upper_strike = 10250
        self.lower_product = f"VOLCANIC_ROCK_VOUCHER_{self.lower_strike}"
        self.upper_product = f"VOLCANIC_ROCK_VOUCHER_{self.upper_strike}"
        self.max_position = 50  # can be tuned depending on allowed limits

    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}

        # Current positions
        positions = state.position
        pos_low = positions.get(self.lower_product, 0)
        pos_high = positions.get(self.upper_product, 0)

        # Order books
        depth_low: OrderDepth = state.order_depths.get(self.lower_product)
        depth_high: OrderDepth = state.order_depths.get(self.upper_product)

        product_orders = []

        # Sell lower strike option (short call)
        if depth_low and len(depth_low.buy_orders) > 0:
            best_bid_low = max(depth_low.buy_orders.keys())
            bid_volume = depth_low.buy_orders[best_bid_low]
            size = min(bid_volume, self.max_position + pos_low)
            if size > 0:
                product_orders.append(Order(self.lower_product, best_bid_low, -size))

        if depth_low and len(depth_low.sell_orders) > 0:
            best_ask_low = min(depth_low.sell_orders.keys())
            ask_volume = depth_low.sell_orders[best_ask_low]
            size = min(-ask_volume, self.max_position - pos_low)
            if size > 0:
                product_orders.append(Order(self.lower_product, best_ask_low, size))

        orders[self.lower_product] = product_orders
        product_orders = []

        # Buy higher strike option (long call)
        if depth_high and len(depth_high.sell_orders) > 0:
            best_ask_high = min(depth_high.sell_orders.keys())
            ask_volume = depth_high.sell_orders[best_ask_high]
            size = min(-ask_volume, self.max_position - pos_high)
            if size > 0:
                product_orders.append(Order(self.upper_product, best_ask_high, size))

        if depth_high and len(depth_high.buy_orders) > 0:
            best_bid_high = max(depth_high.buy_orders.keys())
            bid_volume = depth_high.buy_orders[best_bid_high]
            size = min(bid_volume, self.max_position + pos_high)
            if size > 0:
                product_orders.append(Order(self.upper_product, best_bid_high, -size))

        orders[self.upper_product] = product_orders

        traderData = "BEAR_SPREAD_9750_10250"
        conversions = 0  # no conversions used
        return orders, conversions, traderData