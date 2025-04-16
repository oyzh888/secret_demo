from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle

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
        self.price_history = []
        self.position_history = {}
        self.profit_threshold = 0.3  # 30%利润阈值
        self.close_time_threshold = 0.8  # 80%时间后开始平仓
        self.max_rounds = 7  # 总轮数

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        return None

    def calculate_spread_profit(self, lower_strike: int, upper_strike: int, state: TradingState) -> tuple[float, float]:
        """Calculate spread profit potential"""
        lower_voucher = f"VOLCANIC_ROCK_VOUCHER_{lower_strike}"
        upper_voucher = f"VOLCANIC_ROCK_VOUCHER_{upper_strike}"
        
        if lower_voucher not in state.order_depths or upper_voucher not in state.order_depths:
            return 0.0, 0.0
            
        lower_voucher_depth = state.order_depths[lower_voucher]
        upper_voucher_depth = state.order_depths[upper_voucher]
        
        if not lower_voucher_depth.sell_orders or not upper_voucher_depth.buy_orders:
            return 0.0, 0.0
            
        lower_ask = min(lower_voucher_depth.sell_orders.keys())
        upper_bid = max(upper_voucher_depth.buy_orders.keys())
        
        # Bull spread: Buy lower strike, sell upper strike
        bull_spread_cost = lower_ask - upper_bid
        bull_spread_profit = upper_strike - lower_strike - bull_spread_cost
        
        if not lower_voucher_depth.buy_orders or not upper_voucher_depth.sell_orders:
            return bull_spread_profit, 0.0
            
        lower_bid = max(lower_voucher_depth.buy_orders.keys())
        upper_ask = min(upper_voucher_depth.sell_orders.keys())
        
        # Bear spread: Sell lower strike, buy upper strike
        bear_spread_cost = upper_ask - lower_bid
        bear_spread_profit = upper_strike - lower_strike - bear_spread_cost
        
        return bull_spread_profit, bear_spread_profit

    def should_take_profit(self, state: TradingState, current_positions: Dict[str, int]) -> bool:
        """Determine if we should take profit based on various conditions"""
        # 检查是否接近收盘
        if self.round / self.max_rounds >= self.close_time_threshold:
            return True
            
        # 检查价格是否达到局部高点
        if len(self.price_history) >= 3:
            current_price = self.get_mid_price(state.order_depths["VOLCANIC_ROCK"])
            if current_price is not None:
                if current_price > max(self.price_history[-3:]):
                    return True
                    
        # 检查是否有足够大的利润
        total_profit = 0
        for symbol, position in current_positions.items():
            if position != 0 and symbol in state.order_depths:
                mid_price = self.get_mid_price(state.order_depths[symbol])
                if mid_price is not None:
                    # 计算当前持仓的利润
                    if symbol in self.position_history:
                        entry_price = self.position_history[symbol]["price"]
                        profit = (mid_price - entry_price) * position
                        total_profit += profit
                        
        if total_profit > self.current_capital * self.profit_threshold:
            return True
            
        return False

    def execute_profit_taking(self, state: TradingState, current_positions: Dict[str, int]) -> Dict[str, List[Order]]:
        """Execute profit taking orders"""
        result = {}
        
        for symbol, position in current_positions.items():
            if position != 0 and symbol in state.order_depths:
                # 平仓方向与持仓相反
                order_size = -position
                
                # 获取最优价格
                if order_size > 0:  # 买入平仓
                    best_price = min(state.order_depths[symbol].sell_orders.keys())
                else:  # 卖出平仓
                    best_price = max(state.order_depths[symbol].buy_orders.keys())
                    
                if symbol not in result:
                    result[symbol] = []
                result[symbol].append(Order(symbol, best_price, order_size))
                print(f"PROFIT TAKING: {symbol} at {best_price} with size {order_size}")
                
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
            
        # Update price history
        self.price_history.append(volcanic_rock_mid)
        if len(self.price_history) > 10:
            self.price_history.pop(0)
            
        # Check if we should take profit
        if self.should_take_profit(state, current_positions):
            profit_orders = self.execute_profit_taking(state, current_positions)
            for symbol, orders in profit_orders.items():
                if symbol not in result:
                    result[symbol] = []
                result[symbol].extend(orders)
            return result, conversions, jsonpickle.encode(traderObject)
            
        # Calculate time to expiration (7 days - current round)
        tte = max(7 - self.round, 0.1)
        
        # Evaluate spreads aggressively
        strikes = [9500, 9750, 10000, 10250, 10500]
        for i in range(len(strikes) - 1):
            lower_strike = strikes[i]
            upper_strike = strikes[i + 1]
            
            bull_profit, bear_profit = self.calculate_spread_profit(lower_strike, upper_strike, state)
            
            # Check if we have enough capital and position limits
            lower_voucher = f"VOLCANIC_ROCK_VOUCHER_{lower_strike}"
            upper_voucher = f"VOLCANIC_ROCK_VOUCHER_{upper_strike}"
            
            position_limit = min(
                self.position_limits[lower_voucher] - current_positions.get(lower_voucher, 0),
                self.position_limits[upper_voucher] - current_positions.get(upper_voucher, 0)
            )
            
            # More aggressive trading - lower profit threshold
            if bull_profit > 0.5 and position_limit > 0:
                # Implement bull spread
                if lower_voucher not in result:
                    result[lower_voucher] = []
                if upper_voucher not in result:
                    result[upper_voucher] = []
                    
                # Buy lower strike, sell upper strike
                result[lower_voucher].append(Order(lower_voucher, min(state.order_depths[lower_voucher].sell_orders.keys()), position_limit))
                result[upper_voucher].append(Order(upper_voucher, max(state.order_depths[upper_voucher].buy_orders.keys()), -position_limit))
                print(f"BULL SPREAD: Buy {lower_voucher}, Sell {upper_voucher}, Profit: {bull_profit:.2f}")
                
                # Record position history
                self.position_history[lower_voucher] = {
                    "price": min(state.order_depths[lower_voucher].sell_orders.keys()),
                    "size": position_limit
                }
                self.position_history[upper_voucher] = {
                    "price": max(state.order_depths[upper_voucher].buy_orders.keys()),
                    "size": -position_limit
                }
                
            elif bear_profit > 0.5 and position_limit > 0:
                # Implement bear spread
                if lower_voucher not in result:
                    result[lower_voucher] = []
                if upper_voucher not in result:
                    result[upper_voucher] = []
                    
                # Sell lower strike, buy upper strike
                result[lower_voucher].append(Order(lower_voucher, max(state.order_depths[lower_voucher].buy_orders.keys()), -position_limit))
                result[upper_voucher].append(Order(upper_voucher, min(state.order_depths[upper_voucher].sell_orders.keys()), position_limit))
                print(f"BEAR SPREAD: Sell {lower_voucher}, Buy {upper_voucher}, Profit: {bear_profit:.2f}")
                
                # Record position history
                self.position_history[lower_voucher] = {
                    "price": max(state.order_depths[lower_voucher].buy_orders.keys()),
                    "size": -position_limit
                }
                self.position_history[upper_voucher] = {
                    "price": min(state.order_depths[upper_voucher].sell_orders.keys()),
                    "size": position_limit
                }

        # Increment round counter
        self.round += 1

        return result, conversions, jsonpickle.encode(traderObject) 