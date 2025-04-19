import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# Aggressive Trend Following Parameters
PARAMS = {
    # Base parameters
    'position_limit': 75,  # Position limit
    'conversion_limit': 10,  # Conversion limit (unused)
    
    # Trend following parameters
    'ma_short_window': 5,   # Very short-term moving average
    'ma_long_window': 15,  # Short-term moving average
    'crossover_threshold': 0.1, # Minimal difference for crossover signal
    
    # Trading parameters
    'max_trade_quantity': 40,  # Very aggressive trade quantity
    'max_spread_ratio': 0.005, # Max relative spread to allow trading (0.5%)
}

class Trader:
    def __init__(self):
        # Initialize parameters
        self.position_limit = PARAMS['position_limit']
        
        # Trend parameters
        self.ma_short_window = PARAMS['ma_short_window']
        self.ma_long_window = PARAMS['ma_long_window']
        self.crossover_threshold = PARAMS['crossover_threshold']
        
        # Trading parameters
        self.max_trade_quantity = PARAMS['max_trade_quantity']
        self.max_spread_ratio = PARAMS['max_spread_ratio']

        # Historical data
        self.price_history = [] # Stores mid-prices
        self.ma_short_history = []
        self.ma_long_history = []
        
        # State
        self.timestamp = 0

    def calculate_ma(self, prices: List[float], window: int) -> float:
        """Calculate the moving average for a given window."""
        if len(prices) < window:
            return None # Not enough data
        return np.mean(prices[-window:])

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        self.timestamp = state.timestamp
        result = {}
        conversions = 0
        traderData = ""
        product = "MAGNIFICENT_MACARONS"

        if product not in state.order_depths:
            return result, conversions, traderData
            
        order_depth = state.order_depths[product]
        current_position = state.position.get(product, 0)
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            # Need market prices to calculate MAs and trade
            return result, conversions, traderData
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        relative_spread = spread / mid_price if mid_price > 0 else 0

        # Update price history
        self.price_history.append(mid_price)
        if len(self.price_history) > self.ma_long_window + 2: # Keep history manageable
            self.price_history.pop(0)

        # Calculate MAs
        ma_short = self.calculate_ma(self.price_history, self.ma_short_window)
        ma_long = self.calculate_ma(self.price_history, self.ma_long_window)
        
        # Store MA history
        if ma_short is not None: self.ma_short_history.append(ma_short)
        if ma_long is not None: self.ma_long_history.append(ma_long)
        if len(self.ma_short_history) > 2: self.ma_short_history.pop(0)
        if len(self.ma_long_history) > 2: self.ma_long_history.pop(0)
        
        orders = []
        
        # Check conditions for trading
        if ma_short is None or ma_long is None or len(self.ma_short_history) < 2 or len(self.ma_long_history) < 2:
            # Not enough data for crossover signal
            return result, conversions, traderData
            
        # Check if spread is too wide
        if relative_spread > self.max_spread_ratio:
            print(f"SPREAD TOO WIDE ({relative_spread:.4f} > {self.max_spread_ratio}). Holding orders.")
            # If spread is wide, maybe close positions?
            # For now, just don't open new ones aggressively.
            if current_position > 0:
                # Consider closing long if trend reverses? (MA short crosses below long)
                if self.ma_short_history[-1] < self.ma_long_history[-1] and self.ma_short_history[-2] >= self.ma_long_history[-2]:
                    orders.append(Order(product, best_bid, -current_position)) # Close entire position
                    print(f"TREND CLOSE LONG due to wide spread and MA cross down.")
            elif current_position < 0:
                # Consider closing short if trend reverses? (MA short crosses above long)
                if self.ma_short_history[-1] > self.ma_long_history[-1] and self.ma_short_history[-2] <= self.ma_long_history[-2]:
                    orders.append(Order(product, best_ask, -current_position)) # Close entire position
                    print(f"TREND CLOSE SHORT due to wide spread and MA cross up.")
            result[product] = orders
            return result, conversions, traderData

        # --- Aggressive Trend Following Logic ---
        available_buy = self.position_limit - current_position
        available_sell = self.position_limit + current_position
        
        # Check for Bullish Crossover (Short MA crosses above Long MA)
        crossed_up = (self.ma_short_history[-1] > self.ma_long_history[-1] + self.crossover_threshold and 
                    self.ma_short_history[-2] <= self.ma_long_history[-2])
        
        # Check for Bearish Crossover (Short MA crosses below Long MA)
        crossed_down = (self.ma_short_history[-1] < self.ma_long_history[-1] - self.crossover_threshold and 
                      self.ma_short_history[-2] >= self.ma_long_history[-2])

        if crossed_up:
            # Close any existing short position first
            if current_position < 0:
                print(f"TREND REVERSAL: Closing short position {-current_position} due to bullish crossover.")
                orders.append(Order(product, best_ask, -current_position))
                available_buy += abs(current_position) # Update available capital
                current_position = 0 # Reset position after closing
                
            # Open aggressive long position
            if available_buy > 0:
                buy_quantity = min(available_buy, self.max_trade_quantity)
                buy_price = best_ask # Aggressively take the ask
                print(f"TREND BUY SIGNAL: MA{self.ma_short_window} crossed above MA{self.ma_long_window}. Placing BUY order for {buy_quantity} at {buy_price}")
                orders.append(Order(product, buy_price, buy_quantity))

        elif crossed_down:
            # Close any existing long position first
            if current_position > 0:
                print(f"TREND REVERSAL: Closing long position {current_position} due to bearish crossover.")
                orders.append(Order(product, best_bid, -current_position))
                available_sell += current_position # Update available capital
                current_position = 0 # Reset position after closing
                
            # Open aggressive short position
            if available_sell > 0:
                sell_quantity = min(available_sell, self.max_trade_quantity)
                sell_price = best_bid # Aggressively hit the bid
                print(f"TREND SELL SIGNAL: MA{self.ma_short_window} crossed below MA{self.ma_long_window}. Placing SELL order for {sell_quantity} at {sell_price}")
                orders.append(Order(product, sell_price, -sell_quantity))
                
        # Optional: Add logic to hold position if trend continues (MA short still above/below MA long)?
        # Current logic only trades on the crossover event.

        result[product] = orders
        return result, conversions, traderData 