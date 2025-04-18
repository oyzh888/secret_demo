import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# Aggressive Microstructure Parameters
PARAMS = {
    # Base parameters
    'position_limit': 75,  # Position limit
    'conversion_limit': 10,  # Conversion limit (unused)
    
    # Microstructure strategy parameters
    'depth_threshold': 3,       # Consider fewer levels for imbalance/pressure (more sensitive)
    'pressure_window': 5,       # Shorter window for pressure calculation
    'buy_pressure_threshold': 0.65, # Lower threshold to trigger sell signal
    'sell_pressure_threshold': 0.65,# Lower threshold to trigger buy signal
    'spread_threshold': 0.15,   # Slightly wider acceptable spread
    'imbalance_threshold': 0.4,   # Lower imbalance threshold to trigger trades
    'flow_ratio_threshold': 1.2, # Lower flow ratio threshold
    
    # Trading parameters
    'max_trade_quantity': 20,  # Moderate trade quantity
    'min_trade_interval': 2,  # Minimum ticks between trades (to limit frequency slightly)
}

class Trader:
    def __init__(self):
        # Initialize parameters
        self.position_limit = PARAMS['position_limit']
        
        # Microstructure parameters
        self.depth_threshold = PARAMS['depth_threshold']
        self.pressure_window = PARAMS['pressure_window']
        self.buy_pressure_threshold = PARAMS['buy_pressure_threshold']
        self.sell_pressure_threshold = PARAMS['sell_pressure_threshold']
        self.spread_threshold = PARAMS['spread_threshold']
        self.imbalance_threshold = PARAMS['imbalance_threshold']
        self.flow_ratio_threshold = PARAMS['flow_ratio_threshold']
        
        # Trading parameters
        self.max_trade_quantity = PARAMS['max_trade_quantity']
        self.min_trade_interval = PARAMS['min_trade_interval']

        # Historical data for microstructure
        self.buy_pressure_history = []
        self.sell_pressure_history = []
        self.imbalance_history = []
        self.last_trade_timestamp = -self.min_trade_interval # Allow trading from the start
        self.timestamp = 0

    def calculate_market_pressure(self, order_depth: OrderDepth) -> tuple[float, float]:
        """Calculate market buy/sell pressure based on weighted order book."""
        buy_pressure = 0
        sell_pressure = 0
        total_buy_vol = 0
        total_sell_vol = 0

        # Calculate buy pressure (sum of price * quantity for top bids)
        if order_depth.buy_orders:
            sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
            for i, (price, quantity) in enumerate(sorted_bids):
                 if i >= self.depth_threshold:
                     break
                 buy_pressure += price * abs(quantity)
                 total_buy_vol += abs(quantity)

        # Calculate sell pressure (sum of price * quantity for top asks)
        if order_depth.sell_orders:
            sorted_asks = sorted(order_depth.sell_orders.items())
            for i, (price, quantity) in enumerate(sorted_asks):
                 if i >= self.depth_threshold:
                     break
                 sell_pressure += price * abs(quantity)
                 total_sell_vol += abs(quantity)

        # Normalize pressures
        total_value = buy_pressure + sell_pressure
        norm_buy_pressure = buy_pressure / total_value if total_value > 0 else 0.5
        norm_sell_pressure = sell_pressure / total_value if total_value > 0 else 0.5
        
        # Append to history (use raw pressure for history maybe? let's stick to normalized for now)
        self.buy_pressure_history.append(norm_buy_pressure)
        self.sell_pressure_history.append(norm_sell_pressure)
        if len(self.buy_pressure_history) > self.pressure_window:
             self.buy_pressure_history.pop(0)
             self.sell_pressure_history.pop(0)
             
        # Return the average pressure over the window
        avg_buy_pressure = np.mean(self.buy_pressure_history) if self.buy_pressure_history else 0.5
        avg_sell_pressure = np.mean(self.sell_pressure_history) if self.sell_pressure_history else 0.5

        return avg_buy_pressure, avg_sell_pressure

    def analyze_order_book_imbalance(self, order_depth: OrderDepth) -> float:
        """Analyze order book imbalance based on volume at top levels."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        buy_volume = 0
        sell_volume = 0
        
        # Sum volume for top buy orders
        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True)[:self.depth_threshold]:
            buy_volume += abs(quantity)
            
        # Sum volume for top sell orders
        for price, quantity in sorted(order_depth.sell_orders.items())[:self.depth_threshold]:
            sell_volume += abs(quantity)
        
        # Calculate imbalance ratio
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0
            
        imbalance = (buy_volume - sell_volume) / total_volume
        
        self.imbalance_history.append(imbalance)
        if len(self.imbalance_history) > self.pressure_window: # Reuse window
             self.imbalance_history.pop(0)
             
        # Return average imbalance over the window
        avg_imbalance = np.mean(self.imbalance_history) if self.imbalance_history else 0
        return avg_imbalance
        
    def analyze_order_flow(self, order_depth: OrderDepth) -> tuple[float, float]:
        """Analyze order flow based on total value at top levels."""
        buy_flow = 0
        sell_flow = 0
        
        if order_depth.buy_orders:
            for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True)[:self.depth_threshold]:
                buy_flow += price * abs(quantity)
                
        if order_depth.sell_orders:
            for price, quantity in sorted(order_depth.sell_orders.items())[:self.depth_threshold]:
                sell_flow += price * abs(quantity)
                
        return buy_flow, sell_flow

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
            return result, conversions, traderData
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        spread = best_ask - best_bid
        relative_spread = spread / best_bid if best_bid > 0 else 0
        mid_price = (best_bid + best_ask) / 2

        # Calculate microstructure indicators
        avg_buy_pressure, avg_sell_pressure = self.calculate_market_pressure(order_depth)
        avg_imbalance = self.analyze_order_book_imbalance(order_depth)
        buy_flow, sell_flow = self.analyze_order_flow(order_depth)
        
        orders = []
        
        # Check if enough time has passed since the last trade
        if self.timestamp < self.last_trade_timestamp + self.min_trade_interval:
             return result, conversions, traderData # Skip trading this tick

        # --- Aggressive Trading Logic Based on Microstructure ---
        available_buy = self.position_limit - current_position
        available_sell = self.position_limit + current_position

        # BUY Signal: Strong sell pressure, negative imbalance, high sell flow
        buy_signal = (avg_sell_pressure > self.sell_pressure_threshold and 
                      relative_spread < self.spread_threshold and 
                      avg_imbalance < -self.imbalance_threshold and 
                      sell_flow > buy_flow * self.flow_ratio_threshold)
                      
        # SELL Signal: Strong buy pressure, positive imbalance, high buy flow
        sell_signal = (avg_buy_pressure > self.buy_pressure_threshold and 
                       relative_spread < self.spread_threshold and 
                       avg_imbalance > self.imbalance_threshold and 
                       buy_flow > sell_flow * self.flow_ratio_threshold)

        if buy_signal:
            if available_buy > 0:
                buy_quantity = min(available_buy, self.max_trade_quantity)
                buy_price = best_ask # Take the ask aggressively
                print(f"MICRO BUY SIGNAL: Press={avg_sell_pressure:.2f}, Imb={avg_imbalance:.2f}. Placing BUY order for {buy_quantity} at {buy_price}")
                orders.append(Order(product, buy_price, buy_quantity))
                self.last_trade_timestamp = self.timestamp

        elif sell_signal:
             if available_sell > 0:
                sell_quantity = min(available_sell, self.max_trade_quantity)
                sell_price = best_bid # Hit the bid aggressively
                print(f"MICRO SELL SIGNAL: Press={avg_buy_pressure:.2f}, Imb={avg_imbalance:.2f}. Placing SELL order for {sell_quantity} at {sell_price}")
                orders.append(Order(product, sell_price, -sell_quantity))
                self.last_trade_timestamp = self.timestamp
                
        # Simple Mean Reversion / Position Closing Logic: Close if spread widens or imbalance neutralizes
        # More aggressive: Only close if signals reverse strongly?
        # Let's keep it simple: close if signals disappear or spread widens significantly
        elif not buy_signal and not sell_signal: 
            if current_position > 0 and (relative_spread > self.spread_threshold * 1.5 or abs(avg_imbalance) < 0.1): 
                 sell_quantity = min(current_position, self.max_trade_quantity)
                 sell_price = best_bid
                 print(f"MICRO CLOSE LONG: Spread={relative_spread:.3f} or Imb={avg_imbalance:.2f}. Closing {sell_quantity} at {sell_price}")
                 orders.append(Order(product, sell_price, -sell_quantity))
                 self.last_trade_timestamp = self.timestamp
            elif current_position < 0 and (relative_spread > self.spread_threshold * 1.5 or abs(avg_imbalance) < 0.1):
                 buy_quantity = min(abs(current_position), self.max_trade_quantity)
                 buy_price = best_ask
                 print(f"MICRO CLOSE SHORT: Spread={relative_spread:.3f} or Imb={avg_imbalance:.2f}. Closing {buy_quantity} at {buy_price}")
                 orders.append(Order(product, buy_price, buy_quantity))
                 self.last_trade_timestamp = self.timestamp

        result[product] = orders
        return result, conversions, traderData 