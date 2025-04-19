import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# Aggressive Fundamentals Parameters
PARAMS = {
    # Base parameters
    'position_limit': 75,  # Position limit
    'conversion_limit': 10,  # Conversion limit (unused in this strategy)
    
    # Fundamental strategy parameters
    'sugar_weight': 0.45,  # Sugar price weight (increased)
    'sunlight_weight': 0.45,  # Sunlight index weight (increased)
    'tariff_weight': 0.1,  # Tariff weight (decreased)
    'buy_threshold': 0.55,   # More aggressive buy signal threshold (normalized score)
    'sell_threshold': 0.45,  # More aggressive sell signal threshold (normalized score)
    'neutral_threshold': 0.05, # Narrower band around neutral for closing positions
    
    # Trading parameters
    'max_trade_quantity': 30,  # Increased max trade quantity
    'history_window': 50,    # Window for normalization
}

class Trader:
    def __init__(self):
        # Initialize parameters
        self.position_limit = PARAMS['position_limit']
        
        # Fundamental strategy parameters
        self.sugar_weight = PARAMS['sugar_weight']
        self.sunlight_weight = PARAMS['sunlight_weight']
        self.tariff_weight = PARAMS['tariff_weight']
        self.buy_threshold = PARAMS['buy_threshold']
        self.sell_threshold = PARAMS['sell_threshold']
        self.neutral_threshold = PARAMS['neutral_threshold']
        
        # Trading parameters
        self.max_trade_quantity = PARAMS['max_trade_quantity']
        self.history_window = PARAMS['history_window']

        # Historical data for normalization
        self.sugar_history = []
        self.sunlight_history = []
        self.import_tariff_history = []
        
    def normalize_value(self, value: float, history: List[float]) -> float:
        """Normalize value based on recent history (0 to 1 scale)."""
        # Ensure history has enough data for meaningful normalization
        hist = history[-self.history_window:]
        if len(hist) < 2:
            return 0.5 # Return neutral if not enough data
            
        min_val = min(hist)
        max_val = max(hist)
        
        # Avoid division by zero if min and max are the same
        if max_val == min_val:
            return 0.5 # Return neutral if values are constant
            
        # Normalize the value
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized)) # Clamp between 0 and 1
    
    def calculate_price_score(self, state: TradingState) -> float:
        """Calculate a score based on fundamental observations."""
        # Get observations, default to 0 if missing
        sugar_price = getattr(state.observations, 'sugarPrice', 0)
        sunlight_index = getattr(state.observations, 'sunlightIndex', 0)
        import_tariff = getattr(state.observations, 'importTariff', 0)
        
        # Append current values to history for normalization
        self.sugar_history.append(sugar_price)
        self.sunlight_history.append(sunlight_index)
        self.import_tariff_history.append(import_tariff)
        
        # Keep history within the window size
        if len(self.sugar_history) > self.history_window:
            self.sugar_history.pop(0)
        if len(self.sunlight_history) > self.history_window:
            self.sunlight_history.pop(0)
        if len(self.import_tariff_history) > self.history_window:
            self.import_tariff_history.pop(0)

        # Normalize current values based on historical context
        sugar_score = self.normalize_value(sugar_price, self.sugar_history)
        sunlight_score = self.normalize_value(sunlight_index, self.sunlight_history)
        tariff_score = self.normalize_value(import_tariff, self.import_tariff_history)
        
        # Calculate the combined score based on weights
        # Higher score suggests undervaluation (buy signal)
        # Lower score suggests overvaluation (sell signal)
        total_score = (self.sugar_weight * sugar_score - 
                       self.sunlight_weight * sunlight_score - 
                       self.tariff_weight * tariff_score)
                       
        # Invert the score so higher values mean BUY (undervalued based on fundamentals)
        # and lower values mean SELL (overvalued based on fundamentals)
        # A score > 0.5 suggests buying, < 0.5 suggests selling. Center is 0.5.
        # The formula is designed such that high sugar price (bad) -> higher score (sell)
        # High sunlight (good) -> lower score (buy)
        # High tariff (bad) -> higher score (sell)
        # We need to invert this logic for buy/sell thresholds.
        # Let's redefine score calculation or threshold interpretation.
        
        # Original baseline calculation implies: 
        # High score = High sugar OR Low sunlight OR Low tariff => Suggests SELL
        # Low score = Low sugar OR High sunlight OR High tariff => Suggests BUY
        # We will keep this logic and adjust thresholds.
        # Buy when score is low (e.g., < sell_threshold = 0.45)
        # Sell when score is high (e.g., > buy_threshold = 0.55)
        
        return total_score
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        result = {}
        conversions = 0
        traderData = ""
        product = "MAGNIFICENT_MACARONS"

        if product not in state.order_depths:
            return result, conversions, traderData
            
        order_depth = state.order_depths[product]
        current_position = state.position.get(product, 0)
        
        # Check if we have valid bid/ask prices to trade
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return result, conversions, traderData
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # Calculate the fundamental score
        price_score = self.calculate_price_score(state)
        
        orders = []
        
        # Determine available capital for buying/selling
        available_buy = self.position_limit - current_position
        available_sell = self.position_limit + current_position # Max quantity we can sell
        
        # --- Aggressive Trading Logic Based on Fundamental Score ---
        
        # BUY Signal: If score is very low (strong indication of undervaluation)
        if price_score < self.sell_threshold: 
            if available_buy > 0:
                buy_quantity = min(available_buy, self.max_trade_quantity)
                # Be aggressive: take the ask
                buy_price = best_ask 
                print(f"BUY SIGNAL: Score {price_score:.2f} < {self.sell_threshold}. Placing BUY order for {buy_quantity} at {buy_price}")
                orders.append(Order(product, buy_price, buy_quantity))
        
        # SELL Signal: If score is very high (strong indication of overvaluation)
        elif price_score > self.buy_threshold:
            if available_sell > 0:
                sell_quantity = min(available_sell, self.max_trade_quantity)
                # Be aggressive: hit the bid
                sell_price = best_bid 
                print(f"SELL SIGNAL: Score {price_score:.2f} > {self.buy_threshold}. Placing SELL order for {sell_quantity} at {sell_price}")
                orders.append(Order(product, sell_price, -sell_quantity))
                
        # Close Position Logic: If score is near neutral, close existing positions
        elif abs(price_score - 0.5) < self.neutral_threshold:
             if current_position > 0:
                 # Close long position
                 sell_quantity = min(current_position, self.max_trade_quantity)
                 sell_price = best_bid # Hit the bid to close
                 print(f"NEUTRAL CLOSE: Score {price_score:.2f} near 0.5. Closing {sell_quantity} long position at {sell_price}")
                 orders.append(Order(product, sell_price, -sell_quantity))
             elif current_position < 0:
                 # Close short position
                 buy_quantity = min(abs(current_position), self.max_trade_quantity)
                 buy_price = best_ask # Take the ask to close
                 print(f"NEUTRAL CLOSE: Score {price_score:.2f} near 0.5. Closing {buy_quantity} short position at {buy_price}")
                 orders.append(Order(product, buy_price, buy_quantity))

        result[product] = orders
        # No conversion logic in this simple strategy
        return result, conversions, traderData 