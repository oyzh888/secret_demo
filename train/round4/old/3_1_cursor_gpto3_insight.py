import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

class Trader:
    def __init__(self):
        # Position and conversion limits
        self.position_limit = 75
        self.conversion_limit = 10
        
        # Fair value model coefficients from GPT-O3 insights
        self.sugar_price_coef = 5.0      # Sugar price has positive relationship
        self.sunlight_coef = -3.3        # Sunlight has negative relationship
        self.export_tariff_coef = -62.0  # Export tariff impacts price negatively
        self.import_tariff_coef = -52.0  # Import tariff impacts price negatively
        self.transport_fee_coef = 62.0   # Transport fee has positive impact
        self.const = 180.0               # Constant term (approximate from regression)
        
        # Dynamic threshold parameters
        self.k1 = 1.1                    # Spread scaling factor
        self.k2 = 0.8                    # Volatility scaling factor
        self.min_threshold = 1.0         # Minimum threshold
        
        # Trading parameters
        self.min_delta_to_trade = 5      # Minimum position delta to trigger trading
        self.order_step_size = 5         # Max quantity per single order
        self.price_offset = 0.5          # Price offset for aggressive orders
        self.min_spread_for_maker = 1.0  # Minimum spread to place maker orders
        
        # Risk management
        self.stop_loss_threshold = -10   # Stop loss threshold per unit
        self.take_profit_threshold = 0.2 # Threshold for mean reversion profit taking
        
        # Historical data
        self.epsilon_history = []        # History of price discrepancies
        self.spread_history = []         # History of market spreads
        self.mid_price_history = []      # History of mid prices
        self.fair_value_history = []     # History of calculated fair values
        self.position_history = []       # History of positions
        self.pnl_history = []            # History of PnL
        
        # State variables
        self.position = 0                # Current position
        self.last_fair_value = None      # Last calculated fair value
        self.last_mid_price = None       # Last observed mid price
        self.spread_mean = 1.5           # Initial estimate of spread mean
        self.epsilon_sigma = 2.0         # Initial estimate of epsilon standard deviation
        self.history_window = 60         # Window for historical calculations
        self.volatility_window = 20      # Window for volatility calculations
        
        # Storage cost (fixed)
        self.storage_cost = 0.1
        
    def calculate_fair_value(self, observation):
        """Calculate the fair value based on the regression model"""
        if observation is None:
            return self.last_fair_value
        
        # Extract features from observation
        sugar_price = getattr(observation, 'sugarPrice', 0)
        sunlight = getattr(observation, 'sunlightIndex', 0)
        export_tariff = getattr(observation, 'exportTariff', 0)
        import_tariff = getattr(observation, 'importTariff', 0)
        transport_fee = getattr(observation, 'transportFee', 0)
        
        # Calculate fair value
        fair_value = (
            self.sugar_price_coef * sugar_price +
            self.sunlight_coef * sunlight +
            self.export_tariff_coef * export_tariff +
            self.import_tariff_coef * import_tariff +
            self.transport_fee_coef * transport_fee +
            self.const - 
            self.storage_cost  # Include storage cost
        )
        
        return fair_value
    
    def calculate_mid_price(self, order_depth):
        """Calculate mid price from the order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return self.last_mid_price
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        return mid_price, best_bid, best_ask
    
    def update_dynamic_parameters(self):
        """Update dynamic parameters like spread mean and epsilon sigma"""
        # Update spread mean if we have enough history
        if len(self.spread_history) > self.volatility_window:
            recent_spreads = self.spread_history[-self.volatility_window:]
            self.spread_mean = np.mean(recent_spreads)
        
        # Update epsilon sigma if we have enough history
        if len(self.epsilon_history) > self.volatility_window:
            recent_epsilons = self.epsilon_history[-self.volatility_window:]
            self.epsilon_sigma = np.std(recent_epsilons)
    
    def calculate_dynamic_threshold(self):
        """Calculate dynamic threshold based on spread and volatility"""
        spread_threshold = self.spread_mean * self.k1
        volatility_threshold = self.epsilon_sigma * self.k2
        
        # Take the max of the two thresholds
        threshold = max(spread_threshold, volatility_threshold, self.min_threshold)
        
        return threshold
    
    def calculate_target_position(self, epsilon, threshold):
        """Calculate target position based on price discrepancy"""
        # Clip position between position limits
        # negative epsilon (undervalued) -> positive target (buy)
        # positive epsilon (overvalued) -> negative target (sell)
        raw_target = -epsilon / threshold
        target_position = max(-self.position_limit, min(self.position_limit, raw_target))
        
        # Discrete target position (whole numbers only)
        return int(target_position)
    
    def should_close_position(self, position, fair_value, mid_price):
        """Determine if we should close out position based on risk parameters"""
        if position == 0:
            return False
        
        # Check if we've reached our take profit level (epsilon near zero)
        epsilon = mid_price - fair_value
        if abs(epsilon) < self.take_profit_threshold:
            return True
        
        # Check for losing positions that need to be cut
        if len(self.fair_value_history) > 1 and len(self.mid_price_history) > 1:
            entry_price = self.mid_price_history[-2]
            current_price = mid_price
            
            # For long positions
            if position > 0 and (current_price - entry_price) < self.stop_loss_threshold:
                return True
            
            # For short positions
            if position < 0 and (entry_price - current_price) < self.stop_loss_threshold:
                return True
        
        return False
    
    def update_history(self, position, fair_value, mid_price, spread, epsilon):
        """Update historical data tracking"""
        # Update price histories
        if fair_value is not None:
            self.fair_value_history.append(fair_value)
            self.last_fair_value = fair_value
        
        if mid_price is not None:
            self.mid_price_history.append(mid_price)
            self.last_mid_price = mid_price
        
        # Update metric histories
        if spread is not None:
            self.spread_history.append(spread)
        
        if epsilon is not None:
            self.epsilon_history.append(epsilon)
        
        # Update position history
        self.position_history.append(position)
        self.position = position
        
        # Trim histories to window size
        if len(self.mid_price_history) > self.history_window:
            self.mid_price_history = self.mid_price_history[-self.history_window:]
            self.fair_value_history = self.fair_value_history[-self.history_window:]
            self.position_history = self.position_history[-self.history_window:]
            self.spread_history = self.spread_history[-self.history_window:]
            self.epsilon_history = self.epsilon_history[-self.history_window:]
            self.pnl_history = self.pnl_history[-self.history_window:]
    
    def get_order_book_depth(self, order_depth, depth_levels=3):
        """Analyze order book depth on bid and ask sides"""
        bid_volume = 0
        ask_volume = 0
        
        # Calculate volume on bid side
        if order_depth.buy_orders:
            sorted_bids = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
            for i, (price, quantity) in enumerate(sorted_bids):
                if i >= depth_levels:
                    break
                bid_volume += abs(quantity)
        
        # Calculate volume on ask side
        if order_depth.sell_orders:
            sorted_asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
            for i, (price, quantity) in enumerate(sorted_asks):
                if i >= depth_levels:
                    break
                ask_volume += abs(quantity)
        
        # Calculate imbalance
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            imbalance = (bid_volume - ask_volume) / total_volume
        else:
            imbalance = 0
        
        return bid_volume, ask_volume, imbalance
    
    def adjust_order_price(self, is_buy, best_bid, best_ask, imbalance):
        """Adjust order price based on order book imbalance"""
        spread = best_ask - best_bid
        
        # If spread is tight, use more aggressive pricing
        if spread < self.min_spread_for_maker:
            if is_buy:
                price = best_ask  # Pay the ask when buying
            else:
                price = best_bid  # Hit the bid when selling
        else:
            # Use imbalance to adjust pricing
            imbalance_factor = min(1.0, max(-1.0, imbalance * 2))  # Scale imbalance effect
            
            if is_buy:
                # More sellers (negative imbalance) -> be patient
                # More buyers (positive imbalance) -> be aggressive
                if imbalance < -0.3:
                    price = best_bid + self.price_offset * 0.5  # More passive
                elif imbalance > 0.3:
                    price = best_ask  # More aggressive
                else:
                    price = best_bid + self.price_offset  # Default
            else:
                # More buyers (positive imbalance) -> be patient
                # More sellers (negative imbalance) -> be aggressive
                if imbalance > 0.3:
                    price = best_ask - self.price_offset * 0.5  # More passive
                elif imbalance < -0.3:
                    price = best_bid  # More aggressive
                else:
                    price = best_ask - self.price_offset  # Default
        
        return price
    
    def calculate_order_quantity(self, delta):
        """Calculate appropriate order quantity based on position delta"""
        # Limit order size to step size, preserve sign
        sign = 1 if delta > 0 else -1
        quantity = min(abs(delta), self.order_step_size)
        
        return sign * quantity
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        """Main strategy execution method"""
        result = {}
        conversions = 0
        
        # Check if our product exists in the order depths
        if "MAGNIFICENT_MACARONS" not in state.order_depths:
            return result, conversions, state.traderData
        
        # Get order depth and position
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        position = state.position.get("MAGNIFICENT_MACARONS", 0)
        
        # Get observation for MAGNIFICENT_MACARONS if available
        observation = None
        if hasattr(state, 'observations') and state.observations:
            if isinstance(state.observations, dict) and "MAGNIFICENT_MACARONS" in state.observations:
                observation = state.observations["MAGNIFICENT_MACARONS"]
            elif not isinstance(state.observations, dict):
                observation = state.observations
        
        # Calculate fair value from regression model
        fair_value = self.calculate_fair_value(observation)
        
        # Calculate market price (mid, bid, ask)
        market_data = self.calculate_mid_price(order_depth)
        if not market_data:
            # Cannot trade without market data
            return result, conversions, state.traderData
        
        mid_price, best_bid, best_ask = market_data
        spread = best_ask - best_bid
        
        # Calculate epsilon (price discrepancy)
        epsilon = mid_price - fair_value
        
        # Update history and dynamic parameters
        self.update_history(position, fair_value, mid_price, spread, epsilon)
        self.update_dynamic_parameters()
        
        # Calculate dynamic threshold
        threshold = self.calculate_dynamic_threshold()
        
        # Calculate target position
        target_position = self.calculate_target_position(epsilon, threshold)
        
        # Calculate delta (how much to trade)
        delta = target_position - position
        
        # Initialize orders list
        orders = []
        
        # Check if we should close position based on risk management
        close_position = self.should_close_position(position, fair_value, mid_price)
        
        # Analyze order book depth and imbalance
        bid_volume, ask_volume, imbalance = self.get_order_book_depth(order_depth)
        
        # Trading logic
        if close_position:
            # Close out position
            if position > 0:
                # SELL to close long position
                price = best_bid  # Hit the bid when closing
                orders.append(Order("MAGNIFICENT_MACARONS", price, -position))
            elif position < 0:
                # BUY to close short position
                price = best_ask  # Pay the ask when closing
                orders.append(Order("MAGNIFICENT_MACARONS", price, -position))
        
        elif abs(delta) >= self.min_delta_to_trade:
            # Calculate order quantity with step size limit
            quantity = self.calculate_order_quantity(delta)
            
            # Set price based on order direction and market conditions
            is_buy = quantity > 0
            price = self.adjust_order_price(is_buy, best_bid, best_ask, imbalance)
            
            # Create order with correct format
            orders.append(Order("MAGNIFICENT_MACARONS", price, quantity))
        
        # Return results with correct format
        result["MAGNIFICENT_MACARONS"] = orders
        return result, conversions, state.traderData 