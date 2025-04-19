import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# Global parameters
PARAMS = {
    # Position limits
    'position_limit': 75,  # Maximum position limit
    'conversion_limit': 10,  # Conversion limit
    
    # Fair value model parameters (from regression analysis)
    'intercept': 187.6120,
    'sunlight_coef': -3.3115,
    'sugar_price_coef': 4.9708,
    'transport_fee_coef': 61.5302,
    'export_tariff_coef': -62.5394,
    'import_tariff_coef': -52.0653,
    
    # Trading parameters
    'alpha': 0.15,  # Price discrepancy threshold (percentage)
    'max_trade_quantity': 25,  # Maximum quantity per trade
    'min_spread': 0.5,  # Minimum required spread for market making
    'order_levels': 3,  # Number of price levels to place orders
    'level_spacing': 0.5,  # Price spacing between order levels
    
    # Risk management
    'max_position_scale': 0.8,  # Scale down orders as position approaches limit
    'dynamic_alpha': True,  # Dynamically adjust alpha based on position
    'alpha_scale_factor': 0.05,  # Alpha adjustment factor based on position
    
    # Volatility parameters
    'volatility_window': 20,  # Window for volatility calculation
    'volatility_scale': 1.5,  # Scale alpha by volatility
    
    # Data history
    'history_window': 100,  # Window for historical data
    
    # 回归模型参数
    'regression_coefs': {
        'intercept': 187.6120,
        'sunlight': -3.3115,
        'sugar_price': 4.9708,
        'transport_fee': 61.5302,
        'export_tariff': -62.5394,
        'import_tariff': -52.0653
    },
    
    # 交易参数
    'price_buffer': 5.0,  # 价格缓冲区间
    'imbalance_threshold': 0.2,  # 订单簿不平衡阈值
    
    # 技术分析参数
    'ma_short_window': 10,  # 短期移动平均窗口
    'ma_long_window': 50,   # 长期移动平均窗口
    
    # 市场微观结构参数
    'depth_threshold': 5,  # 订单簿深度阈值
    'spread_threshold': 0.1,  # 价差阈值
    'volume_threshold': 3,  # 交易量阈值
    'pressure_window': 10,  # 压力计算窗口
    'pressure_threshold': 0.7,  # 压力阈值
}

class Trader:
    def __init__(self):
        # Initialize parameters from global config
        self.position_limit = PARAMS['position_limit']
        self.conversion_limit = PARAMS['conversion_limit']
        
        # Fair value model parameters
        self.intercept = PARAMS['intercept']
        self.sunlight_coef = PARAMS['sunlight_coef']
        self.sugar_price_coef = PARAMS['sugar_price_coef']
        self.transport_fee_coef = PARAMS['transport_fee_coef']
        self.export_tariff_coef = PARAMS['export_tariff_coef']
        self.import_tariff_coef = PARAMS['import_tariff_coef']
        
        # Trading parameters
        self.alpha = PARAMS['alpha']
        self.max_trade_quantity = PARAMS['max_trade_quantity']
        self.min_spread = PARAMS['min_spread']
        self.order_levels = PARAMS['order_levels']
        self.level_spacing = PARAMS['level_spacing']
        
        # Data storage
        self.price_history = []
        self.fair_price_history = []
        self.mid_price_history = []
        self.position_history = []
        self.volatility_history = []
        self.observation_history = []
        
        # Misc
        self.timestamp = 0
        self.last_fair_price = None
        self.volatility = 0
        self.storage_cost = 0.1  # Fixed storage cost
        
        # 回归模型系数
        self.reg_coefs = PARAMS['regression_coefs']
        
    def calculate_ma(self, prices: List[float], window: int) -> float:
        """计算移动平均"""
        if len(prices) < window:
            return prices[-1] if prices else 0
        return sum(prices[-window:]) / window
    
    def estimate_fair_price(self, state: TradingState) -> float:
        """使用回归模型估计公平价格"""
        sugar_price = getattr(state.observations, 'sugarPrice', 0)
        sunlight_index = getattr(state.observations, 'sunlightIndex', 0)
        import_tariff = getattr(state.observations, 'importTariff', 0)
        export_tariff = getattr(state.observations, 'exportTariff', 0)
        transport_fee = getattr(state.observations, 'transportFee', 0)
        
        fair_price = (
            self.reg_coefs['intercept'] +
            self.reg_coefs['sunlight'] * sunlight_index +
            self.reg_coefs['sugar_price'] * sugar_price +
            self.reg_coefs['transport_fee'] * transport_fee +
            self.reg_coefs['export_tariff'] * export_tariff +
            self.reg_coefs['import_tariff'] * import_tariff
        )
        
        return fair_price
    
    def calculate_market_pressure(self, order_depth: OrderDepth) -> tuple[float, float]:
        """计算市场买卖压力"""
        buy_pressure = 0
        sell_pressure = 0
        
        if order_depth.buy_orders:
            for price, quantity in order_depth.buy_orders.items():
                buy_pressure += price * quantity
            
        if order_depth.sell_orders:
            for price, quantity in order_depth.sell_orders.items():
                sell_pressure += price * quantity
            
        total_pressure = buy_pressure + sell_pressure
        if total_pressure > 0:
            buy_pressure = buy_pressure / total_pressure
            sell_pressure = sell_pressure / total_pressure
            
        return buy_pressure, sell_pressure
    
    def analyze_order_book_imbalance(self, order_depth: OrderDepth) -> float:
        """分析订单簿不平衡度"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        buy_volume = 0
        sell_volume = 0
        
        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True)[:PARAMS['depth_threshold']]:
            buy_volume += price * quantity
            
        for price, quantity in sorted(order_depth.sell_orders.items())[:PARAMS['depth_threshold']]:
            sell_volume += price * quantity
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0
            
        imbalance = (buy_volume - sell_volume) / total_volume
        return imbalance
        
    def calculate_volatility(self):
        """Calculate price volatility for adaptive alpha"""
        if len(self.mid_price_history) < PARAMS['volatility_window']:
            return 0.01  # Default low volatility
            
        # Use standard deviation of recent price changes
        recent_prices = self.mid_price_history[-PARAMS['volatility_window']:]
        price_changes = np.diff(recent_prices)
        volatility = np.std(price_changes)
        normalized_volatility = volatility / np.mean(recent_prices)
        
        return max(0.01, normalized_volatility)
        
    def get_dynamic_alpha(self, position):
        """Adjust alpha based on position and volatility"""
        if not PARAMS['dynamic_alpha']:
            return self.alpha
            
        # Base alpha adjusted by position
        position_ratio = abs(position) / self.position_limit
        position_factor = 1 + position_ratio * PARAMS['alpha_scale_factor']
        
        # Further adjust by volatility
        volatility_factor = min(2.0, max(0.5, self.volatility * PARAMS['volatility_scale']))
        
        # Discourage trading against position (increase alpha when adding to position)
        if (position > 0 and self.last_fair_price < self.mid_price_history[-1]) or \
           (position < 0 and self.last_fair_price > self.mid_price_history[-1]):
            position_factor *= 1.2
            
        return self.alpha * position_factor * volatility_factor
        
    def calculate_order_quantity(self, position, is_buy):
        """Calculate order quantity based on position limits"""
        available_position = self.position_limit - abs(position)
        if is_buy and position < 0:
            # Closing short position
            max_quantity = min(abs(position), self.max_trade_quantity)
        elif is_buy and position >= 0:
            # Adding to long position
            max_quantity = min(available_position, self.max_trade_quantity)
            # Scale down as we approach position limit
            scale_factor = 1 - (position / self.position_limit) * PARAMS['max_position_scale']
            max_quantity *= scale_factor
        elif not is_buy and position > 0:
            # Closing long position
            max_quantity = min(position, self.max_trade_quantity)
        else:
            # Adding to short position
            max_quantity = min(available_position, self.max_trade_quantity)
            # Scale down as we approach position limit
            scale_factor = 1 - (abs(position) / self.position_limit) * PARAMS['max_position_scale']
            max_quantity *= scale_factor
            
        return max(1, int(max_quantity))
        
    def should_trade(self, fair_price, market_price, position):
        """Determine if we should trade based on price discrepancy"""
        if fair_price is None or market_price is None:
            return False, False
            
        dynamic_alpha = self.get_dynamic_alpha(position)
        
        # Calculate price discrepancy as percentage
        discrepancy = (fair_price - market_price) / market_price
        
        # Buy signal when market price is significantly lower than fair price
        buy_signal = discrepancy > dynamic_alpha
        
        # Sell signal when market price is significantly higher than fair price
        sell_signal = discrepancy < -dynamic_alpha
        
        return buy_signal, sell_signal
        
    def manage_position(self, position, fair_price, market_price, order_depth):
        """Determine whether to reduce position based on price movement"""
        if len(self.fair_price_history) < 5 or fair_price is None or market_price is None:
            return False, False
            
        # Get market data
        _, best_bid, best_ask = self.calculate_market_price(order_depth)
        
        # Check for closing long positions
        close_long = (position > 0 and
                     (fair_price < market_price * 0.99 or  # Fair price dropped
                      fair_price < self.fair_price_history[-2] * 0.99))  # Fair price trend down
                      
        # Check for closing short positions
        close_short = (position < 0 and
                      (fair_price > market_price * 1.01 or  # Fair price increased
                       fair_price > self.fair_price_history[-2] * 1.01))  # Fair price trend up
                       
        return close_long, close_short
        
    def place_multi_level_orders(self, base_price, is_buy, quantity, order_depth):
        """Place orders at multiple price levels"""
        orders = []
        remaining_quantity = quantity
        
        # Number of levels to use (maximum 3)
        num_levels = min(self.order_levels, 3)
        
        # Quantity distribution across levels (front-loaded)
        quantity_distribution = [0.6, 0.3, 0.1]  # 60%/30%/10% distribution
        
        for level in range(num_levels):
            # Calculate price for this level
            if is_buy:
                # For buy orders, move price down at deeper levels
                level_price = base_price - (level * self.level_spacing)
                # Ensure we don't bid higher than existing orders
                if order_depth.buy_orders and level_price > max(order_depth.buy_orders.keys()):
                    level_price = max(order_depth.buy_orders.keys())
            else:
                # For sell orders, move price up at deeper levels
                level_price = base_price + (level * self.level_spacing)
                # Ensure we don't offer lower than existing orders
                if order_depth.sell_orders and level_price < min(order_depth.sell_orders.keys()):
                    level_price = min(order_depth.sell_orders.keys())
                    
            # Calculate quantity for this level
            level_quantity = max(1, int(quantity * quantity_distribution[level]))
            
            # Ensure we don't exceed remaining quantity
            level_quantity = min(level_quantity, remaining_quantity)
            remaining_quantity -= level_quantity
            
            if level_quantity > 0:
                # Create the order (negative quantity for sell)
                if is_buy:
                    orders.append(Order("MAGNIFICENT_MACARONS", level_price, level_quantity))
                else:
                    orders.append(Order("MAGNIFICENT_MACARONS", level_price, -level_quantity))
                
        return orders
        
    def update_history(self, position, fair_price, mid_price, observation):
        """Update historical data tracking"""
        # Store current timestamp
        self.timestamp += 1
        
        # Update price histories
        if fair_price is not None:
            self.fair_price_history.append(fair_price)
            self.last_fair_price = fair_price
        
        if mid_price is not None:
            self.mid_price_history.append(mid_price)
            
        # Update position history
        self.position_history.append(position)
        
        # Store observation data if available
        if observation is not None:
            self.observation_history.append({
                'timestamp': self.timestamp,
                'sunlight': getattr(observation, 'sunlightIndex', 0),
                'sugar_price': getattr(observation, 'sugarPrice', 0),
                'transport_fee': getattr(observation, 'transportFee', 0),
                'export_tariff': getattr(observation, 'exportTariff', 0),
                'import_tariff': getattr(observation, 'importTariff', 0)
            })
            
        # Update volatility calculation
        self.volatility = self.calculate_volatility()
        self.volatility_history.append(self.volatility)
        
        # Trim histories to window size
        if len(self.mid_price_history) > PARAMS['history_window']:
            self.mid_price_history = self.mid_price_history[-PARAMS['history_window']:]
            self.fair_price_history = self.fair_price_history[-PARAMS['history_window']:]
            self.position_history = self.position_history[-PARAMS['history_window']:]
            self.volatility_history = self.volatility_history[-PARAMS['history_window']:]
            self.observation_history = self.observation_history[-PARAMS['history_window']:]
            
    def calculate_market_price(self, order_depth):
        """Calculate mid price from order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        return mid_price, best_bid, best_ask
        
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
        
        # Calculate fair price from regression model
        fair_price = self.estimate_fair_price(state)
        
        # Calculate market price
        market_data = self.calculate_market_price(order_depth)
        if market_data:
            mid_price, best_bid, best_ask = market_data
        else:
            # Cannot trade without market data
            return result, conversions, state.traderData
            
        # Update historical data
        self.update_history(position, fair_price, mid_price, observation)
        
        # Calculate market pressure
        buy_pressure, sell_pressure = self.calculate_market_pressure(order_depth)
        
        # Calculate order book imbalance
        imbalance = self.analyze_order_book_imbalance(order_depth)
        
        # Calculate technical indicators
        ma_short = self.calculate_ma(self.price_history, PARAMS['ma_short_window'])
        ma_long = self.calculate_ma(self.price_history, PARAMS['ma_long_window'])
        
        # Determine trading signals
        buy_signal, sell_signal = self.should_trade(fair_price, mid_price, position)
        
        # Check if we should manage/reduce position
        close_long, close_short = self.manage_position(position, fair_price, mid_price, order_depth)
        
        # Initialize orders list
        orders = []
        
        # Trading logic
        if buy_signal and not close_long:
            # BUY signal - place buy order at the best ask
            buy_quantity = self.calculate_order_quantity(position, True)
            if buy_quantity > 0:
                # Place orders at multiple levels
                buy_orders = self.place_multi_level_orders(best_ask, True, buy_quantity, order_depth)
                orders.extend(buy_orders)
                
        elif sell_signal and not close_short:
            # SELL signal - place sell order at the best bid
            sell_quantity = self.calculate_order_quantity(position, False)
            if sell_quantity > 0:
                # Place orders at multiple levels
                sell_orders = self.place_multi_level_orders(best_bid, False, sell_quantity, order_depth)
                orders.extend(sell_orders)
                
        # Position management logic
        elif close_long and position > 0:
            # Close long position - sell at best bid
            sell_quantity = min(position, self.max_trade_quantity)
            if sell_quantity > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -sell_quantity))
                
        elif close_short and position < 0:
            # Close short position - buy at best ask
            buy_quantity = min(abs(position), self.max_trade_quantity)
            if buy_quantity > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", best_ask, buy_quantity))
                
        # Return results
        result["MAGNIFICENT_MACARONS"] = orders
        return result, conversions, state.traderData 