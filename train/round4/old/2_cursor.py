from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import Dict, List, Any
import numpy as np
import json

# Global parameters
PARAMS = {
    # Position limits
    'position_limit': 75,
    'conversion_limit': 10,
    
    # Market making parameters
    'mm_spread_factor': 1.0,
    'min_profit_margin': 3.0,
    'inventory_scale_factor': 0.8,
    
    # Technical analysis parameters
    'ema_short_period': 5,
    'ema_long_period': 20,
    'regime_window': 20,
    
    # Fundamental analysis weights
    'sugar_weight': 0.2,
    'sunlight_weight': -0.1,
    'import_tariff_weight': -1.0,
    'export_tariff_weight': -0.5,
    'transport_weight': -1.0,
    
    # Regime-specific adjustments
    'cost_up_sugar_weight': 0.3,
    'cost_down_sugar_weight': 0.1,
    'tariff_up_import_weight': -1.5,
    'tariff_up_export_weight': -0.8,
    'tariff_down_import_weight': -0.5,
    'tariff_down_export_weight': -0.3,
}

class Trader:
    def __init__(self):
        # Initialize state tracking
        self.position_history = {}
        self.prev_observations = {}
        self.market_regimes = {}
        self.prev_mid_prices = {}
        self.ema_short = {}
        self.ema_long = {}
        self.volatility = {}
        
        # Load parameters
        self.POSITION_LIMIT = PARAMS['position_limit']
        self.CONVERSION_LIMIT = PARAMS['conversion_limit']
        self.mm_spread_factor = PARAMS['mm_spread_factor']
        self.min_profit_margin = PARAMS['min_profit_margin']
        self.inventory_scale_factor = PARAMS['inventory_scale_factor']
        self.ema_short_period = PARAMS['ema_short_period']
        self.ema_long_period = PARAMS['ema_long_period']
        self.regime_window = PARAMS['regime_window']
    
    def initialize_product(self, product: str, state: TradingState):
        """Initialize tracking for a new product"""
        if product not in self.position_history:
            self.position_history[product] = []
            self.prev_observations[product] = None
            self.market_regimes[product] = "UNKNOWN"
            self.prev_mid_prices[product] = []
            self.ema_short[product] = None
            self.ema_long[product] = None
            self.volatility[product] = None
    
    def calculate_mid_price(self, order_depth: OrderDepth) -> float:
        """Calculate the mid price from the order book"""
        if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        elif len(order_depth.buy_orders) > 0:
            return max(order_depth.buy_orders.keys())
        elif len(order_depth.sell_orders) > 0:
            return min(order_depth.sell_orders.keys())
        else:
            return None
    
    def update_market_metrics(self, product: str, state: TradingState):
        """Update market metrics based on current state"""
        if product not in state.order_depths:
            return
        
        order_depth = state.order_depths[product]
        mid_price = self.calculate_mid_price(order_depth)
        
        if mid_price:
            # Update price history
            self.prev_mid_prices[product].append(mid_price)
            if len(self.prev_mid_prices[product]) > self.regime_window:
                self.prev_mid_prices[product].pop(0)
            
            # Update EMAs
            if self.ema_short[product] is None:
                self.ema_short[product] = mid_price
                self.ema_long[product] = mid_price
            else:
                alpha_short = 2 / (self.ema_short_period + 1)
                alpha_long = 2 / (self.ema_long_period + 1)
                self.ema_short[product] = alpha_short * mid_price + (1 - alpha_short) * self.ema_short[product]
                self.ema_long[product] = alpha_long * mid_price + (1 - alpha_long) * self.ema_long[product]
            
            # Update volatility
            if len(self.prev_mid_prices[product]) > 1:
                price_changes = np.diff(self.prev_mid_prices[product])
                self.volatility[product] = np.std(price_changes)
    
    def detect_market_regime(self, product: str, state: TradingState):
        """Detect market regime based on price trends and fundamentals"""
        if product != "MAGNIFICENT_MACARONS" or product not in state.observations.conversionObservations:
            return "UNKNOWN"
        
        obs = state.observations.conversionObservations[product]
        prev_obs = self.prev_observations[product]
        
        # Check if we have enough data
        if prev_obs is None or len(self.prev_mid_prices[product]) < self.regime_window:
            self.prev_observations[product] = obs
            return "UNKNOWN"
        
        # Calculate indicators
        price_trend = 0
        if self.ema_short[product] is not None and self.ema_long[product] is not None:
            price_trend = self.ema_short[product] - self.ema_long[product]
        
        # Check fundamental changes
        sugar_change = obs.sugarPrice - prev_obs.sugarPrice if hasattr(obs, 'sugarPrice') else 0
        sunlight_change = obs.sunlight - prev_obs.sunlight if hasattr(obs, 'sunlight') else 0
        tariff_change = (obs.importTariff - prev_obs.importTariff) + (obs.exportTariff - prev_obs.exportTariff)
        
        # Determine regime
        if abs(tariff_change) > 0.5:
            if tariff_change > 0:
                return "TARIFF_UP"
            else:
                return "TARIFF_DOWN"
        elif abs(sugar_change) > 0.5:
            if sugar_change > 0:
                return "COST_UP"
            else:
                return "COST_DOWN"
        elif price_trend > 0.5 * self.volatility[product]:
            return "UPTREND"
        elif price_trend < -0.5 * self.volatility[product]:
            return "DOWNTREND"
        else:
            return "RANGE"
        
        self.prev_observations[product] = obs
    
    def estimate_fair_value(self, product: str, state: TradingState) -> float:
        """Estimate fair value of a product based on observations and market data"""
        if product != "MAGNIFICENT_MACARONS" or product not in state.observations.conversionObservations:
            return None
        
        obs = state.observations.conversionObservations[product]
        mid_price = self.calculate_mid_price(state.order_depths[product])
        
        if mid_price is None:
            return None
        
        # Base value calculation
        base_value = mid_price
        
        # Adjust for fundamentals based on regime
        regime = self.market_regimes[product]
        
        # Get weights based on regime
        sugar_weight = PARAMS['sugar_weight']
        import_tariff_weight = PARAMS['import_tariff_weight']
        export_tariff_weight = PARAMS['export_tariff_weight']
        
        if regime == "COST_UP":
            sugar_weight = PARAMS['cost_up_sugar_weight']
        elif regime == "COST_DOWN":
            sugar_weight = PARAMS['cost_down_sugar_weight']
        elif regime == "TARIFF_UP":
            import_tariff_weight = PARAMS['tariff_up_import_weight']
            export_tariff_weight = PARAMS['tariff_up_export_weight']
        elif regime == "TARIFF_DOWN":
            import_tariff_weight = PARAMS['tariff_down_import_weight']
            export_tariff_weight = PARAMS['tariff_down_export_weight']
        
        # Calculate adjustments
        sugar_adjustment = sugar_weight * obs.sugarPrice if hasattr(obs, 'sugarPrice') else 0
        sunlight_adjustment = PARAMS['sunlight_weight'] * obs.sunlight if hasattr(obs, 'sunlight') else 0
        import_tariff_adjustment = import_tariff_weight * obs.importTariff
        export_tariff_adjustment = export_tariff_weight * obs.exportTariff
        transport_adjustment = PARAMS['transport_weight'] * obs.transportFees
        
        # Add adjustments to base value
        fair_value = base_value + sugar_adjustment + sunlight_adjustment + import_tariff_adjustment + export_tariff_adjustment + transport_adjustment
        
        return fair_value
    
    def get_optimal_spread(self, product: str, state: TradingState) -> tuple:
        """Calculate optimal bid/ask spread based on volatility and position"""
        if product not in state.order_depths or self.volatility[product] is None:
            return (1.0, 1.0)
        
        # Base spread as factor of volatility
        base_spread = self.volatility[product] * self.mm_spread_factor
        
        # Adjust for inventory
        current_pos = state.position.get(product, 0)
        pos_factor = 1.0 + abs(current_pos / self.POSITION_LIMIT) ** 2 * self.inventory_scale_factor
        
        # Asymmetric spreads based on position
        if current_pos > 0:
            # Long position - tighter bid (buying), wider ask (selling)
            bid_spread = base_spread * 1.5 * pos_factor
            ask_spread = base_spread * 0.8
        elif current_pos < 0:
            # Short position - tighter ask (selling), wider bid (buying)
            bid_spread = base_spread * 0.8
            ask_spread = base_spread * 1.5 * pos_factor
        else:
            # Neutral position - symmetric spreads
            bid_spread = base_spread
            ask_spread = base_spread
        
        # Ensure minimum spread
        bid_spread = max(bid_spread, self.min_profit_margin / 2)
        ask_spread = max(ask_spread, self.min_profit_margin / 2)
        
        return (bid_spread, ask_spread)
    
    def should_convert(self, product: str, state: TradingState) -> int:
        """Determine if conversion should be done and in what direction"""
        if product != "MAGNIFICENT_MACARONS" or product not in state.observations.conversionObservations:
            return 0
        
        current_pos = state.position.get(product, 0)
        
        # No position, no conversion needed
        if current_pos == 0:
            return 0
        
        obs = state.observations.conversionObservations[product]
        
        # Calculate conversion cost
        conversion_cost = obs.transportFees + obs.exportTariff + obs.importTariff
        
        # If we have a long position and cost is favorable, convert (export)
        if current_pos > 0:
            conversion_amount = min(current_pos, self.CONVERSION_LIMIT)
            return conversion_amount if conversion_amount > 0 else 0
        
        # If we have a short position and cost is favorable, convert (import)
        if current_pos < 0:
            conversion_amount = min(abs(current_pos), self.CONVERSION_LIMIT)
            return -conversion_amount if conversion_amount > 0 else 0
        
        return 0
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        """
        Main trading logic
        """
        result = {}
        conversions = 0
        
        # Initialize state from traderData if available
        if state.traderData and state.traderData != "":
            try:
                trader_state = json.loads(state.traderData)
                if "position_history" in trader_state:
                    self.position_history = trader_state["position_history"]
                if "market_regimes" in trader_state:
                    self.market_regimes = trader_state["market_regimes"]
                if "prev_mid_prices" in trader_state:
                    self.prev_mid_prices = trader_state["prev_mid_prices"]
                if "ema_short" in trader_state:
                    self.ema_short = trader_state["ema_short"]
                if "ema_long" in trader_state:
                    self.ema_long = trader_state["ema_long"]
                if "volatility" in trader_state:
                    self.volatility = trader_state["volatility"]
            except:
                # If there's an error parsing, just continue with empty state
                pass
        
        # Process MAGNIFICENT_MACARONS
        product = "MAGNIFICENT_MACARONS"
        if product in state.order_depths:
            # Initialize if needed
            self.initialize_product(product, state)
            
            # Update market metrics
            self.update_market_metrics(product, state)
            
            # Detect market regime
            self.market_regimes[product] = self.detect_market_regime(product, state)
            
            # Calculate fair value
            fair_value = self.estimate_fair_value(product, state)
            
            # Only trade if we have a fair value estimate
            if fair_value is not None:
                order_depth = state.order_depths[product]
                orders = []
                
                # Get optimal spread
                bid_spread, ask_spread = self.get_optimal_spread(product, state)
                
                # Calculate order prices
                bid_price = int(fair_value - bid_spread)
                ask_price = int(fair_value + ask_spread)
                
                # Current position
                current_pos = state.position.get(product, 0)
                
                # Calculate remaining capacity
                buy_capacity = self.POSITION_LIMIT - current_pos
                sell_capacity = self.POSITION_LIMIT + current_pos
                
                # Process sell orders first (matching against buy orders)
                if len(order_depth.buy_orders) > 0:
                    # Sort buy orders by price (highest first)
                    for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
                        if price > fair_value - self.min_profit_margin / 2:  # Accept if price is good
                            # Calculate order quantity (limited by position limits)
                            execute_quantity = min(quantity, sell_capacity)
                            if execute_quantity > 0:
                                orders.append(Order(product, price, -execute_quantity))
                                sell_capacity -= execute_quantity
                
                # Process buy orders next (matching against sell orders)
                if len(order_depth.sell_orders) > 0:
                    # Sort sell orders by price (lowest first)
                    for price, quantity in sorted(order_depth.sell_orders.items()):
                        if price < fair_value + self.min_profit_margin / 2:  # Accept if price is good
                            # Calculate order quantity (limited by position limits)
                            execute_quantity = min(-quantity, buy_capacity)
                            if execute_quantity > 0:
                                orders.append(Order(product, price, execute_quantity))
                                buy_capacity -= execute_quantity
                
                # Place market making orders
                # Only place orders if there's remaining capacity
                
                # Place buy order at our desired price if we still have capacity
                if buy_capacity > 0:
                    # Scale order size based on available capacity and position
                    order_size = max(1, int(buy_capacity * (1 - abs(current_pos) / self.POSITION_LIMIT)))
                    orders.append(Order(product, bid_price, order_size))
                
                # Place sell order at our desired price if we still have capacity
                if sell_capacity > 0:
                    # Scale order size based on available capacity and position
                    order_size = max(1, int(sell_capacity * (1 - abs(current_pos) / self.POSITION_LIMIT)))
                    orders.append(Order(product, ask_price, -order_size))
                
                result[product] = orders
            
            # Check if conversion should be done
            conversions = self.should_convert(product, state)
            
            # Update position history
            self.position_history[product].append(state.position.get(product, 0))
            if len(self.position_history[product]) > self.regime_window:
                self.position_history[product].pop(0)
        
        # Serialize state data
        trader_data = json.dumps({
            "position_history": self.position_history,
            "market_regimes": self.market_regimes,
            "prev_mid_prices": self.prev_mid_prices,
            "ema_short": self.ema_short,
            "ema_long": self.ema_long,
            "volatility": self.volatility
        })
        
        return result, conversions, trader_data 