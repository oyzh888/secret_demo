from typing import Dict, List, Tuple, Optional, Set
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np

# Product limits
LIMIT = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75
}

# Volatility Tiers based on analysis
VOLATILITY_TIERS = {
    # High Volatility
    "PICNIC_BASKET1": "high",
    "VOLCANIC_ROCK": "high",
    "VOLCANIC_ROCK_VOUCHER_9500": "high",
    "VOLCANIC_ROCK_VOUCHER_9750": "high",
    # Medium Volatility
    "MAGNIFICENT_MACARONS": "medium",
    "VOLCANIC_ROCK_VOUCHER_10000": "medium",
    "SQUID_INK": "medium",
    "PICNIC_BASKET2": "medium",
    "DJEMBES": "medium",
    "JAMS": "medium",
    "VOLCANIC_ROCK_VOUCHER_10250": "medium",
    # Low Volatility
    "CROISSANTS": "low",
    "KELP": "low",
    "VOLCANIC_ROCK_VOUCHER_10500": "low",
    "RAINFOREST_RESIN": "low",
}

# Key Counterparties identified in analysis
KEY_COUNTERPARTIES = {"Caesar", "Camilla", "Charlie"}

# Parameters
PARAM = {
    # Base parameters
    "tight_spread": 1,
    "k_vol": 1.2,
    "panic_ratio": 0.8,
    "panic_add": 4,
    "mm_size_frac": 0.25,
    "aggr_take": True,
    
    # Volatility-specific parameters
    "low_vol": {
        "k_vol": 0.5,  # Less sensitive to vol
        "mm_size_frac": 0.3,  # Larger size for stable products
        "aggr_take": False  # Less aggressive for low vol
    },
    "high_vol": {
        "tight_spread": 2,  # Wider base spread
        "k_vol": 1.5,  # More sensitive to vol
        "mm_size_frac": 0.2,  # Smaller size for high vol
        "momentum_window": 5  # Look at recent price change
    },
    
    # Counterparty parameters
    "cp_aggr_add": 1,        # Extra spread for aggressive CPs (Caesar buy)
    "cp_passive_sub": 0,     # Spread reduction for passive CPs (Charlie)
    "cp_trend_follow_add": 1, # Extra spread when trading against trend follower (Camilla buy)
    
    # Trend detection
    "ema_short": 5,
    "ema_long": 15,
    "trend_window": 20,
    "trend_threshold": 0.6
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.last_mid_price = {}
        self.last_counterparty = defaultdict(lambda: None)
        self.ema_short = defaultdict(lambda: None)
        self.ema_long = defaultdict(lambda: None)
        self.trends = defaultdict(int)  # Track market trends (1=up, -1=down, 0=neutral)
        
    def _vol(self, p: str) -> float:
        """Calculate volatility for a product"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """Calculate mid price from order book"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def update_last_counterparty(self, own_trades: Dict[str, List]):
        """Update the last counterparty for each product"""
        for symbol, trades in own_trades.items():
            if trades:
                # Use the counterparty from the most recent trade for this symbol
                # In the backtester, trades might not have counter_party attribute
                trade = trades[-1]
                if hasattr(trade, 'counter_party'):
                    self.last_counterparty[symbol] = trade.counter_party
                # Try to get buyer/seller in backtester environment
                elif hasattr(trade, 'buyer') and hasattr(trade, 'seller'):
                    # If we're the buyer, the counterparty is the seller
                    if trade.buyer == 'SUBMISSION':
                        self.last_counterparty[symbol] = trade.seller
                    # If we're the seller, the counterparty is the buyer
                    elif trade.seller == 'SUBMISSION':
                        self.last_counterparty[symbol] = trade.buyer
    
    def _update_indicators(self, p: str, mid: float):
        """Update technical indicators for a product"""
        self.prices[p].append(mid)
        prices_arr = np.array(self.prices[p])
        
        # Update EMAs
        if len(prices_arr) >= PARAM["ema_short"]:
            self.ema_short[p] = np.mean(prices_arr[-PARAM["ema_short"]:])
        if len(prices_arr) >= PARAM["ema_long"]:
            self.ema_long[p] = np.mean(prices_arr[-PARAM["ema_long"]:])
    
    def detect_trend(self, product: str) -> int:
        """Detect market trend based on recent price movements"""
        prices = self.prices[product]
        if len(prices) < PARAM["trend_window"]:
            return 0  # Not enough data
            
        recent_prices = prices[-PARAM["trend_window"]:]  
        up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
        
        up_ratio = up_moves / (len(recent_prices) - 1)
        down_ratio = down_moves / (len(recent_prices) - 1)
        
        if up_ratio > PARAM["trend_threshold"]:
            return 1  # Uptrend
        elif down_ratio > PARAM["trend_threshold"]:
            return -1  # Downtrend
        return 0  # Neutral
    
    def adjust_price_for_counterparty(self, base_price: int, is_buy: bool, counterparty: Optional[str]) -> int:
        """Adjust price based on counterparty analysis"""
        # Default to base price if no counterparty info
        if not counterparty or counterparty not in KEY_COUNTERPARTIES:
            return base_price
            
        if counterparty == "Caesar":
            # Caesar buys aggressively, sell higher
            if not is_buy:
                return base_price + PARAM["cp_aggr_add"]
        elif counterparty == "Camilla":
            # Camilla buys biased, sell higher (trend following)
            if not is_buy:
                return base_price + PARAM["cp_trend_follow_add"]
        elif counterparty == "Charlie":
            # Charlie is neutral, maybe slightly tighter spread?
            pass  # Keep base price for now
            
        return base_price
    
    def mm_product(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """Market making strategy for a single product"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # Record mid price
        self.last_mid_price[p] = mid
        self._update_indicators(p, mid)
        
        # Detect market trend
        trend = self.detect_trend(p)
        self.trends[p] = trend
        
        # Get volatility tier
        vol_tier = VOLATILITY_TIERS.get(p, "medium")
        
        # Get last counterparty
        last_cp = self.last_counterparty[p]
        
        # Calculate base spread based on volatility tier
        if vol_tier == "low":
            k_vol = PARAM["low_vol"]["k_vol"]
            mm_size_frac = PARAM["low_vol"]["mm_size_frac"]
            aggr_take = PARAM["low_vol"]["aggr_take"]
            tight_spread = PARAM["tight_spread"]
        elif vol_tier == "high":
            k_vol = PARAM["high_vol"]["k_vol"]
            mm_size_frac = PARAM["high_vol"]["mm_size_frac"]
            aggr_take = PARAM["aggr_take"]
            tight_spread = PARAM["high_vol"]["tight_spread"]
        else:  # medium
            k_vol = PARAM["k_vol"]
            mm_size_frac = PARAM["mm_size_frac"]
            aggr_take = PARAM["aggr_take"]
            tight_spread = PARAM["tight_spread"]
        
        # Calculate spread
        spread = int(tight_spread + k_vol * self._vol(p))
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # Adjust prices based on counterparty analysis
        buy_px = self.adjust_price_for_counterparty(buy_px, True, last_cp)
        sell_px = self.adjust_price_for_counterparty(sell_px, False, last_cp)
        
        # Adjust prices based on trend
        if trend == 1:  # Uptrend
            buy_px += 1  # More aggressive buying
            sell_px += 1  # Less aggressive selling
        elif trend == -1:  # Downtrend
            buy_px -= 1  # Less aggressive buying
            sell_px -= 1  # More aggressive selling
        
        # Calculate order size
        size = max(1, int(LIMIT[p] * mm_size_frac))
        
        # Panic mode for large positions
        if abs(pos) >= LIMIT[p] * PARAM["panic_ratio"]:
            buy_px = int(mid - PARAM["panic_add"] - spread)
            sell_px = int(mid + PARAM["panic_add"] + spread)
            size = max(size, abs(pos)//2)
            
        # Take orders if they're favorable
        b, a = best_bid_ask(depth)
        if aggr_take and b is not None and a is not None:
            if a < mid - spread and pos < LIMIT[p]:
                qty = min(size, LIMIT[p] - pos, abs(depth.sell_orders[a]))
                if qty > 0: orders.append(Order(p, a, qty))
            if b > mid + spread and pos > -LIMIT[p]:
                qty = min(size, LIMIT[p] + pos, depth.buy_orders[b])
                if qty > 0: orders.append(Order(p, b, -qty))
                
        # Regular market making orders
        if pos < LIMIT[p]:
            orders.append(Order(p, buy_px, min(size, LIMIT[p] - pos)))
        if pos > -LIMIT[p]:
            orders.append(Order(p, sell_px, -min(size, LIMIT[p] + pos)))
            
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """Main trading logic"""
        result: Dict[str, List[Order]] = {}
        
        # Update counterparty information
        self.update_last_counterparty(state.own_trades)
        
        # Apply market making strategy to all products
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.mm_product(p, depth, state.position.get(p, 0))
        
        # No conversions in this strategy
        conversions = 0
        
        # Return the orders, conversions, and trader data
        return result, conversions, state.traderData
