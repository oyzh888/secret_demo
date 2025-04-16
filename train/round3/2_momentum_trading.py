import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Symbol

# Define Product and logger locally
class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

class Logger:
    def print(self, *args, **kwargs):
        print(*args, **kwargs)

logger = Logger()

class MomentumTradingStrategy:
    def __init__(self, trader):
        self.trader = trader
        self.params = {
            "window": 10,
            "threshold": 0.05
        }
        
    def calculate_momentum(self, price_history: List[float]) -> float:
        if len(price_history) < self.params["window"]:
            return 0.0
        return (price_history[-1] - price_history[0]) / price_history[0]
        
    def run(self, state: TradingState, trader_data: dict) -> Dict[Symbol, List[Order]]:
        result = {}
        
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return result
            
        volcanic_rock = state.order_depths[Product.VOLCANIC_ROCK]
        volcanic_rock_mid = self.trader.get_swmid(volcanic_rock)
        
        if "price_history" not in trader_data:
            trader_data["price_history"] = {}
        if Product.VOLCANIC_ROCK not in trader_data["price_history"]:
            trader_data["price_history"][Product.VOLCANIC_ROCK] = []
            
        momentum = self.calculate_momentum(trader_data["price_history"][Product.VOLCANIC_ROCK])
        
        for strike in self.trader.voucher_strikes:
            voucher_name = f"VOLCANIC_ROCK_VOUCHER_{strike}"
            if voucher_name not in state.order_depths:
                continue
                
            voucher = state.order_depths[voucher_name]
            
            if momentum > self.params["threshold"]:
                # Strong upward momentum
                if abs(strike - volcanic_rock_mid) / volcanic_rock_mid < 0.02:
                    if voucher_name not in result:
                        result[voucher_name] = []
                    result[voucher_name].append(Order(voucher_name, max(voucher.buy_orders.keys()), 3))
                    logger.print(f"MOMENTUM TRADING: Buy {voucher_name} at {max(voucher.buy_orders.keys())}")
                elif strike < volcanic_rock_mid * 0.95:
                    if voucher_name not in result:
                        result[voucher_name] = []
                    result[voucher_name].append(Order(voucher_name, min(voucher.sell_orders.keys()), -2))
                    logger.print(f"MOMENTUM TRADING: Sell {voucher_name} at {min(voucher.sell_orders.keys())}")
                    
            elif momentum < -self.params["threshold"]:
                # Strong downward momentum
                if strike < volcanic_rock_mid * 0.9:
                    if voucher_name not in result:
                        result[voucher_name] = []
                    result[voucher_name].append(Order(voucher_name, max(voucher.buy_orders.keys()), 2))
                    logger.print(f"MOMENTUM TRADING: Buy {voucher_name} at {max(voucher.buy_orders.keys())}")
                elif abs(strike - volcanic_rock_mid) / volcanic_rock_mid < 0.02:
                    if voucher_name not in result:
                        result[voucher_name] = []
                    result[voucher_name].append(Order(voucher_name, min(voucher.sell_orders.keys()), -3))
                    logger.print(f"MOMENTUM TRADING: Sell {voucher_name} at {min(voucher.sell_orders.keys())}")
                    
        return result 