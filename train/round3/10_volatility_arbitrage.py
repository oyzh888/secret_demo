from datamodel import OrderDepth, TradingState, Order, Symbol
from typing import Dict, List
import numpy as np
from scipy.optimize import bisect
from scipy.stats import norm
import math

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

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.params = params
        self.voucher_strikes = [9500, 9750, 10000, 10250, 10500]
        
    def get_swmid(self, order_depth: OrderDepth) -> float:
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        return None

class VolatilityArbitrageStrategy:
    def __init__(self, trader):
        self.trader = trader
        self.params = {
            # 基础配置
            "initial_capital": 100000,
            "risk_per_trade": 0.3,
            "max_drawdown": 0.4,
            
            # 波动率曲面参数
            "ttm": 1.0,  # 到期时间（年）
            "r": 0.0,    # 无风险利率
            "iv_threshold": 0.05,  # 波动率偏离阈值
            
            # 价差配置
            "spreads": [
                {
                    "name": "skew_spread_9500_10250",
                    "low_strike": 9500,
                    "high_strike": 10250,
                    "position_percentage": 0.3
                },
                {
                    "name": "skew_spread_9750_10500",
                    "low_strike": 9750,
                    "high_strike": 10500,
                    "position_percentage": 0.3
                }
            ],
            
            # 风险控制
            "stop_loss_percentage": 0.1,
            "max_position_per_strike": 20,
            "max_total_position": 40
        }
        
    def black_scholes(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        d1 = (math.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if option_type == "call":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
    def implied_volatility(self, price: float, S: float, K: float, T: float, r: float, option_type: str) -> float:
        def objective(sigma):
            return self.black_scholes(S, K, T, r, sigma, option_type) - price
            
        try:
            return bisect(objective, 0.001, 5.0)
        except:
            return None
            
    def calculate_moneyness(self, S: float, K: float, T: float) -> float:
        return math.log(K/S) / math.sqrt(T)
        
    def fit_volatility_surface(self, S: float, voucher_prices: Dict[int, float]) -> tuple:
        m_values = []
        v_values = []
        
        for strike, price in voucher_prices.items():
            m = self.calculate_moneyness(S, strike, self.params["ttm"])
            v = self.implied_volatility(price, S, strike, self.params["ttm"], self.params["r"], "call")
            
            if v is not None:
                m_values.append(m)
                v_values.append(v)
                
        if len(m_values) < 3:
            return None, None
            
        # 二次回归拟合
        coeffs = np.polyfit(m_values, v_values, 2)
        fitted_curve = np.poly1d(coeffs)
        
        return fitted_curve, coeffs
        
    def calculate_position_size(self, spread_config: dict, available_capital: float, current_price: float) -> int:
        position_value = available_capital * spread_config["position_percentage"]
        max_profit = spread_config["high_strike"] - spread_config["low_strike"]
        position_size = int(position_value / max_profit)
        
        risk_limit = int(available_capital * self.params["risk_per_trade"] / max_profit)
        position_size = min(position_size, risk_limit)
        
        return position_size
        
    def execute_spread(self, spread_config: dict, current_price: float,
                      lower_voucher: OrderDepth, higher_voucher: OrderDepth,
                      available_capital: float) -> Dict[Symbol, List[Order]]:
        result = {}
        
        position_size = self.calculate_position_size(spread_config, available_capital, current_price)
        
        if position_size <= 0:
            return result
            
        # 执行价差交易
        if lower_voucher.symbol not in result:
            result[lower_voucher.symbol] = []
        result[lower_voucher.symbol].append(Order(
            lower_voucher.symbol,
            max(lower_voucher.buy_orders.keys()),
            position_size
        ))
        
        if higher_voucher.symbol not in result:
            result[higher_voucher.symbol] = []
        result[higher_voucher.symbol].append(Order(
            higher_voucher.symbol,
            min(higher_voucher.sell_orders.keys()),
            -position_size
        ))
        
        logger.print(f"SKEW SPREAD {spread_config['name']}: Buy {lower_voucher.symbol} and Sell {higher_voucher.symbol} with size {position_size}")
            
        return result
        
    def run(self, state: TradingState, trader_data: dict) -> Dict[Symbol, List[Order]]:
        result = {}
        
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return result
            
        # 获取当前价格
        current_price = self.trader.get_swmid(state.order_depths[Product.VOLCANIC_ROCK])
        if current_price is None:
            return result
            
        # 收集所有voucher价格
        voucher_prices = {}
        for strike in self.trader.voucher_strikes:
            voucher = state.order_depths.get(f"VOLCANIC_ROCK_VOUCHER_{strike}")
            if voucher:
                mid_price = self.trader.get_swmid(voucher)
                if mid_price:
                    voucher_prices[strike] = mid_price
                    
        # 拟合波动率曲面
        fitted_curve, coeffs = self.fit_volatility_surface(current_price, voucher_prices)
        if fitted_curve is None:
            return result
            
        # 计算每个voucher的波动率偏离
        mispricing_scores = {}
        for strike, price in voucher_prices.items():
            m = self.calculate_moneyness(current_price, strike, self.params["ttm"])
            v = self.implied_volatility(price, current_price, strike, self.params["ttm"], self.params["r"], "call")
            if v is not None:
                fitted_v = fitted_curve(m)
                mispricing_scores[strike] = (v - fitted_v) / fitted_v
                
        # 执行价差交易
        available_capital = self.params["initial_capital"]
        if "capital" in trader_data:
            available_capital = trader_data["capital"]
            
        for spread_config in self.params["spreads"]:
            lower_strike = spread_config["low_strike"]
            higher_strike = spread_config["high_strike"]
            
            if lower_strike in mispricing_scores and higher_strike in mispricing_scores:
                # 如果低价voucher被低估且高价voucher被高估，执行价差
                if (mispricing_scores[lower_strike] < -self.params["iv_threshold"] and 
                    mispricing_scores[higher_strike] > self.params["iv_threshold"]):
                    lower_voucher = state.order_depths.get(f"VOLCANIC_ROCK_VOUCHER_{lower_strike}")
                    higher_voucher = state.order_depths.get(f"VOLCANIC_ROCK_VOUCHER_{higher_strike}")
                    
                    if lower_voucher and higher_voucher:
                        spread_orders = self.execute_spread(
                            spread_config,
                            current_price,
                            lower_voucher,
                            higher_voucher,
                            available_capital
                        )
                        for symbol, orders in spread_orders.items():
                            if symbol not in result:
                                result[symbol] = []
                            result[symbol].extend(orders)
                            
        return result 