from datamodel import OrderDepth, UserId, TradingState, Order, ConversionObservation
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math

class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"

# Aggressive Strategy 3 (Most aggressive, with minimal risk controls)
AGGRESSIVE_PARAMS_3 = {
    Product.VOLCANIC_ROCK: {
        "default_spread_mean": 1.49,
        "default_spread_std": 0.50,
        "spread_std_window": 10,
        "zscore_threshold": 0.5,  # Extremely low threshold
        "target_position": 0,
        "take_width": 0.2,  # Very tight spread
        "clear_width": 0.5,  # Very tight spread
        "adverse_volume": 5,
        "max_trade_volume": 400,  # Maximum volume
    },
    Product.VOLCANIC_ROCK_VOUCHER_10500: {
        "default_spread_mean": 1.00,
        "default_spread_std": 0.03,
        "spread_std_window": 10,
        "zscore_threshold": 0.6,  # Extremely low threshold
        "target_position": 0,
        "take_width": 0.2,  # Very tight spread
        "clear_width": 0.5,  # Very tight spread
        "adverse_volume": 5,
        "max_trade_volume": 200,  # Maximum volume
    },
    Product.VOLCANIC_ROCK_VOUCHER_10250: {
        "default_spread_mean": 1.01,
        "default_spread_std": 0.07,
        "spread_std_window": 10,
        "zscore_threshold": 0.5,  # Extremely low threshold
        "target_position": 0,
        "take_width": 0.2,  # Very tight spread
        "clear_width": 0.5,  # Very tight spread
        "adverse_volume": 5,
        "max_trade_volume": 200,  # Maximum volume
    },
    Product.VOLCANIC_ROCK_VOUCHER_10000: {
        "default_spread_mean": 1.03,
        "default_spread_std": 0.16,
        "spread_std_window": 10,
        "zscore_threshold": 0.5,  # Extremely low threshold
        "target_position": 0,
        "take_width": 0.2,  # Very tight spread
        "clear_width": 0.5,  # Very tight spread
        "adverse_volume": 5,
        "max_trade_volume": 200,  # Maximum volume
    },
    Product.VOLCANIC_ROCK_VOUCHER_9750: {
        "default_spread_mean": 1.06,
        "default_spread_std": 0.25,
        "spread_std_window": 10,
        "zscore_threshold": 0.4,  # Extremely low threshold
        "target_position": 0,
        "take_width": 0.2,  # Very tight spread
        "clear_width": 0.5,  # Very tight spread
        "adverse_volume": 5,
        "max_trade_volume": 200,  # Maximum volume
    },
    Product.VOLCANIC_ROCK_VOUCHER_9500: {
        "default_spread_mean": 1.11,
        "default_spread_std": 0.31,
        "spread_std_window": 10,
        "zscore_threshold": 0.3,  # Extremely low threshold
        "target_position": 0,
        "take_width": 0.2,  # Very tight spread
        "clear_width": 0.5,  # Very tight spread
        "adverse_volume": 5,
        "max_trade_volume": 200,  # Maximum volume
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = AGGRESSIVE_PARAMS_3
        self.params = params

        # 设置持仓限制
        self.LIMIT = {
            Product.VOLCANIC_ROCK: 400,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 200,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 200,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
        }

        # 设置行权价
        self.STRIKE_PRICES = {
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10500,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
            Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
        }

        # 初始化历史数据存储
        self.price_history = {}
        self.volatility_history = {}
        self.risk_free_rate = 0.0  # 无风险利率
        self.last_mid_prices = {}
        self.timestamps = {}
        self.fair_value_estimates = {}
        self.realized_pnl = {}
        self.trades_completed = {}

    def normal_cdf(self, x: float) -> float:
        """标准正态分布的累积分布函数"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def normal_pdf(self, x: float) -> float:
        """标准正态分布的概率密度函数"""
        return math.exp(-x**2 / 2.0) / math.sqrt(2.0 * math.pi)

    def get_swmid(self, order_depth) -> float:
        """计算加权中间价"""
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)

    def calculate_black_scholes_price(self, spot_price: float, strike_price: float, time_to_maturity: float, 
                                    volatility: float, is_call: bool = True) -> float:
        """计算Black-Scholes期权价格"""
        # 处理边界情况
        if time_to_maturity <= 0:
            return max(0, spot_price - strike_price) if is_call else max(0, strike_price - spot_price)
            
        # 确保波动率和时间不为0
        volatility = max(volatility, 0.0001)  # 设置最小波动率
        time_to_maturity = max(time_to_maturity, 1/252)  # 设置最小时间（1天）
        
        # 计算d1和d2
        d1 = (math.log(spot_price / strike_price) + (self.risk_free_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * math.sqrt(time_to_maturity))
        d2 = d1 - volatility * math.sqrt(time_to_maturity)

        if is_call:
            price = spot_price * self.normal_cdf(d1) - strike_price * math.exp(-self.risk_free_rate * time_to_maturity) * self.normal_cdf(d2)
        else:
            price = strike_price * math.exp(-self.risk_free_rate * time_to_maturity) * self.normal_cdf(-d2) - spot_price * self.normal_cdf(-d1)

        return price

    def calculate_historical_volatility(self, prices: List[float], window: int = 20) -> float:
        """计算历史波动率"""
        if len(prices) < 2:
            return 0.0

        returns = np.diff(np.log(prices))
        if len(returns) < window:
            return np.std(returns) * math.sqrt(252)  # 年化波动率
        return np.std(returns[-window:]) * math.sqrt(252)

    def calculate_time_to_maturity(self, product: str, timestamp: int) -> float:
        """计算到期时间（以年为单位）
        所有期权在7天后到期（1轮=1天），从round 1开始，round 1的day=0
        """
        try:
            current_round = 3 # 获取round
            current_day = 2    # 获取day
            
            # 计算剩余天数
            remaining_days = 7 - (current_round - 1) - current_day
            
            # 转换为年（假设一年有252个交易日）
            remaining_years = max(remaining_days / 252.0, 1/252)  # 最小为1天
            
            return remaining_years
        except:
            return 7/252  # 如果解析失败，返回默认值7天

    def run(self, state: TradingState):
        """主交易逻辑"""
        result = {}
        conversions = 0

        # 获取现货价格
        spot_price = self.get_swmid(state.order_depths[Product.VOLCANIC_ROCK]) if Product.VOLCANIC_ROCK in state.order_depths else None
        if spot_price is None:
            return result, conversions, state.traderData

        # 更新价格历史
        if Product.VOLCANIC_ROCK not in self.price_history:
            self.price_history[Product.VOLCANIC_ROCK] = []
        self.price_history[Product.VOLCANIC_ROCK].append(spot_price)

        # 计算历史波动率
        historical_volatility = self.calculate_historical_volatility(self.price_history[Product.VOLCANIC_ROCK])

        # 处理每个期权合约
        for product in state.order_depths.keys():
            if "VOUCHER" not in product:
                continue

            if product not in state.order_depths:
                continue

            # 获取期权中间价
            option_price = self.get_swmid(state.order_depths[product])
            if option_price is None:
                continue
                
            # 计算到期时间
            time_to_maturity = self.calculate_time_to_maturity(product, state.timestamp)

            # 获取行权价
            strike_price = self.STRIKE_PRICES[product]

            # 计算理论价格
            theoretical_price = self.calculate_black_scholes_price(
                spot_price, strike_price, time_to_maturity, historical_volatility
            )

            # 计算价格偏差，添加保护措施
            if theoretical_price is None or theoretical_price == 0:
                price_deviation = 0.0
            else:
                price_deviation = (option_price - theoretical_price) / theoretical_price

            # 获取当前持仓
            position = state.position.get(product, 0)

            # 根据价格偏差进行交易，使用产品特定的参数
            if price_deviation > 0.001 and theoretical_price > 0:  # 最激进的阈值
                # 卖出期权
                quantity = min(self.params[product]["max_trade_volume"], self.LIMIT[product] + position)
                if quantity > 0:
                    best_bid = max(state.order_depths[product].buy_orders.keys())
                    if product not in result:
                        result[product] = []
                    result[product].append(Order(product, best_bid, -quantity))
            elif price_deviation < -0.001 and theoretical_price > 0:  # 最激进的阈值
                # 买入期权
                quantity = min(self.params[product]["max_trade_volume"], self.LIMIT[product] - position)
                if quantity > 0:
                    best_ask = min(state.order_depths[product].sell_orders.keys())
                    if product not in result:
                        result[product] = []
                    result[product].append(Order(product, best_ask, quantity))

        return result, conversions, state.traderData 