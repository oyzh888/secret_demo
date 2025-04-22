from typing import Dict, List, Tuple, Optional
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np
import math

# 产品限制
POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75
}

# 策略配置和权重
STRATEGY_WEIGHTS = {
    # 高波动性产品
    "VOLCANIC_ROCK": {
        "volatility_scalping": 0.4,
        "countertrend": 0.3,
        "trend_following": 0.3
    },
    "VOLCANIC_ROCK_VOUCHER_9500": {
        "volatility_scalping": 0.3,
        "countertrend": 0.4,
        "trend_following": 0.3
    },
    "VOLCANIC_ROCK_VOUCHER_9750": {
        "volatility_scalping": 0.3,
        "countertrend": 0.4,
        "trend_following": 0.3
    },
    "VOLCANIC_ROCK_VOUCHER_10000": {
        "volatility_scalping": 0.3,
        "countertrend": 0.4,
        "trend_following": 0.3
    },
    "VOLCANIC_ROCK_VOUCHER_10250": {
        "volatility_scalping": 0.3,
        "countertrend": 0.4,
        "trend_following": 0.3
    },
    "VOLCANIC_ROCK_VOUCHER_10500": {
        "volatility_scalping": 0.3,
        "countertrend": 0.4,
        "trend_following": 0.3
    },
    
    # 中波动性产品
    "PICNIC_BASKET1": {
        "volatility_scalping": 0.2,
        "countertrend": 0.5,
        "trend_following": 0.3
    },
    "PICNIC_BASKET2": {
        "volatility_scalping": 0.2,
        "countertrend": 0.5,
        "trend_following": 0.3
    },
    "MAGNIFICENT_MACARONS": {
        "volatility_scalping": 0.2,
        "countertrend": 0.4,
        "trend_following": 0.4
    },
    
    # 默认/低波动性产品
    "DEFAULT": {
        "volatility_scalping": 0.1,
        "countertrend": 0.6,
        "trend_following": 0.3
    }
}

# 产品参数配置
PRODUCT_CONFIG = {
    # 高波动性产品
    "VOLCANIC_ROCK": {
        "bb_window": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
        "trend_window": 30, "vol_threshold": 0.018, "position_pct": 0.4, "z_threshold": 1.5
    },
    "VOLCANIC_ROCK_VOUCHER_9500": {
        "bb_window": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
        "trend_window": 25, "vol_threshold": 0.015, "position_pct": 0.35, "z_threshold": 1.5
    },
    "VOLCANIC_ROCK_VOUCHER_9750": {
        "bb_window": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
        "trend_window": 25, "vol_threshold": 0.015, "position_pct": 0.35, "z_threshold": 1.5
    },
    "VOLCANIC_ROCK_VOUCHER_10000": {
        "bb_window": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
        "trend_window": 25, "vol_threshold": 0.015, "position_pct": 0.35, "z_threshold": 1.5
    },
    "VOLCANIC_ROCK_VOUCHER_10250": {
        "bb_window": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
        "trend_window": 25, "vol_threshold": 0.015, "position_pct": 0.35, "z_threshold": 1.5
    },
    "VOLCANIC_ROCK_VOUCHER_10500": {
        "bb_window": 20, "bb_std": 2.0, "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
        "trend_window": 25, "vol_threshold": 0.015, "position_pct": 0.35, "z_threshold": 1.5
    },
    
    # 中波动性产品
    "PICNIC_BASKET1": {
        "bb_window": 25, "bb_std": 1.8, "rsi_period": 20, "rsi_overbought": 75, "rsi_oversold": 25,
        "trend_window": 35, "vol_threshold": 0.012, "position_pct": 0.3, "z_threshold": 1.3
    },
    "PICNIC_BASKET2": {
        "bb_window": 25, "bb_std": 1.8, "rsi_period": 20, "rsi_overbought": 75, "rsi_oversold": 25,
        "trend_window": 35, "vol_threshold": 0.012, "position_pct": 0.3, "z_threshold": 1.3
    },
    "MAGNIFICENT_MACARONS": {
        "bb_window": 30, "bb_std": 1.8, "rsi_period": 20, "rsi_overbought": 75, "rsi_oversold": 25,
        "trend_window": 40, "vol_threshold": 0.01, "position_pct": 0.25, "z_threshold": 1.3
    },
    
    # 默认/低波动性产品
    "DEFAULT": {
        "bb_window": 30, "bb_std": 1.5, "rsi_period": 25, "rsi_overbought": 80, "rsi_oversold": 20,
        "trend_window": 50, "vol_threshold": 0.008, "position_pct": 0.2, "z_threshold": 1.0
    }
}

def get_weights(product: str) -> dict:
    """获取产品的策略权重配置"""
    return STRATEGY_WEIGHTS.get(product, STRATEGY_WEIGHTS["DEFAULT"])

def get_product_config(product: str) -> dict:
    """获取产品配置或默认配置"""
    return PRODUCT_CONFIG.get(product, PRODUCT_CONFIG["DEFAULT"])

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """获取最优买卖价和数量"""
    bid_p = ask_p = bid_q = ask_q = None
    if depth.buy_orders:
        bid_p = max(depth.buy_orders.keys())
        bid_q = depth.buy_orders[bid_p]
    if depth.sell_orders:
        ask_p = min(depth.sell_orders.keys())
        ask_q = depth.sell_orders[ask_p]
    return bid_p, bid_q, ask_p, ask_q

def mid_price(depth: OrderDepth) -> Optional[float]:
    """计算中间价"""
    bid, _, ask, _ = best_bid_ask(depth)
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return None

class Trader:
    def __init__(self):
        # 价格历史
        self.prices = defaultdict(list)
        # 价格变动历史
        self.price_changes = defaultdict(list)
        # 波动率历史
        self.volatility = defaultdict(float)
        # RSI值历史
        self.rsi_values = defaultdict(list)
        # 布林带数据
        self.bollinger_bands = defaultdict(dict)
        # 趋势强度
        self.trend_strength = defaultdict(float)
        # 市场状态 (1=上升, -1=下降, 0=中性)
        self.market_state = defaultdict(int)
        # 上次交易时间戳
        self.last_trade_timestamp = defaultdict(int)
        # 策略性能跟踪
        self.strategy_performance = defaultdict(lambda: defaultdict(list))
        # 自适应权重
        self.adaptive_weights = defaultdict(dict)
        
    def update_price_history(self, product: str, depth: OrderDepth):
        """更新价格历史和计算基本指标"""
        # 获取中间价
        price = mid_price(depth)
        if price is None:
            return False
            
        # 更新价格历史
        if self.prices[product]:
            # 计算价格变化
            prev_price = self.prices[product][-1]
            price_change = (price - prev_price) / prev_price if prev_price != 0 else 0
            self.price_changes[product].append(price_change)
            
        self.prices[product].append(price)
        return True
    
    def calculate_volatility(self, product: str, window: int = 20):
        """计算价格波动率"""
        changes = self.price_changes[product]
        if len(changes) < window:
            return 0.0
            
        # 计算波动率 (标准差)
        recent_changes = changes[-window:]
        vol = np.std(recent_changes) * math.sqrt(window)  # 年化调整
        
        self.volatility[product] = vol
        return vol
    
    def calculate_rsi(self, product: str, config: dict):
        """计算RSI指标"""
        prices = self.prices[product]
        period = config["rsi_period"]
        
        if len(prices) < period + 1:
            self.rsi_values[product].append(50)  # 默认中性值
            return 50
            
        # 计算价格变动
        delta = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # 只使用最近的价格变动
        delta = delta[-period:]
        
        # 计算上涨和下跌
        gain = [max(0, d) for d in delta]
        loss = [max(0, -d) for d in delta]
        
        # 计算平均上涨和下跌
        avg_gain = sum(gain) / period
        avg_loss = sum(loss) / period
        
        # 计算相对强度
        if avg_loss == 0:
            rs = 100
        else:
            rs = avg_gain / avg_loss
            
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        self.rsi_values[product].append(rsi)
        return rsi
    
    def calculate_bollinger_bands(self, product: str, config: dict):
        """计算布林带指标"""
        prices = self.prices[product]
        window = config["bb_window"]
        num_std = config["bb_std"]
        
        if len(prices) < window:
            return
            
        # 截取最近window个价格
        recent_prices = prices[-window:]
        
        # 计算移动平均线和标准差
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        # 计算布林带上下轨
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std
        
        # 保存结果
        self.bollinger_bands[product] = {
            "sma": sma,
            "upper": upper_band,
            "lower": lower_band,
            "width": (upper_band - lower_band) / sma if sma != 0 else 0
        }
    
    def calculate_trend_strength(self, product: str, config: dict):
        """计算趋势强度"""
        prices = self.prices[product]
        window = config["trend_window"]
        
        if len(prices) < window:
            self.trend_strength[product] = 0
            self.market_state[product] = 0
            return 0
            
        # 获取最近的价格
        recent_prices = prices[-window:]
        
        # 计算价格变化
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        
        # 计算上涨和下跌的次数
        up_moves = sum(1 for change in price_changes if change > 0)
        down_moves = sum(1 for change in price_changes if change < 0)
        
        # 计算趋势强度 (-1到1之间)
        total_moves = up_moves + down_moves
        if total_moves == 0:
            strength = 0
        else:
            strength = (up_moves - down_moves) / total_moves
            
        self.trend_strength[product] = strength
        
        # 更新市场状态
        if strength > 0.3:
            self.market_state[product] = 1  # 上升
        elif strength < -0.3:
            self.market_state[product] = -1  # 下降
        else:
            self.market_state[product] = 0  # 中性
            
        return strength
        
    def calculate_z_score(self, product: str, price: float, config: dict):
        """计算Z分数"""
        if product not in self.bollinger_bands:
            return 0
            
        bb = self.bollinger_bands[product]
        sma = bb["sma"]
        std = (bb["upper"] - sma) / config["bb_std"]
        
        if std == 0:
            return 0
            
        return (price - sma) / std
    
    def volatility_scalping_strategy(self, product: str, timestamp: int, depth: OrderDepth, position: int, config: dict):
        """波动性套利策略"""
        # 获取价格
        price = mid_price(depth)
        if price is None:
            return []
            
        # 计算Z分数
        z_score = self.calculate_z_score(product, price, config)
        
        # 如果波动率低于阈值，不交易
        if self.volatility[product] < config["vol_threshold"]:
            return []
            
        # 获取买卖价格
        bid, _, ask, _ = best_bid_ask(depth)
        if bid is None or ask is None:
            return []
            
        # 交易大小
        max_position = int(POSITION_LIMITS[product] * config["position_pct"])
        size = max(1, int(max_position * min(1.0, abs(z_score) / config["z_threshold"])))
        
        orders = []
        
        # 基于Z分数的交易信号
        if z_score > config["z_threshold"]:  # 价格高于上轨，卖出信号
            # 检查持仓限制
            sell_size = min(size, POSITION_LIMITS[product] + position)
            if sell_size > 0:
                orders.append(Order(product, bid, -sell_size))
                
        elif z_score < -config["z_threshold"]:  # 价格低于下轨，买入信号
            # 检查持仓限制
            buy_size = min(size, POSITION_LIMITS[product] - position)
            if buy_size > 0:
                orders.append(Order(product, ask, buy_size))
                
        return orders
    
    def countertrend_strategy(self, product: str, timestamp: int, depth: OrderDepth, position: int, config: dict):
        """反趋势策略"""
        # 获取RSI值
        if not self.rsi_values[product]:
            return []
            
        rsi = self.rsi_values[product][-1]
        
        # 获取买卖价格
        bid, _, ask, _ = best_bid_ask(depth)
        if bid is None or ask is None:
            return []
            
        # 交易大小
        max_position = int(POSITION_LIMITS[product] * config["position_pct"])
        
        # 根据RSI值和趋势强度调整交易大小
        trend_factor = abs(self.trend_strength[product])
        size = max(1, int(max_position * trend_factor))
        
        orders = []
        
        # 根据RSI值进行交易
        if rsi > config["rsi_overbought"] and self.market_state[product] == 1:  # 超买 + 上升趋势
            # 检查持仓限制
            sell_size = min(size, POSITION_LIMITS[product] + position)
            if sell_size > 0:
                orders.append(Order(product, bid, -sell_size))
                
        elif rsi < config["rsi_oversold"] and self.market_state[product] == -1:  # 超卖 + 下降趋势
            # 检查持仓限制
            buy_size = min(size, POSITION_LIMITS[product] - position)
            if buy_size > 0:
                orders.append(Order(product, ask, buy_size))
                
        return orders
    
    def trend_following_strategy(self, product: str, timestamp: int, depth: OrderDepth, position: int, config: dict):
        """趋势跟踪策略"""
        # 获取趋势强度
        trend = self.trend_strength[product]
        
        # 如果趋势不够强，不交易
        if abs(trend) < 0.3:
            return []
            
        # 获取买卖价格
        bid, _, ask, _ = best_bid_ask(depth)
        if bid is None or ask is None:
            return []
            
        # 交易大小 - 根据趋势强度动态调整
        max_position = int(POSITION_LIMITS[product] * config["position_pct"])
        size = max(1, int(max_position * abs(trend)))
        
        orders = []
        
        # 根据趋势方向交易
        if trend > 0:  # 上升趋势，买入
            # 检查持仓限制
            buy_size = min(size, POSITION_LIMITS[product] - position)
            if buy_size > 0:
                orders.append(Order(product, ask, buy_size))
                
        elif trend < 0:  # 下降趋势，卖出
            # 检查持仓限制
            sell_size = min(size, POSITION_LIMITS[product] + position)
            if sell_size > 0:
                orders.append(Order(product, bid, -sell_size))
                
        return orders
    
    def update_strategy_weights(self, product: str):
        """更新策略权重 (基于最近的性能)"""
        performances = self.strategy_performance[product]
        
        # 至少有5个数据点才进行更新
        if all(len(perf) >= 5 for perf in performances.values()):
            total_perf = {}
            
            # 计算每个策略的平均性能
            for strategy, perf_list in performances.items():
                # 只取最近的10个性能数据
                recent_perf = perf_list[-10:]
                total_perf[strategy] = sum(recent_perf) / len(recent_perf)
                
            # 如果所有性能都为零，使用默认权重
            if sum(total_perf.values()) == 0:
                self.adaptive_weights[product] = get_weights(product)
            else:
                # 归一化性能作为新权重
                total = sum(total_perf.values())
                if total > 0:
                    self.adaptive_weights[product] = {
                        strategy: perf / total for strategy, perf in total_perf.items()
                    }
                else:
                    # 如果总性能为负，取反并归一化
                    adjusted_perf = {
                        strategy: max(0.1, 1 + perf) for strategy, perf in total_perf.items()
                    }
                    total = sum(adjusted_perf.values())
                    self.adaptive_weights[product] = {
                        strategy: perf / total for strategy, perf in adjusted_perf.items()
                    }
    
    def ensemble_strategy(self, product: str, timestamp: int, depth: OrderDepth, position: int):
        """集成策略 - 结合多种交易策略"""
        # 获取产品配置
        config = get_product_config(product)
        
        # 更新价格历史
        if not self.update_price_history(product, depth):
            return []
            
        # 计算各种指标
        self.calculate_volatility(product)
        self.calculate_rsi(product, config)
        self.calculate_bollinger_bands(product, config)
        self.calculate_trend_strength(product, config)
        
        # 冷却周期检查
        cooldown = 5  # 冷却时间
        if timestamp - self.last_trade_timestamp.get(product, 0) < cooldown:
            return []
            
        # 获取策略权重 - 优先使用自适应权重
        if product in self.adaptive_weights:
            weights = self.adaptive_weights[product]
        else:
            weights = get_weights(product)
            
        # 运行各个策略获取订单建议
        vol_orders = self.volatility_scalping_strategy(product, timestamp, depth, position, config)
        counter_orders = self.countertrend_strategy(product, timestamp, depth, position, config)
        trend_orders = self.trend_following_strategy(product, timestamp, depth, position, config)
        
        # 如果所有策略都没有订单，则返回空列表
        if not vol_orders and not counter_orders and not trend_orders:
            return []
            
        # 如果只有一个策略有订单，直接使用该策略
        active_strategies = []
        if vol_orders:
            active_strategies.append(("volatility_scalping", vol_orders))
        if counter_orders:
            active_strategies.append(("countertrend", counter_orders))
        if trend_orders:
            active_strategies.append(("trend_following", trend_orders))
            
        if len(active_strategies) == 1:
            self.last_trade_timestamp[product] = timestamp
            # 记录使用了哪个策略
            strategy_name = active_strategies[0][0]
            self.strategy_performance[product][strategy_name].append(1.0)
            # 为其他策略记录0分
            for strat in weights.keys():
                if strat != strategy_name:
                    self.strategy_performance[product][strat].append(0.0)
            return active_strategies[0][1]
            
        # 选择权重最高的策略
        strategy_tuples = [(name, weight, orders) 
                         for (name, orders), weight in zip(active_strategies, 
                                                         [weights[s[0]] for s in active_strategies])]
        
        best_strategy = max(strategy_tuples, key=lambda x: x[1])
        
        # 记录使用了哪个策略
        self.strategy_performance[product][best_strategy[0]].append(1.0)
        # 为其他策略记录0分
        for strat in weights.keys():
            if strat != best_strategy[0]:
                self.strategy_performance[product][strat].append(0.0)
                
        # 更新权重
        self.update_strategy_weights(product)
        
        # 更新最后交易时间
        self.last_trade_timestamp[product] = timestamp
        
        return best_strategy[2]
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result = {}
        
        # 处理每个产品
        for product, depth in state.order_depths.items():
            # 跳过没有限制的产品
            if product not in POSITION_LIMITS:
                continue
                
            # 获取当前仓位
            position = state.position.get(product, 0)
            
            # 应用集成策略
            orders = self.ensemble_strategy(
                product, state.timestamp, depth, position
            )
            
            if orders:
                result[product] = orders
                
        return result, 0, state.traderData 