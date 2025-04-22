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

# 产品配置 - 每个产品的适应性参数
PRODUCT_CONFIG = {
    # 高波动性产品
    "VOLCANIC_ROCK": {
        "trend_window": 30, "ml_weight": 0.7, "alpha": 0.15, "vol_threshold": 0.018,
        "position_pct": 0.4, "rsi_period": 14, "max_features": 10
    },
    "VOLCANIC_ROCK_VOUCHER_9500": {
        "trend_window": 25, "ml_weight": 0.65, "alpha": 0.12, "vol_threshold": 0.015,
        "position_pct": 0.35, "rsi_period": 14, "max_features": 8
    },
    "VOLCANIC_ROCK_VOUCHER_9750": {
        "trend_window": 25, "ml_weight": 0.65, "alpha": 0.12, "vol_threshold": 0.015,
        "position_pct": 0.35, "rsi_period": 14, "max_features": 8
    },
    "VOLCANIC_ROCK_VOUCHER_10000": {
        "trend_window": 25, "ml_weight": 0.65, "alpha": 0.12, "vol_threshold": 0.015,
        "position_pct": 0.35, "rsi_period": 14, "max_features": 8
    },
    "VOLCANIC_ROCK_VOUCHER_10250": {
        "trend_window": 25, "ml_weight": 0.65, "alpha": 0.12, "vol_threshold": 0.015,
        "position_pct": 0.35, "rsi_period": 14, "max_features": 8
    },
    "VOLCANIC_ROCK_VOUCHER_10500": {
        "trend_window": 25, "ml_weight": 0.65, "alpha": 0.12, "vol_threshold": 0.015,
        "position_pct": 0.35, "rsi_period": 14, "max_features": 8
    },
    
    # 中波动性产品
    "PICNIC_BASKET1": {
        "trend_window": 35, "ml_weight": 0.6, "alpha": 0.1, "vol_threshold": 0.012,
        "position_pct": 0.3, "rsi_period": 20, "max_features": 7
    },
    "PICNIC_BASKET2": {
        "trend_window": 35, "ml_weight": 0.6, "alpha": 0.1, "vol_threshold": 0.012,
        "position_pct": 0.3, "rsi_period": 20, "max_features": 7
    },
    "MAGNIFICENT_MACARONS": {
        "trend_window": 40, "ml_weight": 0.55, "alpha": 0.08, "vol_threshold": 0.01,
        "position_pct": 0.25, "rsi_period": 20, "max_features": 6
    },
    
    # 默认/低波动性产品
    "DEFAULT": {
        "trend_window": 50, "ml_weight": 0.4, "alpha": 0.05, "vol_threshold": 0.008,
        "position_pct": 0.2, "rsi_period": 25, "max_features": 5
    }
}

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

def get_product_config(product: str) -> dict:
    """获取产品配置或默认配置"""
    return PRODUCT_CONFIG.get(product, PRODUCT_CONFIG["DEFAULT"])

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
        # 特征集
        self.features = defaultdict(list)
        # 预测值
        self.predictions = defaultdict(float)
        # 预测准确度历史
        self.prediction_accuracy = defaultdict(list)
        # 特征权重
        self.feature_weights = defaultdict(np.ndarray)
        # 学习率
        self.learning_rate = 0.01
        # 上次交易时间戳
        self.last_trade_timestamp = defaultdict(int)
        
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
    
    def calculate_bollinger_bands(self, product: str, window: int = 20):
        """计算布林带指标"""
        prices = self.prices[product]
        
        if len(prices) < window:
            return
            
        # 截取最近window个价格
        recent_prices = prices[-window:]
        
        # 计算移动平均线和标准差
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        # 计算布林带上下轨
        upper_band = sma + 2 * std
        lower_band = sma - 2 * std
        
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
    
    def extract_features(self, product: str, config: dict):
        """提取交易特征"""
        max_features = config["max_features"]
        features = []
        
        # 添加价格特征
        if len(self.prices[product]) > 0:
            price = self.prices[product][-1]
            features.append(price)
            
            # 价格变化率
            if len(self.prices[product]) > 1:
                price_change = (price - self.prices[product][-2]) / self.prices[product][-2] if self.prices[product][-2] != 0 else 0
                features.append(price_change)
            else:
                features.append(0)
        else:
            features.extend([0, 0])  # 默认值
            
        # 添加RSI特征
        if self.rsi_values[product]:
            features.append(self.rsi_values[product][-1] / 100)  # 归一化到0-1
        else:
            features.append(0.5)  # 默认中性值
            
        # 添加布林带特征
        if product in self.bollinger_bands:
            bb = self.bollinger_bands[product]
            
            # 价格相对于布林带的位置 (0-1)
            price = self.prices[product][-1]
            bb_position = (price - bb["lower"]) / (bb["upper"] - bb["lower"]) if bb["upper"] != bb["lower"] else 0.5
            features.append(bb_position)
            
            # 布林带宽度
            features.append(bb["width"])
        else:
            features.extend([0.5, 0])  # 默认值
            
        # 添加波动率特征
        features.append(self.volatility[product])
        
        # 添加趋势强度特征
        features.append(self.trend_strength[product])
        
        # 确保特征数量一致，不超过最大特征数
        while len(features) < max_features:
            features.append(0)  # 填充
            
        # 截断特征到最大数量
        features = features[:max_features]
        
        # 保存特征
        self.features[product] = features
        
        return np.array(features)
    
    def initialize_weights(self, product: str, config: dict):
        """初始化特征权重"""
        max_features = config["max_features"]
        
        # 如果尚未初始化
        if product not in self.feature_weights or len(self.feature_weights[product]) != max_features:
            # 随机初始化权重
            weights = np.random.uniform(-0.1, 0.1, max_features)
            self.feature_weights[product] = weights
    
    def predict_price_movement(self, product: str, features: np.ndarray):
        """预测价格变动"""
        # 获取权重
        weights = self.feature_weights[product]
        
        # 确保权重和特征维度匹配
        if len(weights) != len(features):
            weights = np.random.uniform(-0.1, 0.1, len(features))
            self.feature_weights[product] = weights
            
        # 线性预测 (点积)
        prediction = np.dot(features, weights)
        
        # 使用Sigmoid函数将预测值映射到-1到1之间
        prediction = (2 / (1 + np.exp(-prediction))) - 1
        
        self.predictions[product] = prediction
        return prediction
    
    def update_model(self, product: str, prediction: float, actual: float):
        """更新模型权重"""
        # 计算误差
        error = actual - prediction
        
        # 记录预测准确度
        self.prediction_accuracy[product].append(abs(error))
        
        # 如果误差不大，不更新
        if abs(error) < 0.1:
            return
            
        # 获取特征
        features = np.array(self.features[product])
        
        # 获取权重
        weights = self.feature_weights[product]
        
        # 使用梯度下降更新权重
        gradient = error * features
        self.feature_weights[product] = weights + self.learning_rate * gradient
    
    def calculate_order_size(self, product: str, prediction: float, position: int, config: dict):
        """根据预测计算订单大小"""
        # 获取最大仓位
        max_position = int(POSITION_LIMITS[product] * config["position_pct"])
        
        # 根据预测的强度调整订单大小
        prediction_strength = abs(prediction)
        base_size = int(max_position * prediction_strength)
        
        # 确保至少为1
        size = max(1, base_size)
        
        # 考虑当前持仓，避免超过限制
        if prediction > 0:  # 买入信号
            size = min(size, POSITION_LIMITS[product] - position)
        else:  # 卖出信号
            size = min(size, POSITION_LIMITS[product] + position)
            
        return size
    
    def adaptive_ml_strategy(self, product: str, timestamp: int, depth: OrderDepth, position: int):
        """自适应机器学习策略"""
        # 获取产品配置
        config = get_product_config(product)
        
        # 更新价格历史
        if not self.update_price_history(product, depth):
            return []
            
        # 计算各种指标
        self.calculate_volatility(product)
        self.calculate_rsi(product, config)
        self.calculate_bollinger_bands(product)
        self.calculate_trend_strength(product, config)
        
        # 初始化权重
        self.initialize_weights(product, config)
        
        # 提取特征
        features = self.extract_features(product, config)
        
        # 预测价格变动
        prediction = self.predict_price_movement(product, features)
        
        # 冷却周期检查
        cooldown = 10  # 冷却时间
        if timestamp - self.last_trade_timestamp.get(product, 0) < cooldown:
            return []
            
        # 获取买卖价格
        bid, _, ask, _ = best_bid_ask(depth)
        if bid is None or ask is None:
            return []
            
        # 交易信号阈值
        signal_threshold = 0.3
        
        # 检查预测信号是否足够强
        if abs(prediction) < signal_threshold:
            return []  # 不进行交易
            
        # 计算订单大小
        size = self.calculate_order_size(product, prediction, position, config)
        
        # 如果大小为0，不交易
        if size == 0:
            return []
            
        orders = []
        
        # 执行交易
        if prediction > 0:  # 买入信号
            orders.append(Order(product, ask, size))
        else:  # 卖出信号
            orders.append(Order(product, bid, -size))
            
        # 更新最后交易时间
        if orders:
            self.last_trade_timestamp[product] = timestamp
            
        return orders
    
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
            
            # 应用自适应机器学习策略
            orders = self.adaptive_ml_strategy(
                product, state.timestamp, depth, position
            )
            
            if orders:
                result[product] = orders
                
        return result, 0, state.traderData 