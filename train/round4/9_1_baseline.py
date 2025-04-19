import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# 全局参数配置 - 变种1：自适应高敏度因子交易策略
PARAMS = {
    # 基础参数
    'position_limit': 75,  # 仓位限制
    'conversion_limit': 10,  # 转换限制
    
    # 策略1（基本面+技术面）参数 - 增加权重和敏感度
    'sugar_weight': 0.5,  # 糖价权重 - 提高权重
    'sunlight_weight': 0.5,  # 阳光指数权重 - 提高权重
    'tariff_weight': 0.3,  # 关税权重 - 提高权重
    'ema_short_window': 5,  # 短期EMA窗口 - 比原版更短
    'ema_long_window': 20,   # 长期EMA窗口 - 比原版更短
    'ema_alpha_short': 0.4,  # 短期EMA平滑系数 - 高敏感度
    'ema_alpha_long': 0.2,   # 长期EMA平滑系数
    
    # 策略2（市场微观结构）参数 - 更敏感的市场结构分析
    'depth_threshold': 3,  # 订单簿深度阈值 - 降低以增加敏感度
    'spread_threshold': 0.05,  # 价差阈值 - 降低以增加交易频率
    'volume_threshold': 2,  # 交易量阈值 - 降低以增加敏感度
    'pressure_window': 5,  # 压力计算窗口 - 降低以更快响应
    'pressure_threshold': 0.6,  # 压力阈值 - 降低以增加交易机会
    
    # 策略选择参数
    'strategy_window': 5,  # 策略评估窗口 - 降低以更快调整策略
    'strategy_weight_threshold': 0.5,  # 策略权重阈值 - 降低以更容易触发策略
    
    # 交易参数 - 更激进的参数
    'max_trade_quantity': 25,  # 最大交易数量 - 提高以增加每次交易量
    'imbalance_threshold': 0.1,  # 订单簿不平衡阈值 - 降低以增加敏感度
    'price_score_neutral': 0.5,  # 价格得分中性值
    'price_score_threshold': 0.05,  # 价格得分阈值 - 降低以增加交易机会
    'spread_multiplier': 1.2,  # 价差倍数 - 降低以增加交易机会
    'flow_ratio': 1.2,  # 订单流比率 - 降低以增加交易机会
    'imbalance_ratio': 0.2,  # 订单簿不平衡比率 - 降低以增加敏感度
    
    # 新增波动率调整参数
    'volatility_window': 10,  # 波动率计算窗口
    'volatility_weight_factor': 1.5,  # 波动率对权重的影响因子
}

class Trader:
    def __init__(self):
        # 初始化参数
        self.position_limit = PARAMS['position_limit']
        self.conversion_limit = PARAMS['conversion_limit']
        
        # 策略1（基本面+技术面）参数
        self.sugar_weight = PARAMS['sugar_weight']
        self.sunlight_weight = PARAMS['sunlight_weight']
        self.tariff_weight = PARAMS['tariff_weight']
        self.ema_short_window = PARAMS['ema_short_window']
        self.ema_long_window = PARAMS['ema_long_window']
        self.ema_alpha_short = PARAMS['ema_alpha_short']
        self.ema_alpha_long = PARAMS['ema_alpha_long']
        
        # 策略2（市场微观结构）参数
        self.depth_threshold = PARAMS['depth_threshold']
        self.spread_threshold = PARAMS['spread_threshold']
        self.volume_threshold = PARAMS['volume_threshold']
        self.pressure_window = PARAMS['pressure_window']
        self.pressure_threshold = PARAMS['pressure_threshold']
        
        # 历史数据
        self.price_history = []
        self.sugar_history = []
        self.sunlight_history = []
        self.import_tariff_history = []
        self.buy_pressure_history = []
        self.sell_pressure_history = []
        self.spread_history = []
        self.imbalance_history = []
        
        # EMA历史
        self.price_ema_short = None
        self.price_ema_long = None
        
        # 策略选择参数
        self.strategy_window = PARAMS['strategy_window']
        self.strategy_scores = []
        
        # 波动率历史
        self.volatility_history = []
        self.volatility_window = PARAMS['volatility_window']
        
    def calculate_ema(self, current_value: float, previous_ema: float, alpha: float) -> float:
        """计算指数移动平均"""
        if previous_ema is None:
            return current_value
        return alpha * current_value + (1 - alpha) * previous_ema
    
    def normalize_value(self, value: float, history: List[float]) -> float:
        """标准化数值，处理除零情况"""
        if len(history) <= 1:
            return 0.5
            
        min_val = min(history)
        max_val = max(history)
        
        if max_val == min_val:
            return 0.5
            
        return (value - min_val) / (max_val - min_val)
    
    def calculate_volatility(self) -> float:
        """计算价格波动率"""
        if len(self.price_history) < self.volatility_window:
            return 0.0
        
        # 使用最近n个价格计算波动率
        recent_prices = self.price_history[-self.volatility_window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        return np.std(returns)
    
    def adjust_weights_by_volatility(self, base_weight: float, volatility: float) -> float:
        """根据波动率调整权重"""
        if not self.volatility_history:
            return base_weight
            
        avg_volatility = sum(self.volatility_history) / len(self.volatility_history)
        
        if avg_volatility == 0:
            return base_weight
            
        # 调整因子: 波动率越高，权重越大
        adjustment = (volatility / avg_volatility) * PARAMS['volatility_weight_factor']
        
        # 应用调整，但保持权重在合理范围内
        adjusted_weight = base_weight * adjustment
        return min(max(adjusted_weight, 0.1), 1.0)  # 限制在0.1到1.0之间
    
    def calculate_price_score(self, state: TradingState) -> float:
        """计算价格得分（策略1）- 使用波动率调整权重"""
        sugar_price = getattr(state.observations, 'sugarPrice', 0)
        sunlight_index = getattr(state.observations, 'sunlightIndex', 0)
        import_tariff = getattr(state.observations, 'importTariff', 0)
        
        self.sugar_history.append(sugar_price)
        self.sunlight_history.append(sunlight_index)
        self.import_tariff_history.append(import_tariff)
        
        # 计算当前波动率
        current_volatility = self.calculate_volatility()
        self.volatility_history.append(current_volatility)
        
        # 根据波动率调整权重
        adjusted_sugar_weight = self.adjust_weights_by_volatility(self.sugar_weight, current_volatility)
        adjusted_sunlight_weight = self.adjust_weights_by_volatility(self.sunlight_weight, current_volatility)
        adjusted_tariff_weight = self.adjust_weights_by_volatility(self.tariff_weight, current_volatility)
        
        sugar_score = self.normalize_value(sugar_price, self.sugar_history)
        sunlight_score = self.normalize_value(sunlight_index, self.sunlight_history)
        tariff_score = self.normalize_value(import_tariff, self.import_tariff_history)
        
        total_score = (sugar_score * adjusted_sugar_weight - 
                      sunlight_score * adjusted_sunlight_weight - 
                      tariff_score * adjusted_tariff_weight)
        
        # 归一化总分到0-1之间
        return max(0, min(1, total_score))
    
    def calculate_market_pressure(self, order_depth: OrderDepth) -> tuple[float, float]:
        """计算市场买卖压力（策略2）"""
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
    
    def analyze_order_flow(self, order_depth: OrderDepth) -> tuple[float, float]:
        """分析订单流（策略2）"""
        buy_flow = 0
        sell_flow = 0
        
        if order_depth.buy_orders:
            for price, quantity in order_depth.buy_orders.items():
                buy_flow += price * quantity
                
        if order_depth.sell_orders:
            for price, quantity in order_depth.sell_orders.items():
                sell_flow += price * quantity
                
        return buy_flow, sell_flow
    
    def calculate_spread_metrics(self, order_depth: OrderDepth) -> tuple[float, float]:
        """计算价差指标（策略2）"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0, 0
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        spread = best_ask - best_bid
        relative_spread = spread / best_bid
        
        return spread, relative_spread
    
    def analyze_order_book_imbalance(self, order_depth: OrderDepth) -> float:
        """分析订单簿不平衡度（策略2）"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        buy_volume = 0
        sell_volume = 0
        
        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True)[:self.depth_threshold]:
            buy_volume += price * quantity
            
        for price, quantity in sorted(order_depth.sell_orders.items())[:self.depth_threshold]:
            sell_volume += price * quantity
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0
            
        imbalance = (buy_volume - sell_volume) / total_volume
        return imbalance
    
    def evaluate_strategy_performance(self) -> tuple[float, float]:
        """评估两个策略的表现"""
        if len(self.price_history) < self.strategy_window:
            return 0.5, 0.5  # 初始时两个策略权重相等
            
        # 计算价格变化
        price_changes = np.diff(self.price_history[-self.strategy_window:])
        
        # 策略1得分：基于价格趋势
        strategy1_score = np.mean(price_changes) if len(price_changes) > 0 else 0
        
        # 策略2得分：基于市场压力变化
        if len(self.buy_pressure_history) >= self.strategy_window:
            pressure_changes = np.diff(self.buy_pressure_history[-self.strategy_window:])
            strategy2_score = np.mean(pressure_changes) if len(pressure_changes) > 0 else 0
        else:
            strategy2_score = 0
            
        # 标准化得分
        total_score = abs(strategy1_score) + abs(strategy2_score)
        if total_score > 0:
            strategy1_weight = abs(strategy1_score) / total_score
            strategy2_weight = abs(strategy2_score) / total_score
        else:
            strategy1_weight = 0.5
            strategy2_weight = 0.5
            
        return strategy1_weight, strategy2_weight
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        result = {}
        conversions = 0
        
        if "MAGNIFICENT_MACARONS" not in state.order_depths:
            return result, conversions, state.traderData
            
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        current_position = state.position.get("MAGNIFICENT_MACARONS", 0)
        
        # 获取当前价格
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.price_history.append(mid_price)
            
            # 更新EMA
            if self.price_ema_short is None:
                self.price_ema_short = mid_price
                self.price_ema_long = mid_price
            else:
                self.price_ema_short = self.calculate_ema(mid_price, self.price_ema_short, self.ema_alpha_short)
                self.price_ema_long = self.calculate_ema(mid_price, self.price_ema_long, self.ema_alpha_long)
        else:
            return result, conversions, state.traderData
            
        # 计算策略权重
        strategy1_weight, strategy2_weight = self.evaluate_strategy_performance()
        
        # 策略1：基本面+技术面分析
        price_score = self.calculate_price_score(state)
        
        # 策略2：市场微观结构分析
        buy_pressure, sell_pressure = self.calculate_market_pressure(order_depth)
        self.buy_pressure_history.append(buy_pressure)
        self.sell_pressure_history.append(sell_pressure)
        
        buy_flow, sell_flow = self.analyze_order_flow(order_depth)
        spread, relative_spread = self.calculate_spread_metrics(order_depth)
        self.spread_history.append(relative_spread)
        
        imbalance = self.analyze_order_book_imbalance(order_depth)
        self.imbalance_history.append(imbalance)
        
        # 确定交易方向
        orders = []
        
        # 策略1信号 - 使用EMA而非MA
        strategy1_buy = (price_score > 0.6 and self.price_ema_short > self.price_ema_long)
        strategy1_sell = (price_score < 0.4 and self.price_ema_short < self.price_ema_long)
        
        # 策略2信号
        strategy2_buy = (sell_pressure > self.pressure_threshold and
                        relative_spread < self.spread_threshold and
                        imbalance < -PARAMS['imbalance_ratio'] and
                        sell_flow > buy_flow * PARAMS['flow_ratio'])
        
        strategy2_sell = (buy_pressure > self.pressure_threshold and
                         relative_spread < self.spread_threshold and
                         imbalance > PARAMS['imbalance_ratio'] and
                         buy_flow > sell_flow * PARAMS['flow_ratio'])
        
        # 综合两个策略的信号 - 降低触发阈值使策略更激进
        if (strategy1_buy and strategy1_weight > PARAMS['strategy_weight_threshold']) or \
           (strategy2_buy and strategy2_weight > PARAMS['strategy_weight_threshold']):
            # 买入信号
            available_buy = self.position_limit - current_position
            if available_buy > 0:
                buy_price = best_ask
                # 更激进的仓位管理
                buy_quantity = min(available_buy, PARAMS['max_trade_quantity'])
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                
        elif (strategy1_sell and strategy1_weight > PARAMS['strategy_weight_threshold']) or \
             (strategy2_sell and strategy2_weight > PARAMS['strategy_weight_threshold']):
            # 卖出信号
            available_sell = self.position_limit + current_position
            if available_sell > 0:
                sell_price = best_bid
                # 更激进的仓位管理
                sell_quantity = min(available_sell, PARAMS['max_trade_quantity'])
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                
        # 平仓逻辑 - 降低平仓阈值，更激进
        elif (abs(imbalance) < PARAMS['imbalance_threshold'] and
              (relative_spread > self.spread_threshold * PARAMS['spread_multiplier'] or
               abs(price_score - PARAMS['price_score_neutral']) < PARAMS['price_score_threshold'])):
            
            if current_position > 0:
                sell_price = best_bid
                sell_quantity = min(current_position, PARAMS['max_trade_quantity'])
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
            elif current_position < 0:
                buy_price = best_ask
                buy_quantity = min(abs(current_position), PARAMS['max_trade_quantity'])
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
        
        result["MAGNIFICENT_MACARONS"] = orders
        return result, conversions, state.traderData 