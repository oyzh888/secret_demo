import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# 全局参数配置 - 更激进的变种1：高频+大仓位
PARAMS = {
    # 基础参数
    'position_limit': 75,  # 仓位限制
    'conversion_limit': 10,  # 转换限制
    
    # 策略1（基本面+技术面）参数
    'sugar_weight': 0.6,  # 糖价权重 - 进一步增加权重
    'sunlight_weight': 0.2,  # 阳光指数权重 - 进一步降低权重
    'tariff_weight': 0.2,  # 关税权重
    'ma_short_window': 3,  # 短期移动平均窗口 - 更短的窗口，更快速响应
    'ma_long_window': 15,   # 长期移动平均窗口 - 更短的窗口，更快速适应
    
    # 策略2（市场微观结构）参数
    'depth_threshold': 2,  # 订单簿深度阈值 - 进一步降低以更快响应
    'spread_threshold': 0.2,  # 价差阈值 - 进一步增加以更激进利用价差
    'volume_threshold': 1,  # 交易量阈值 - 最小值，更敏感
    'pressure_window': 3,  # 压力计算窗口 - 更短的窗口
    'pressure_threshold': 0.55,  # 压力阈值 - 进一步降低以更敏感
    
    # 策略选择参数
    'strategy_window': 3,  # 策略评估窗口 - 更短的窗口
    'strategy_weight_threshold': 0.5,  # 策略权重阈值 - 降低到0.5，两种策略机会平等
    
    # 交易参数
    'max_trade_quantity': 30,  # 最大交易数量 - 大幅增加以更激进的交易规模
    'imbalance_threshold': 0.1,  # 订单簿不平衡阈值 - 进一步降低以更敏感
    'price_score_neutral': 0.5,  # 价格得分中性值
    'price_score_threshold': 0.1,  # 价格得分阈值 - 降回原来的阈值，更激进
    'spread_multiplier': 1.1,  # 价差倍数 - 降低以更积极地交易
    'flow_ratio': 1.1,  # 订单流比率 - 降低以更敏感
    'imbalance_ratio': 0.15,  # 订单簿不平衡比率 - 进一步降低以更敏感
    
    # 新增参数
    'storage_cost': 0.1,  # 存储成本
    'min_profit_margin': 0.15,  # 最小利润边际 - 降低以更容易进行交易
    'max_spread': 9.0,  # 最大价差 - 假设最大价差略高
    'volatility_window': 8,  # 波动率计算窗口 - 略短一点
    'volatility_threshold': 0.003,  # 波动率阈值 - 降低阈值，更容易被判定为低波动
    'aggressive_factor': 1.5,  # 激进因子 - 用于调整交易规模
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
        
        # 历史数据
        self.price_history = []
        self.sugar_history = []
        self.sunlight_history = []
        self.import_tariff_history = []
        self.buy_pressure_history = []
        self.sell_pressure_history = []
        self.spread_history = []
        self.imbalance_history = []
        self.volatility_history = []
        
        # 策略选择参数
        self.strategy_scores = []
        
        # 记录最后一次交易信息
        self.last_trade_price = None
        self.last_trade_side = None
        self.consecutive_trades = 0
        
    def calculate_ma(self, prices: List[float], window: int) -> float:
        """计算移动平均"""
        if len(prices) < window:
            return prices[-1] if prices else 0
        return sum(prices[-window:]) / window
    
    def normalize_value(self, value: float, history: List[float]) -> float:
        """标准化数值，处理除零情况"""
        if len(history) <= 1:
            return 0.5
            
        min_val = min(history)
        max_val = max(history)
        
        if max_val == min_val:
            return 0.5
            
        return (value - min_val) / (max_val - min_val)
    
    def calculate_price_score(self, state: TradingState) -> float:
        """计算价格得分（策略1）"""
        sugar_price = getattr(state.observations, 'sugarPrice', 0)
        sunlight_index = getattr(state.observations, 'sunlightIndex', 0)
        import_tariff = getattr(state.observations, 'importTariff', 0)
        
        self.sugar_history.append(sugar_price)
        self.sunlight_history.append(sunlight_index)
        self.import_tariff_history.append(import_tariff)
        
        sugar_score = self.normalize_value(sugar_price, self.sugar_history)
        sunlight_score = self.normalize_value(sunlight_index, self.sunlight_history)
        tariff_score = self.normalize_value(import_tariff, self.import_tariff_history)
        
        # 更激进的计算方式，对tariff的变化做额外加权
        tariff_change = 0
        if len(self.import_tariff_history) > 1:
            tariff_change = import_tariff - self.import_tariff_history[-2]
        
        tariff_impact = 0.2 * np.sign(tariff_change) * min(1.0, abs(tariff_change) * 5)
        
        total_score = (sugar_score * self.sugar_weight - 
                      sunlight_score * self.sunlight_weight - 
                      tariff_score * self.tariff_weight + tariff_impact)
        
        # 将分数限制在0-1范围
        total_score = max(0, min(1, total_score))
        
        return total_score
    
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
        
        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True)[:PARAMS['depth_threshold']]:
            buy_volume += price * quantity
            
        for price, quantity in sorted(order_depth.sell_orders.items())[:PARAMS['depth_threshold']]:
            sell_volume += price * quantity
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0
            
        imbalance = (buy_volume - sell_volume) / total_volume
        return imbalance
    
    def evaluate_strategy_performance(self) -> tuple[float, float]:
        """评估两个策略的表现"""
        if len(self.price_history) < PARAMS['strategy_window']:
            return 0.5, 0.5  # 初始时两个策略权重相等
            
        # 计算价格变化
        price_changes = np.diff(self.price_history[-PARAMS['strategy_window']:])
        
        # 策略1得分：基于价格趋势
        strategy1_score = np.mean(price_changes) if len(price_changes) > 0 else 0
        
        # 策略2得分：基于市场压力变化
        if len(self.buy_pressure_history) >= PARAMS['strategy_window']:
            pressure_changes = np.diff(self.buy_pressure_history[-PARAMS['strategy_window']:])
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
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """计算价格波动率"""
        if len(prices) < 2:
            return 0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) if len(returns) > 0 else 0

    def calculate_storage_cost(self, position: int) -> float:
        """计算存储成本"""
        return abs(position) * PARAMS['storage_cost']

    def calculate_profit_margin(self, buy_price: float, sell_price: float) -> float:
        """计算利润边际"""
        return (sell_price - buy_price) / buy_price

    def should_convert(self, state: TradingState, product: str) -> bool:
        """判断是否应该进行转换"""
        if product not in state.position:
            return False
            
        position = state.position[product]
        if abs(position) < 2:  # 至少转换2个，否则不值得
            return False
            
        # 计算转换成本
        conversion_obs = getattr(state.observations, 'conversionObservations', {}).get(product)
        if not conversion_obs:
            return False
            
        total_cost = (conversion_obs.transportFees + 
                     conversion_obs.exportTariff + 
                     conversion_obs.importTariff +
                     self.calculate_storage_cost(position))
                     
        # 如果转换成本低于存储成本，考虑转换
        storage_cost = self.calculate_storage_cost(position)
        return total_cost < storage_cost * 1.2  # 更激进：只要接近就转换

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
            
            # 计算波动率
            volatility = self.calculate_volatility(self.price_history[-PARAMS['volatility_window']:] if len(self.price_history) >= PARAMS['volatility_window'] else self.price_history)
            self.volatility_history.append(volatility)
            
            # 根据波动率调整参数，更激进的策略
            if volatility > PARAMS['volatility_threshold']:
                # 高波动率时更进取 (与原来的逻辑相反)
                adjusted_max_trade = int(PARAMS['max_trade_quantity'] * PARAMS['aggressive_factor'])
                adjusted_spread_threshold = PARAMS['spread_threshold'] * 0.8
            else:
                # 低波动率时保持正常
                adjusted_max_trade = PARAMS['max_trade_quantity']
                adjusted_spread_threshold = PARAMS['spread_threshold']
        else:
            return result, conversions, state.traderData
            
        # 分析最近的交易以确定趋势
        recent_trades = state.market_trades.get("MAGNIFICENT_MACARONS", [])
        recent_own_trades = state.own_trades.get("MAGNIFICENT_MACARONS", [])
        
        # 分析市场交易趋势
        buy_volume = 0
        sell_volume = 0
        for trade in recent_trades:
            if trade.buyer and trade.buyer != "":
                buy_volume += trade.quantity
            if trade.seller and trade.seller != "":
                sell_volume += trade.quantity
                
        market_trend = np.sign(buy_volume - sell_volume) if (buy_volume != 0 or sell_volume != 0) else 0
        
        # 计算策略权重
        strategy1_weight, strategy2_weight = self.evaluate_strategy_performance()
        
        # 策略1：基本面+技术面分析
        price_score = self.calculate_price_score(state)
        ma_short = self.calculate_ma(self.price_history, PARAMS['ma_short_window'])
        ma_long = self.calculate_ma(self.price_history, PARAMS['ma_long_window'])
        
        # 策略2：市场微观结构分析
        buy_pressure, sell_pressure = self.calculate_market_pressure(order_depth)
        self.buy_pressure_history.append(buy_pressure)
        self.sell_pressure_history.append(sell_pressure)
        
        buy_flow, sell_flow = self.analyze_order_flow(order_depth)
        spread, relative_spread = self.calculate_spread_metrics(order_depth)
        self.spread_history.append(relative_spread)
        
        imbalance = self.analyze_order_book_imbalance(order_depth)
        self.imbalance_history.append(imbalance)
        
        # 计算趋势指标
        price_trend = 0
        if len(self.price_history) >= 3:
            price_trend = np.sign(self.price_history[-1] - self.price_history[-3])
        
        # 确定交易方向
        orders = []
        
        # 策略1信号 - 更激进的阈值
        strategy1_buy = (price_score > 0.55 and ma_short > ma_long) or (price_score > 0.65)
        strategy1_sell = (price_score < 0.45 and ma_short < ma_long) or (price_score < 0.35)
        
        # 策略2信号 - 更激进的条件
        strategy2_buy = (sell_pressure > PARAMS['pressure_threshold'] and
                        (relative_spread < adjusted_spread_threshold or imbalance < -PARAMS['imbalance_ratio']) and
                        (sell_flow > buy_flow * PARAMS['flow_ratio'] or market_trend > 0))
        
        strategy2_sell = (buy_pressure > PARAMS['pressure_threshold'] and
                         (relative_spread < adjusted_spread_threshold or imbalance > PARAMS['imbalance_ratio']) and
                         (buy_flow > sell_flow * PARAMS['flow_ratio'] or market_trend < 0))
        
        # 追踪趋势交易 - 新增的策略
        trend_buy = price_trend > 0 and market_trend > 0 and self.last_trade_side == "BUY"
        trend_sell = price_trend < 0 and market_trend < 0 and self.last_trade_side == "SELL"
        
        # 综合所有策略的信号 - 更激进地组合多种信号
        if ((strategy1_buy and strategy1_weight > PARAMS['strategy_weight_threshold']) or 
            (strategy2_buy and strategy2_weight > PARAMS['strategy_weight_threshold']) or
            trend_buy):
            # 买入信号
            available_buy = PARAMS['position_limit'] - current_position
            if available_buy > 0:
                buy_price = best_ask
                
                # 更激进的交易规模
                buy_quantity = min(available_buy, adjusted_max_trade)
                
                # 如果连续买入，逐步增加数量
                if self.last_trade_side == "BUY" and self.consecutive_trades > 0:
                    buy_quantity = min(available_buy, int(adjusted_max_trade * (1 + 0.1 * min(self.consecutive_trades, 5))))
                
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                self.last_trade_side = "BUY"
                self.last_trade_price = buy_price
                
                if self.last_trade_side == "BUY":
                    self.consecutive_trades += 1
                else:
                    self.consecutive_trades = 1
                
        elif ((strategy1_sell and strategy1_weight > PARAMS['strategy_weight_threshold']) or 
              (strategy2_sell and strategy2_weight > PARAMS['strategy_weight_threshold']) or
              trend_sell):
            # 卖出信号
            available_sell = PARAMS['position_limit'] + current_position
            if available_sell > 0:
                sell_price = best_bid
                
                # 更激进的交易规模
                sell_quantity = min(available_sell, adjusted_max_trade)
                
                # 如果连续卖出，逐步增加数量
                if self.last_trade_side == "SELL" and self.consecutive_trades > 0:
                    sell_quantity = min(available_sell, int(adjusted_max_trade * (1 + 0.1 * min(self.consecutive_trades, 5))))
                
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                self.last_trade_side = "SELL"
                self.last_trade_price = sell_price
                
                if self.last_trade_side == "SELL":
                    self.consecutive_trades += 1
                else:
                    self.consecutive_trades = 1
                
        # 平仓逻辑 - 更加激进的止盈止损
        elif (abs(imbalance) < PARAMS['imbalance_threshold'] and
              (relative_spread > adjusted_spread_threshold * PARAMS['spread_multiplier'] or
               abs(price_score - PARAMS['price_score_neutral']) < PARAMS['price_score_threshold'])):
            
            if current_position > 0:
                # 止盈止损条件：价格相对于上次交易已经变化超过2%
                if self.last_trade_price and (best_bid/self.last_trade_price - 1 > 0.02 or 1 - best_bid/self.last_trade_price > 0.015):
                    sell_price = best_bid
                    sell_quantity = min(current_position, adjusted_max_trade)
                    orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                    self.last_trade_side = "SELL"
                    self.last_trade_price = sell_price
                    self.consecutive_trades = 1
                    
            elif current_position < 0:
                # 止盈止损条件：价格相对于上次交易已经变化超过2%
                if self.last_trade_price and (1 - best_ask/self.last_trade_price > 0.02 or best_ask/self.last_trade_price - 1 > 0.015):
                    buy_price = best_ask
                    buy_quantity = min(abs(current_position), adjusted_max_trade)
                    orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                    self.last_trade_side = "BUY"
                    self.last_trade_price = buy_price
                    self.consecutive_trades = 1
        
        # 检查是否需要转换
        if self.should_convert(state, "MAGNIFICENT_MACARONS"):
            # 更激进的转换逻辑，尽可能转换更多
            conversions = min(abs(current_position), PARAMS['conversion_limit'])
            if current_position < 0:
                conversions = -conversions
        
        result["MAGNIFICENT_MACARONS"] = orders
        return result, conversions, str(state.timestamp)  # 使用时间戳作为trader data 