import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# 全局参数配置 - 更激进的参数设置
PARAMS = {
    # 基础参数 - 更大的仓位限制利用
    'position_limit': 75,  # 仓位限制
    'conversion_limit': 10,  # 转换限制
    'position_scale': 0.9,  # 仓位利用比例（更激进）
    
    # 策略1（基本面+技术面）参数
    'sugar_weight': 0.5,  # 糖价权重（调高）
    'sunlight_weight': 0.5,  # 阳光指数权重（调高）
    'tariff_weight': 0.3,  # 关税权重（调高）
    'ma_short_window': 5,  # 短期移动平均窗口（更短）
    'ma_long_window': 20,   # 长期移动平均窗口（更短）
    
    # 策略2（市场微观结构）参数
    'depth_threshold': 7,  # 订单簿深度阈值（增加）
    'spread_threshold': 0.05,  # 价差阈值（降低）
    'volume_threshold': 2,  # 交易量阈值（降低）
    'pressure_window': 8,  # 压力计算窗口（减少）
    'pressure_threshold': 0.6,  # 压力阈值（降低）
    
    # 策略选择参数
    'strategy_window': 8,  # 策略评估窗口（减少）
    'strategy_weight_threshold': 0.55,  # 策略权重阈值（降低）
    
    # 回归模型参数
    'regression_coefs': {
        'intercept': 187.6120,
        'sunlight': -3.3115,
        'sugar_price': 4.9708,
        'transport_fee': 61.5302,
        'export_tariff': -62.5394,
        'import_tariff': -52.0653
    },
    
    # 交易参数 - 更激进的设置
    'max_trade_quantity': 25,  # 最大交易数量（增加）
    'min_trade_quantity': 5,   # 最小交易数量（新增）
    'price_buffer': 3.0,  # 价格缓冲区间（减少）
    'imbalance_threshold': 0.15,  # 订单簿不平衡阈值（降低）
    'price_score_neutral': 0.5,  # 价格得分中性值
    'price_score_threshold': 0.05,  # 价格得分阈值（降低）
    'spread_multiplier': 1.2,  # 价差倍数（降低）
    'flow_ratio': 1.3,  # 订单流比率（降低）
    'imbalance_ratio': 0.25,  # 订单簿不平衡比率（降低）
    
    # 加仓参数（新增）
    'double_down_threshold': 8.0,  # 加仓阈值 
    'double_down_factor': 1.5,  # 加仓倍数
    
    # 止盈止损参数（新增）
    'take_profit_ticks': 12.0,  # 止盈点数
    'stop_loss_ticks': 20.0,  # 止损点数
}

class Trader:
    def __init__(self):
        # 初始化参数
        self.position_limit = PARAMS['position_limit']
        self.conversion_limit = PARAMS['conversion_limit']
        self.position_scale = PARAMS['position_scale']
        
        # 策略1（基本面+技术面）参数
        self.sugar_weight = PARAMS['sugar_weight']
        self.sunlight_weight = PARAMS['sunlight_weight']
        self.tariff_weight = PARAMS['tariff_weight']
        self.ma_short_window = PARAMS['ma_short_window']
        self.ma_long_window = PARAMS['ma_long_window']
        
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
        self.position_history = []
        self.entry_price = None  # 记录入场价格
        
        # 策略选择参数
        self.strategy_window = PARAMS['strategy_window']
        self.strategy_scores = []
        
        # 回归模型系数
        self.reg_coefs = PARAMS['regression_coefs']
        
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
        
        total_score = (sugar_score * self.sugar_weight - 
                      sunlight_score * self.sunlight_weight - 
                      tariff_score * self.tariff_weight)
        
        return total_score
    
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
    
    def should_take_profit(self, current_price, position) -> bool:
        """判断是否应该止盈"""
        if self.entry_price is None or position == 0:
            return False
        
        if position > 0 and current_price > self.entry_price + PARAMS['take_profit_ticks']:
            return True
        if position < 0 and current_price < self.entry_price - PARAMS['take_profit_ticks']:
            return True
        
        return False
    
    def should_stop_loss(self, current_price, position) -> bool:
        """判断是否应该止损"""
        if self.entry_price is None or position == 0:
            return False
        
        if position > 0 and current_price < self.entry_price - PARAMS['stop_loss_ticks']:
            return True
        if position < 0 and current_price > self.entry_price + PARAMS['stop_loss_ticks']:
            return True
        
        return False
    
    def calculate_position_size(self, current_position: int, is_buy: bool, price_distance: float) -> int:
        """计算仓位大小（考虑价格距离）"""
        available_position = self.position_limit * self.position_scale - abs(current_position)
        
        # 基础交易量
        base_quantity = min(PARAMS['max_trade_quantity'], available_position)
        if base_quantity <= 0:
            return 0
            
        # 根据价格距离调整交易量
        if price_distance > PARAMS['double_down_threshold']:
            base_quantity = min(base_quantity * PARAMS['double_down_factor'], 
                               PARAMS['max_trade_quantity'], 
                               available_position)
        
        # 保证最小交易量
        return max(int(base_quantity), PARAMS['min_trade_quantity'])
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        result = {}
        conversions = 0
        
        if "MAGNIFICENT_MACARONS" not in state.order_depths:
            return result, conversions, state.traderData
            
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        current_position = state.position.get("MAGNIFICENT_MACARONS", 0)
        
        # 记录当前仓位
        self.position_history.append(current_position)
        
        # 获取当前价格
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.price_history.append(mid_price)
        else:
            return result, conversions, state.traderData
            
        # 计算策略权重
        strategy1_weight, strategy2_weight = self.evaluate_strategy_performance()
        
        # 计算公平价格
        fair_price = self.estimate_fair_price(state)
        price_diff = fair_price - mid_price
        
        # 策略1：基本面+技术面分析
        price_score = self.calculate_price_score(state)
        ma_short = self.calculate_ma(self.price_history, self.ma_short_window)
        ma_long = self.calculate_ma(self.price_history, self.ma_long_window)
        
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
        
        # 止盈止损检查
        if self.should_take_profit(mid_price, current_position) or self.should_stop_loss(mid_price, current_position):
            if current_position > 0:
                # 平多仓
                sell_price = best_bid
                sell_quantity = current_position
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                self.entry_price = None
                return {"MAGNIFICENT_MACARONS": orders}, conversions, state.traderData
            elif current_position < 0:
                # 平空仓
                buy_price = best_ask
                buy_quantity = abs(current_position)
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                self.entry_price = None
                return {"MAGNIFICENT_MACARONS": orders}, conversions, state.traderData
        
        # 策略1信号
        strategy1_buy = price_score > 0.6 and ma_short > ma_long
        strategy1_sell = price_score < 0.4 and ma_short < ma_long
        
        # 策略2信号
        strategy2_buy = (sell_pressure > self.pressure_threshold and
                        relative_spread < self.spread_threshold and
                        imbalance < -PARAMS['imbalance_ratio'] and
                        sell_flow > buy_flow * PARAMS['flow_ratio'])
        
        strategy2_sell = (buy_pressure > self.pressure_threshold and
                         relative_spread < self.spread_threshold and
                         imbalance > PARAMS['imbalance_ratio'] and
                         buy_flow > sell_flow * PARAMS['flow_ratio'])
        
        # 回归模型信号
        regression_buy = price_diff > PARAMS['price_buffer']
        regression_sell = price_diff < -PARAMS['price_buffer']
        
        # 综合三个信号
        buy_signal = (regression_buy and 
                      ((strategy1_buy and strategy1_weight > PARAMS['strategy_weight_threshold']) or 
                       (strategy2_buy and strategy2_weight > PARAMS['strategy_weight_threshold'])))
                       
        sell_signal = (regression_sell and
                       ((strategy1_sell and strategy1_weight > PARAMS['strategy_weight_threshold']) or 
                        (strategy2_sell and strategy2_weight > PARAMS['strategy_weight_threshold'])))
        
        # 交易决策
        if buy_signal:
            # 买入信号
            available_buy = self.position_limit - current_position
            if available_buy > 0:
                buy_price = best_ask
                # 基于价格距离计算买入数量
                buy_quantity = self.calculate_position_size(current_position, True, abs(price_diff))
                if buy_quantity > 0:
                    orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                    # 记录入场价格（如果是新仓位）
                    if current_position <= 0:
                        self.entry_price = buy_price
                
        elif sell_signal:
            # 卖出信号
            available_sell = self.position_limit + current_position
            if available_sell > 0:
                sell_price = best_bid
                # 基于价格距离计算卖出数量
                sell_quantity = self.calculate_position_size(current_position, False, abs(price_diff))
                if sell_quantity > 0:
                    orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                    # 记录入场价格（如果是新仓位）
                    if current_position >= 0:
                        self.entry_price = sell_price
                
        # 平仓逻辑
        elif (abs(imbalance) < PARAMS['imbalance_threshold'] and
              (relative_spread > self.spread_threshold * PARAMS['spread_multiplier'] or
               abs(price_score - PARAMS['price_score_neutral']) < PARAMS['price_score_threshold'])):
            
            if current_position > 0:
                sell_price = best_bid
                sell_quantity = current_position
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                self.entry_price = None
            elif current_position < 0:
                buy_price = best_ask
                buy_quantity = abs(current_position)
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                self.entry_price = None
        
        result["MAGNIFICENT_MACARONS"] = orders
        return result, conversions, state.traderData 