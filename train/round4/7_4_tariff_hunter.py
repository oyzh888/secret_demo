import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# 全局参数配置 - 变种3：关税猎手，专注于关税变化和基本面冲击带来的机会
PARAMS = {
    # 基础参数
    'position_limit': 75,  # 仓位限制
    'conversion_limit': 10,  # 转换限制
    
    # 策略1（基本面+技术面）参数 - 提高关税权重
    'sugar_weight': 0.3,  # 糖价权重
    'sunlight_weight': 0.2,  # 阳光指数权重
    'tariff_weight': 0.5,  # 关税权重 - 大幅增加权重
    'ma_short_window': 3,  # 短期移动平均窗口 - 非常短的窗口以快速反应
    'ma_long_window': 8,   # 长期移动平均窗口 - 也较短
    
    # 策略2（市场微观结构）参数
    'depth_threshold': 3,  # 订单簿深度阈值
    'spread_threshold': 0.15,  # 价差阈值
    'volume_threshold': 2,  # 交易量阈值
    'pressure_window': 3,  # 压力计算窗口
    'pressure_threshold': 0.55,  # 压力阈值
    
    # 关税相关参数 - 新增
    'tariff_change_threshold': 0.01,  # 关税变化阈值
    'tariff_reaction_multiplier': 2.0,  # 关税反应乘数
    'tariff_shock_window': 3,  # 关税冲击窗口
    'sugar_change_threshold': 0.015,  # 糖价变化阈值
    'sunlight_change_threshold': 0.02,  # 阳光指数变化阈值
    
    # 策略选择参数
    'strategy_window': 3,  # 策略评估窗口 - 短窗口以快速适应
    'strategy_weight_threshold': 0.45,  # 策略权重阈值
    
    # 交易参数
    'max_trade_quantity': 30,  # 最大交易数量 - 进取的规模
    'imbalance_threshold': 0.1,  # 订单簿不平衡阈值
    'price_score_neutral': 0.5,  # 价格得分中性值
    'price_score_threshold': 0.15,  # 价格得分阈值
    'spread_multiplier': 1.2,  # 价差倍数
    'flow_ratio': 1.1,  # 订单流比率
    'imbalance_ratio': 0.15,  # 订单簿不平衡比率
    
    # 冲击交易参数 - 新增
    'shock_trade_multiplier': 1.5,  # 冲击交易量倍数
    'shock_trade_threshold': 0.25,  # 冲击交易阈值
    'shock_trade_timeout': 5,  # 冲击交易超时
    
    # 止盈止损参数 - 新增
    'take_profit_threshold': 0.03,  # 止盈阈值 - 3%就止盈
    'stop_loss_threshold': 0.02,  # 止损阈值 - 2%就止损
    
    # 新增参数
    'storage_cost': 0.1,  # 存储成本
    'min_profit_margin': 0.15,  # 最小利润边际
    'max_spread': 8.8,  # 最大价差
    'volatility_window': 5,  # 波动率计算窗口 - 更短以快速反应
    'volatility_threshold': 0.0035,  # 波动率阈值
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
        self.export_tariff_history = []
        self.transport_fees_history = []
        self.buy_pressure_history = []
        self.sell_pressure_history = []
        self.spread_history = []
        self.imbalance_history = []
        self.volatility_history = []
        
        # 状态追踪
        self.in_shock_trade = False
        self.shock_trade_timer = 0
        self.shock_trade_direction = 0  # 1:买入, -1:卖出
        self.tariff_change_detected = False
        self.tariff_change_direction = 0
        self.last_tariff_update = 0
        
        # 策略选择参数
        self.strategy_scores = []
        
        # 记录最后一次交易信息
        self.last_trade_price = None
        self.last_trade_side = None
        self.last_trade_timestamp = 0
        self.consecutive_trades = 0
        
        # 性能追踪
        self.trades_count = 0
        self.profitable_trades = 0
        self.trade_returns = []
        
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
    
    def detect_fundamental_change(self, timestamp: int) -> tuple[bool, float, str]:
        """检测基本面变化"""
        # 至少需要2个数据点来检测变化
        if (len(self.import_tariff_history) < 2 or 
            len(self.sugar_history) < 2 or 
            len(self.sunlight_history) < 2):
            return False, 0, ""
            
        changes = []
        sources = []
        
        # 检测关税变化
        import_tariff_change = self.import_tariff_history[-1] - self.import_tariff_history[-2]
        import_tariff_pct_change = abs(import_tariff_change / self.import_tariff_history[-2]) if self.import_tariff_history[-2] != 0 else 0
        
        if import_tariff_pct_change > PARAMS['tariff_change_threshold']:
            changes.append((import_tariff_pct_change, np.sign(import_tariff_change)))
            sources.append("import_tariff")
            
        # 检测糖价变化
        sugar_change = self.sugar_history[-1] - self.sugar_history[-2]
        sugar_pct_change = abs(sugar_change / self.sugar_history[-2]) if self.sugar_history[-2] != 0 else 0
        
        if sugar_pct_change > PARAMS['sugar_change_threshold']:
            changes.append((sugar_pct_change, np.sign(sugar_change)))
            sources.append("sugar")
            
        # 检测阳光指数变化
        sunlight_change = self.sunlight_history[-1] - self.sunlight_history[-2]
        sunlight_pct_change = abs(sunlight_change / self.sunlight_history[-2]) if self.sunlight_history[-2] != 0 else 0
        
        if sunlight_pct_change > PARAMS['sunlight_change_threshold']:
            changes.append((sunlight_pct_change, -np.sign(sunlight_change)))  # 阳光增加通常会导致价格下降
            sources.append("sunlight")
            
        # 如果检测到变化，返回最大变化及其方向
        if changes:
            max_idx = np.argmax([c[0] for c in changes])
            change_magnitude = changes[max_idx][0]
            change_direction = changes[max_idx][1]
            change_source = sources[max_idx]
            
            # 更新状态
            self.in_shock_trade = True
            self.shock_trade_timer = PARAMS['shock_trade_timeout']
            self.shock_trade_direction = change_direction
            self.last_tariff_update = timestamp
            
            return True, change_magnitude * change_direction, change_source
        
        return False, 0, ""
    
    def calculate_price_score(self, state: TradingState) -> float:
        """计算价格得分（策略1）- 加强对关税变化的反应"""
        sugar_price = getattr(state.observations, 'sugarPrice', 0)
        sunlight_index = getattr(state.observations, 'sunlightIndex', 0)
        import_tariff = getattr(state.observations, 'importTariff', 0)
        export_tariff = getattr(state.observations, 'exportTariff', 0)
        transport_fees = getattr(state.observations, 'transportFees', 0)
        
        self.sugar_history.append(sugar_price)
        self.sunlight_history.append(sunlight_index)
        self.import_tariff_history.append(import_tariff)
        self.export_tariff_history.append(export_tariff)
        self.transport_fees_history.append(transport_fees)
        
        sugar_score = self.normalize_value(sugar_price, self.sugar_history)
        sunlight_score = self.normalize_value(sunlight_index, self.sunlight_history)
        tariff_score = self.normalize_value(import_tariff, self.import_tariff_history)
        
        # 计算关税变化
        tariff_change = 0
        if len(self.import_tariff_history) > 1:
            tariff_change = import_tariff - self.import_tariff_history[-2]
            tariff_pct_change = abs(tariff_change / self.import_tariff_history[-2]) if self.import_tariff_history[-2] != 0 else 0
            
            if tariff_pct_change > PARAMS['tariff_change_threshold']:
                self.tariff_change_detected = True
                self.tariff_change_direction = np.sign(tariff_change)
                self.last_tariff_update = state.timestamp
            elif state.timestamp - self.last_tariff_update > PARAMS['tariff_shock_window']:
                self.tariff_change_detected = False
        
        # 计算关税冲击影响
        tariff_impact = 0.5 * np.sign(tariff_change) * min(1.0, abs(tariff_change) * 10)
        
        # 考虑糖价和关税的相互作用
        sugar_tariff_interaction = sugar_score * tariff_score * 0.2
        
        total_score = (sugar_score * self.sugar_weight - 
                      sunlight_score * self.sunlight_weight - 
                      tariff_score * self.tariff_weight + 
                      tariff_impact - 
                      sugar_tariff_interaction)
        
        # 加入基本面冲击的额外权重
        if self.in_shock_trade and self.shock_trade_timer > 0:
            shock_factor = min(1.0, self.shock_trade_timer / PARAMS['shock_trade_timeout'])
            total_score += 0.3 * self.shock_trade_direction * shock_factor
            
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
            # 初始时，由于我们关注关税冲击，倾向于策略1
            return 0.7, 0.3
            
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
            
        # 如果检测到关税冲击，提高策略1的权重
        if self.in_shock_trade and self.shock_trade_timer > 0:
            shock_factor = min(1.0, self.shock_trade_timer / PARAMS['shock_trade_timeout'])
            strategy1_score *= (1 + shock_factor)
            
        # 标准化得分
        total_score = abs(strategy1_score) + abs(strategy2_score)
        if total_score > 0:
            strategy1_weight = abs(strategy1_score) / total_score
            strategy2_weight = abs(strategy2_score) / total_score
        else:
            # 默认时更注重策略1
            strategy1_weight = 0.6
            strategy2_weight = 0.4
            
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
        if abs(position) < 3:  # 至少转换3个，否则不值得
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
        
        # 如果处于冲击交易中，更倾向于持有仓位
        if self.in_shock_trade and self.shock_trade_timer > 0:
            return total_cost < storage_cost * 0.9  # 更高的转换门槛
        
        return total_cost < storage_cost * 1.1  # 正常情况下的转换门槛
    
    def should_take_profit_or_stop_loss(self, current_price: float, position: int) -> tuple[bool, str]:
        """检查是否应该止盈或止损"""
        if self.last_trade_price is None or position == 0:
            return False, ""
            
        if position > 0:  # 多头仓位
            profit_pct = (current_price - self.last_trade_price) / self.last_trade_price
            if profit_pct >= PARAMS['take_profit_threshold']:
                return True, "take_profit"
            elif profit_pct <= -PARAMS['stop_loss_threshold']:
                return True, "stop_loss"
        else:  # 空头仓位
            profit_pct = (self.last_trade_price - current_price) / self.last_trade_price
            if profit_pct >= PARAMS['take_profit_threshold']:
                return True, "take_profit"
            elif profit_pct <= -PARAMS['stop_loss_threshold']:
                return True, "stop_loss"
                
        return False, ""

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
            
            # 根据波动率调整参数
            if volatility > PARAMS['volatility_threshold']:
                # 高波动率时更激进 - 适合关税冲击策略
                adjusted_max_trade = int(PARAMS['max_trade_quantity'] * 1.3)
                adjusted_spread_threshold = PARAMS['spread_threshold'] * 0.9
            else:
                adjusted_max_trade = PARAMS['max_trade_quantity']
                adjusted_spread_threshold = PARAMS['spread_threshold']
        else:
            return result, conversions, state.traderData
            
        # 检测基本面变化
        fundamental_change, change_magnitude, change_source = self.detect_fundamental_change(state.timestamp)
        
        # 如果在冲击交易中，减少计时器
        if self.in_shock_trade and self.shock_trade_timer > 0:
            self.shock_trade_timer -= 1
            if self.shock_trade_timer == 0:
                self.in_shock_trade = False
        
        # 分析最近的交易以确定趋势
        recent_trades = state.market_trades.get("MAGNIFICENT_MACARONS", [])
        
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
        
        # 检查止盈止损条件
        should_close, close_reason = self.should_take_profit_or_stop_loss(mid_price, current_position)
        
        if should_close:
            # 平仓操作
            if current_position > 0:
                sell_price = best_bid
                sell_quantity = min(current_position, adjusted_max_trade)
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                self.last_trade_side = "SELL"
                self.last_trade_price = sell_price
                self.last_trade_timestamp = state.timestamp
                self.consecutive_trades = 1
                
                # 记录交易绩效
                if close_reason == "take_profit":
                    self.profitable_trades += 1
                self.trades_count += 1
                
            elif current_position < 0:
                buy_price = best_ask
                buy_quantity = min(abs(current_position), adjusted_max_trade)
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                self.last_trade_side = "BUY"
                self.last_trade_price = buy_price
                self.last_trade_timestamp = state.timestamp
                self.consecutive_trades = 1
                
                # 记录交易绩效
                if close_reason == "take_profit":
                    self.profitable_trades += 1
                self.trades_count += 1
        
        # 如果有基本面冲击，优先考虑关税冲击交易策略
        elif fundamental_change and abs(change_magnitude) > PARAMS['shock_trade_threshold']:
            # 关税冲击交易
            if change_magnitude > 0:  # 正面冲击，价格可能上涨
                available_buy = PARAMS['position_limit'] - current_position
                if available_buy > 0:
                    buy_price = best_ask
                    # 根据冲击幅度决定交易量
                    shock_ratio = min(1.0, abs(change_magnitude) / PARAMS['shock_trade_threshold'])
                    buy_quantity = min(available_buy, int(adjusted_max_trade * PARAMS['shock_trade_multiplier'] * shock_ratio))
                    orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                    self.last_trade_side = "BUY"
                    self.last_trade_price = buy_price
                    self.last_trade_timestamp = state.timestamp
                    self.consecutive_trades = 1
            else:  # 负面冲击，价格可能下跌
                available_sell = PARAMS['position_limit'] + current_position
                if available_sell > 0:
                    sell_price = best_bid
                    # 根据冲击幅度决定交易量
                    shock_ratio = min(1.0, abs(change_magnitude) / PARAMS['shock_trade_threshold'])
                    sell_quantity = min(available_sell, int(adjusted_max_trade * PARAMS['shock_trade_multiplier'] * shock_ratio))
                    orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                    self.last_trade_side = "SELL"
                    self.last_trade_price = sell_price
                    self.last_trade_timestamp = state.timestamp
                    self.consecutive_trades = 1
        
        # 常规交易策略
        else:
            # 策略1信号 - 注重关税因素
            strategy1_buy = (price_score > 0.6 and ma_short > ma_long) or price_score > 0.7
            strategy1_sell = (price_score < 0.4 and ma_short < ma_long) or price_score < 0.3
            
            # 如果正在进行冲击交易，使用更激进的信号
            if self.in_shock_trade and self.shock_trade_timer > 0:
                if self.shock_trade_direction > 0:
                    strategy1_buy = strategy1_buy or price_score > 0.55
                else:
                    strategy1_sell = strategy1_sell or price_score < 0.45
            
            # 策略2信号
            strategy2_buy = (sell_pressure > PARAMS['pressure_threshold'] and
                           relative_spread < adjusted_spread_threshold and
                           imbalance < -PARAMS['imbalance_ratio'] and
                           sell_flow > buy_flow * PARAMS['flow_ratio'])
            
            strategy2_sell = (buy_pressure > PARAMS['pressure_threshold'] and
                            relative_spread < adjusted_spread_threshold and
                            imbalance > PARAMS['imbalance_ratio'] and
                            buy_flow > sell_flow * PARAMS['flow_ratio'])
            
            # 综合信号
            should_buy = ((strategy1_buy and strategy1_weight > PARAMS['strategy_weight_threshold']) or 
                         (strategy2_buy and strategy2_weight > PARAMS['strategy_weight_threshold']) or
                         (self.in_shock_trade and self.shock_trade_timer > 0 and self.shock_trade_direction > 0))
            
            should_sell = ((strategy1_sell and strategy1_weight > PARAMS['strategy_weight_threshold']) or 
                          (strategy2_sell and strategy2_weight > PARAMS['strategy_weight_threshold']) or
                          (self.in_shock_trade and self.shock_trade_timer > 0 and self.shock_trade_direction < 0))
            
            if should_buy:
                # 买入信号
                available_buy = PARAMS['position_limit'] - current_position
                if available_buy > 0:
                    buy_price = best_ask
                    
                    # 如果是冲击交易，使用更大的交易量
                    if self.in_shock_trade and self.shock_trade_timer > 0 and self.shock_trade_direction > 0:
                        shock_factor = min(1.0, self.shock_trade_timer / PARAMS['shock_trade_timeout'])
                        buy_quantity = min(available_buy, int(adjusted_max_trade * (1 + shock_factor)))
                    else:
                        buy_quantity = min(available_buy, adjusted_max_trade)
                    
                    orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                    self.last_trade_side = "BUY"
                    self.last_trade_price = buy_price
                    self.last_trade_timestamp = state.timestamp
                    
                    # 连续交易计数
                    if self.last_trade_side == "BUY":
                        self.consecutive_trades += 1
                    else:
                        self.consecutive_trades = 1
                    
            elif should_sell:
                # 卖出信号
                available_sell = PARAMS['position_limit'] + current_position
                if available_sell > 0:
                    sell_price = best_bid
                    
                    # 如果是冲击交易，使用更大的交易量
                    if self.in_shock_trade and self.shock_trade_timer > 0 and self.shock_trade_direction < 0:
                        shock_factor = min(1.0, self.shock_trade_timer / PARAMS['shock_trade_timeout'])
                        sell_quantity = min(available_sell, int(adjusted_max_trade * (1 + shock_factor)))
                    else:
                        sell_quantity = min(available_sell, adjusted_max_trade)
                    
                    orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                    self.last_trade_side = "SELL"
                    self.last_trade_price = sell_price
                    self.last_trade_timestamp = state.timestamp
                    
                    # 连续交易计数
                    if self.last_trade_side == "SELL":
                        self.consecutive_trades += 1
                    else:
                        self.consecutive_trades = 1
                    
            # 平仓逻辑 - 保持原有逻辑，但如果在冲击交易中不主动平仓
            elif not self.in_shock_trade and (abs(imbalance) < PARAMS['imbalance_threshold'] and
                  (relative_spread > adjusted_spread_threshold * PARAMS['spread_multiplier'] or
                   abs(price_score - PARAMS['price_score_neutral']) < PARAMS['price_score_threshold'])):
                
                if current_position > 0:
                    sell_price = best_bid
                    sell_quantity = min(current_position, adjusted_max_trade)
                    orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                    self.last_trade_side = "SELL"
                    self.last_trade_price = sell_price
                    self.last_trade_timestamp = state.timestamp
                    self.consecutive_trades = 1
                    
                elif current_position < 0:
                    buy_price = best_ask
                    buy_quantity = min(abs(current_position), adjusted_max_trade)
                    orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                    self.last_trade_side = "BUY"
                    self.last_trade_price = buy_price
                    self.last_trade_timestamp = state.timestamp
                    self.consecutive_trades = 1
        
        # 检查是否需要转换
        if self.should_convert(state, "MAGNIFICENT_MACARONS"):
            conversions = min(abs(current_position), PARAMS['conversion_limit'])
            if current_position < 0:
                conversions = -conversions
        
        result["MAGNIFICENT_MACARONS"] = orders
        
        # 返回一些性能指标作为trader data，以便跟踪
        trader_data = {
            "timestamp": state.timestamp,
            "trades_count": self.trades_count,
            "profitable_ratio": self.profitable_trades / self.trades_count if self.trades_count > 0 else 0,
            "position": current_position,
            "in_shock_trade": int(self.in_shock_trade),
            "shock_timer": self.shock_trade_timer
        }
        
        return result, conversions, str(trader_data) 