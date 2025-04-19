import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# 全局参数配置 - 重点市场微观结构
PARAMS = {
    # 基础参数
    'position_limit': 75,  # 仓位限制
    'conversion_limit': 10,  # 转换限制
    
    # 回归模型参数
    'regression_coefs': {
        'intercept': 187.6120,
        'sunlight': -3.3115,
        'sugar_price': 4.9708,
        'transport_fee': 61.5302,
        'export_tariff': -62.5394,
        'import_tariff': -52.0653
    },
    
    # 市场微观结构参数 - 更激进设置
    'depth_threshold': 8,  # 订单簿深度阈值（增加）
    'book_pressure_levels': [3, 5, 8],  # 多级订单簿压力检测
    'level_weights': [0.6, 0.3, 0.1],  # 各级权重
    'imbalance_threshold': 0.12,  # 订单簿不平衡阈值（更低）
    'extreme_imbalance': 0.25,  # 极端不平衡阈值
    'spread_threshold': 0.05,  # 价差阈值（更低）
    'momentum_window': 3,  # 动量窗口（短期）
    'momentum_threshold': 0.01,  # 动量阈值（低）
    'volume_ratio_threshold': 1.5,  # 交易量比率阈值
    
    # 微观信号组合参数
    'signal_count_threshold': 2,  # 需要的信号数量
    'max_signal_lookback': 3,  # 信号最大回看次数
    
    # 订单流分析参数
    'flow_imbalance_threshold': 0.15,  # 订单流不平衡阈值
    'price_cluster_threshold': 0.4,  # 价格集群阈值
    'tape_speed_threshold': 3,  # 行情速度阈值
    
    # 订单执行参数
    'max_trade_quantity': 20,  # 最大交易数量
    'aggressive_quantity_multiplier': 1.5,  # 进攻性交易倍数
    'price_improvement': 0.1,  # 价格改进（激进交易）
    'scaling_factor': 0.8,  # 缩放因子
    
    # 多级定单策略
    'order_levels': 3,  # 订单级别数
    'level_distances': [0, 0.5, 1.0],  # 各级距离
    'level_quantities': [0.6, 0.3, 0.1],  # 各级数量比例
    
    # 止盈止损参数
    'take_profit_ticks': 10,  # 止盈点数
    'stop_loss_ticks': 15,   # 止损点数
    'trailing_stop_activation': 5,  # 移动止损激活点数
    'trailing_stop_distance': 8,  # 移动止损距离
}

class Trader:
    def __init__(self):
        # 初始化参数
        self.position_limit = PARAMS['position_limit']
        self.conversion_limit = PARAMS['conversion_limit']
        
        # 历史数据
        self.price_history = []
        self.mid_price_history = []
        self.bid_history = []
        self.ask_history = []
        self.volume_history = []
        self.spread_history = []
        self.depth_imbalance_history = []
        self.order_flow_imbalance_history = []
        
        # 仓位管理
        self.position_history = []
        self.entry_price = None
        self.highest_since_entry = None
        self.lowest_since_entry = None
        self.trailing_stop_active = False
        self.trailing_stop_level = None
        
        # 回归模型系数
        self.reg_coefs = PARAMS['regression_coefs']
        
        # 微观结构信号缓存
        self.micro_signals = {
            'imbalance': [],
            'momentum': [],
            'spread': [],
            'flow': [],
            'volume': []
        }
    
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
    
    def calculate_market_price(self, order_depth: OrderDepth) -> tuple:
        """计算市场价格数据"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None, None, None, 0
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        return mid_price, best_bid, best_ask, spread
    
    def analyze_order_book_multi_level(self, order_depth: OrderDepth) -> dict:
        """多级分析订单簿"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return {
                'total_imbalance': 0,
                'level_imbalances': [],
                'buy_pressure': 0,
                'sell_pressure': 0
            }
            
        results = {
            'level_imbalances': [],
            'total_imbalance': 0,
            'buy_pressure': 0,
            'sell_pressure': 0
        }
        
        total_buy_volume = 0
        total_sell_volume = 0
        weighted_imbalance = 0
        
        # 分析不同深度级别
        for i, level in enumerate(PARAMS['book_pressure_levels']):
            buy_volume = 0
            sell_volume = 0
            
            # 计算买单量
            for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True)[:level]:
                buy_volume += quantity
                
            # 计算卖单量
            for price, quantity in sorted(order_depth.sell_orders.items())[:level]:
                sell_volume += quantity
                
            # 该级别的不平衡度
            if buy_volume + sell_volume > 0:
                level_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
            else:
                level_imbalance = 0
                
            # 添加到结果
            results['level_imbalances'].append(level_imbalance)
            
            # 以权重累加
            weighted_imbalance += level_imbalance * PARAMS['level_weights'][i]
            
            # 累计总量
            total_buy_volume += buy_volume
            total_sell_volume += sell_volume
            
        # 计算总体不平衡
        if total_buy_volume + total_sell_volume > 0:
            results['total_imbalance'] = weighted_imbalance
            results['buy_pressure'] = total_buy_volume / (total_buy_volume + total_sell_volume)
            results['sell_pressure'] = total_sell_volume / (total_buy_volume + total_sell_volume)
        
        return results
    
    def calculate_order_flow_imbalance(self) -> float:
        """计算订单流不平衡"""
        if len(self.bid_history) < 2 or len(self.ask_history) < 2:
            return 0
            
        # 计算最近的价格变化
        bid_changes = np.diff(self.bid_history[-PARAMS['momentum_window']-1:])
        ask_changes = np.diff(self.ask_history[-PARAMS['momentum_window']-1:])
        
        # 计算上涨和下跌的次数
        up_moves = sum(1 for x in bid_changes if x > 0) + sum(1 for x in ask_changes if x > 0)
        down_moves = sum(1 for x in bid_changes if x < 0) + sum(1 for x in ask_changes if x < 0)
        
        # 计算不平衡
        total_moves = up_moves + down_moves
        if total_moves > 0:
            return (up_moves - down_moves) / total_moves
        return 0
    
    def calculate_price_momentum(self) -> float:
        """计算价格动量"""
        if len(self.mid_price_history) < PARAMS['momentum_window'] + 1:
            return 0
            
        prices = self.mid_price_history[-PARAMS['momentum_window']-1:]
        price_changes = np.diff(prices)
        
        # 使用指数加权计算动量（更注重最近的变化）
        weights = np.exp(np.linspace(0, 1, len(price_changes)))
        weights = weights / np.sum(weights)
        
        momentum = np.sum(price_changes * weights)
        return momentum
    
    def detect_volume_surge(self) -> bool:
        """检测交易量激增"""
        if len(self.volume_history) < 5:
            return False
            
        recent_volume = np.mean(self.volume_history[-3:])
        prior_volume = np.mean(self.volume_history[-8:-3])
        
        if prior_volume > 0 and recent_volume / prior_volume > PARAMS['volume_ratio_threshold']:
            return True
        return False
    
    def should_take_profit(self, current_price: float, position: int) -> bool:
        """止盈逻辑"""
        if self.entry_price is None or position == 0:
            return False
            
        if position > 0 and current_price > self.entry_price + PARAMS['take_profit_ticks']:
            return True
        if position < 0 and current_price < self.entry_price - PARAMS['take_profit_ticks']:
            return True
        
        return False
    
    def should_stop_loss(self, current_price: float, position: int) -> bool:
        """止损逻辑"""
        if self.entry_price is None or position == 0:
            return False
            
        # 检查是否应该激活移动止损
        if not self.trailing_stop_active:
            if position > 0 and current_price > self.entry_price + PARAMS['trailing_stop_activation']:
                self.trailing_stop_active = True
                self.trailing_stop_level = current_price - PARAMS['trailing_stop_distance']
            elif position < 0 and current_price < self.entry_price - PARAMS['trailing_stop_activation']:
                self.trailing_stop_active = True
                self.trailing_stop_level = current_price + PARAMS['trailing_stop_distance']
        
        # 如果移动止损已激活，则更新止损水平
        if self.trailing_stop_active:
            if position > 0:
                if current_price > self.trailing_stop_level + PARAMS['trailing_stop_distance']:
                    self.trailing_stop_level = current_price - PARAMS['trailing_stop_distance']
                if current_price < self.trailing_stop_level:
                    return True
            elif position < 0:
                if current_price < self.trailing_stop_level - PARAMS['trailing_stop_distance']:
                    self.trailing_stop_level = current_price + PARAMS['trailing_stop_distance']
                if current_price > self.trailing_stop_level:
                    return True
        # 常规止损
        else:
            if position > 0 and current_price < self.entry_price - PARAMS['stop_loss_ticks']:
                return True
            if position < 0 and current_price > self.entry_price + PARAMS['stop_loss_ticks']:
                return True
        
        return False
    
    def get_trade_signals(self, book_analysis: dict, fair_price: float, mid_price: float, spread: float) -> tuple:
        """获取综合交易信号"""
        # 初始化信号计数
        buy_signals = 0
        sell_signals = 0
        
        # 订单簿不平衡信号
        if book_analysis['total_imbalance'] > PARAMS['imbalance_threshold']:
            buy_signals += 1
            self.micro_signals['imbalance'].append(1)  # 买入信号
        elif book_analysis['total_imbalance'] < -PARAMS['imbalance_threshold']:
            sell_signals += 1
            self.micro_signals['imbalance'].append(-1)  # 卖出信号
        else:
            self.micro_signals['imbalance'].append(0)  # 无信号
            
        # 极端不平衡（更强信号）
        if book_analysis['total_imbalance'] > PARAMS['extreme_imbalance']:
            buy_signals += 1  # 额外信号
        elif book_analysis['total_imbalance'] < -PARAMS['extreme_imbalance']:
            sell_signals += 1  # 额外信号
            
        # 价格动量信号
        momentum = self.calculate_price_momentum()
        if momentum > PARAMS['momentum_threshold']:
            buy_signals += 1
            self.micro_signals['momentum'].append(1)
        elif momentum < -PARAMS['momentum_threshold']:
            sell_signals += 1
            self.micro_signals['momentum'].append(-1)
        else:
            self.micro_signals['momentum'].append(0)
            
        # 订单流不平衡信号
        flow_imbalance = self.calculate_order_flow_imbalance()
        if flow_imbalance > PARAMS['flow_imbalance_threshold']:
            buy_signals += 1
            self.micro_signals['flow'].append(1)
        elif flow_imbalance < -PARAMS['flow_imbalance_threshold']:
            sell_signals += 1
            self.micro_signals['flow'].append(-1)
        else:
            self.micro_signals['flow'].append(0)
            
        # 交易量信号
        if self.detect_volume_surge():
            # 如果有明确的价格方向，则跟随
            if momentum > 0:
                buy_signals += 1
                self.micro_signals['volume'].append(1)
            elif momentum < 0:
                sell_signals += 1
                self.micro_signals['volume'].append(-1)
            else:
                self.micro_signals['volume'].append(0)
                
        # 价差信号 - 价差收窄表示可能有趋势
        if len(self.spread_history) > 1:
            if spread < np.mean(self.spread_history[-3:]) * 0.8:
                # 价差收窄 - 根据其他信号方向增强
                if buy_signals > sell_signals:
                    buy_signals += 1
                elif sell_signals > buy_signals:
                    sell_signals += 1
                self.micro_signals['spread'].append(1 if buy_signals > sell_signals else -1 if sell_signals > buy_signals else 0)
            else:
                self.micro_signals['spread'].append(0)
                
        # 公平价格与市场价格的差异信号
        price_diff = fair_price - mid_price
        if abs(price_diff) > spread * 2:  # 价差的2倍以上
            if price_diff > 0:
                buy_signals += 1
            else:
                sell_signals += 1
                
        # 维持信号历史记录长度
        for signal_type in self.micro_signals:
            if len(self.micro_signals[signal_type]) > PARAMS['max_signal_lookback']:
                self.micro_signals[signal_type] = self.micro_signals[signal_type][-PARAMS['max_signal_lookback']:]
                
        # 判断是否有持续信号（连续出现）
        for signal_type in ['imbalance', 'momentum', 'flow']:
            if len(self.micro_signals[signal_type]) >= 2:
                if all(s > 0 for s in self.micro_signals[signal_type][-2:]):
                    buy_signals += 1
                elif all(s < 0 for s in self.micro_signals[signal_type][-2:]):
                    sell_signals += 1
                    
        return buy_signals, sell_signals
    
    def calculate_trade_quantity(self, direction: int, current_position: int, order_book_analysis: dict) -> int:
        """计算交易数量"""
        # 确定可用仓位空间
        if direction > 0:  # 买入
            available_position = self.position_limit - current_position
        else:  # 卖出
            available_position = self.position_limit + current_position
            
        # 基础交易数量
        base_quantity = min(PARAMS['max_trade_quantity'], available_position)
        
        # 根据订单簿压力调整数量
        if direction > 0:
            pressure_factor = order_book_analysis['sell_pressure'] * PARAMS['scaling_factor']
        else:
            pressure_factor = order_book_analysis['buy_pressure'] * PARAMS['scaling_factor']
            
        # 极端情况下的激进交易
        extreme_imbalance = abs(order_book_analysis['total_imbalance']) > PARAMS['extreme_imbalance']
        if extreme_imbalance:
            adjusted_quantity = base_quantity * PARAMS['aggressive_quantity_multiplier']
        else:
            adjusted_quantity = base_quantity * (1 - pressure_factor)
            
        return max(1, int(adjusted_quantity))
    
    def create_multi_level_orders(self, direction: int, base_price: float, quantity: int) -> List[Order]:
        """创建多级订单"""
        orders = []
        remaining_quantity = quantity
        
        for i in range(PARAMS['order_levels']):
            # 计算该级别的价格
            if direction > 0:  # 买单
                level_price = base_price - PARAMS['level_distances'][i]
            else:  # 卖单
                level_price = base_price + PARAMS['level_distances'][i]
                
            # 计算该级别的数量
            level_quantity = max(1, int(quantity * PARAMS['level_quantities'][i]))
            
            # 确保不超过剩余数量
            level_quantity = min(level_quantity, remaining_quantity)
            remaining_quantity -= level_quantity
            
            if level_quantity > 0:
                if direction > 0:
                    orders.append(Order("MAGNIFICENT_MACARONS", level_price, level_quantity))
                else:
                    orders.append(Order("MAGNIFICENT_MACARONS", level_price, -level_quantity))
                
        return orders
        
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        result = {}
        conversions = 0
        
        if "MAGNIFICENT_MACARONS" not in state.order_depths:
            return result, conversions, state.traderData
            
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        current_position = state.position.get("MAGNIFICENT_MACARONS", 0)
        
        # 计算市场价格
        market_data = self.calculate_market_price(order_depth)
        if market_data[0] is None:
            return result, conversions, state.traderData
            
        mid_price, best_bid, best_ask, spread = market_data
        
        # 更新价格历史
        self.mid_price_history.append(mid_price)
        self.bid_history.append(best_bid)
        self.ask_history.append(best_ask)
        self.spread_history.append(spread)
        
        # 更新交易量
        total_volume = sum(abs(q) for q in order_depth.buy_orders.values()) + sum(abs(q) for q in order_depth.sell_orders.values())
        self.volume_history.append(total_volume)
        
        # 更新仓位历史
        self.position_history.append(current_position)
        
        # 更新记录的最高/最低价格（用于移动止损）
        if self.entry_price is not None:
            if current_position > 0:
                if self.highest_since_entry is None or mid_price > self.highest_since_entry:
                    self.highest_since_entry = mid_price
            elif current_position < 0:
                if self.lowest_since_entry is None or mid_price < self.lowest_since_entry:
                    self.lowest_since_entry = mid_price
        
        # 估算公平价格
        fair_price = self.estimate_fair_price(state)
        
        # 分析订单簿
        book_analysis = self.analyze_order_book_multi_level(order_depth)
        self.depth_imbalance_history.append(book_analysis['total_imbalance'])
        
        # 获取交易信号
        buy_signals, sell_signals = self.get_trade_signals(book_analysis, fair_price, mid_price, spread)
        
        # 初始化订单列表
        orders = []
        
        # 止盈止损逻辑
        if self.should_take_profit(mid_price, current_position) or self.should_stop_loss(mid_price, current_position):
            if current_position > 0:
                # 平多仓
                sell_price = best_bid
                sell_quantity = current_position
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                self.entry_price = None
                self.highest_since_entry = None
                self.lowest_since_entry = None
                self.trailing_stop_active = False
                self.trailing_stop_level = None
                return {"MAGNIFICENT_MACARONS": orders}, conversions, state.traderData
            elif current_position < 0:
                # 平空仓
                buy_price = best_ask
                buy_quantity = abs(current_position)
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                self.entry_price = None
                self.highest_since_entry = None
                self.lowest_since_entry = None
                self.trailing_stop_active = False
                self.trailing_stop_level = None
                return {"MAGNIFICENT_MACARONS": orders}, conversions, state.traderData
        
        # 交易决策
        if buy_signals >= PARAMS['signal_count_threshold'] and buy_signals > sell_signals:
            # 买入信号
            available_buy = self.position_limit - current_position
            if available_buy > 0:
                # 计算交易数量
                buy_quantity = self.calculate_trade_quantity(1, current_position, book_analysis)
                
                if buy_quantity > 0:
                    # 创建多级订单
                    buy_orders = self.create_multi_level_orders(1, best_ask, buy_quantity)
                    orders.extend(buy_orders)
                    
                    # 记录入场价格（如果是新仓位）
                    if current_position <= 0:
                        self.entry_price = best_ask
                        self.highest_since_entry = best_ask
                        self.lowest_since_entry = best_ask
                        self.trailing_stop_active = False
                        self.trailing_stop_level = None
                
        elif sell_signals >= PARAMS['signal_count_threshold'] and sell_signals > buy_signals:
            # 卖出信号
            available_sell = self.position_limit + current_position
            if available_sell > 0:
                # 计算交易数量
                sell_quantity = self.calculate_trade_quantity(-1, current_position, book_analysis)
                
                if sell_quantity > 0:
                    # 创建多级订单
                    sell_orders = self.create_multi_level_orders(-1, best_bid, sell_quantity)
                    orders.extend(sell_orders)
                    
                    # 记录入场价格（如果是新仓位）
                    if current_position >= 0:
                        self.entry_price = best_bid
                        self.highest_since_entry = best_bid
                        self.lowest_since_entry = best_bid
                        self.trailing_stop_active = False
                        self.trailing_stop_level = None
        
        # 市场中性时平仓
        elif book_analysis['total_imbalance'] < PARAMS['imbalance_threshold'] / 2 and book_analysis['total_imbalance'] > -PARAMS['imbalance_threshold'] / 2:
            # 订单簿接近平衡，考虑平仓
            if current_position > 0 and fair_price < mid_price:
                # 平多仓
                sell_quantity = current_position
                orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -sell_quantity))
                self.entry_price = None
                self.highest_since_entry = None
                self.lowest_since_entry = None
                self.trailing_stop_active = False
                self.trailing_stop_level = None
            elif current_position < 0 and fair_price > mid_price:
                # 平空仓
                buy_quantity = abs(current_position)
                orders.append(Order("MAGNIFICENT_MACARONS", best_ask, buy_quantity))
                self.entry_price = None
                self.highest_since_entry = None
                self.lowest_since_entry = None
                self.trailing_stop_active = False
                self.trailing_stop_level = None
        
        result["MAGNIFICENT_MACARONS"] = orders
        return result, conversions, state.traderData 