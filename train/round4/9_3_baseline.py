import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# 全局参数配置 - 变种3：价格跳跃与流动性缺口利用策略
PARAMS = {
    # 基础参数
    'position_limit': 75,  # 仓位限制
    'conversion_limit': 10,  # 转换限制
    
    # 价格跳跃检测参数
    'jump_threshold_base': 0.003,  # 基础跳跃阈值（0.3%）
    'jump_lookback': 2,  # 跳跃检测回看周期
    'jump_volatility_multiplier': 1.5,  # 波动率乘数（用于动态阈值）
    'jump_volume_threshold': 3,  # 成交量增加阈值（倍数）
    
    # 流动性缺口参数
    'liquidity_depth_threshold': 3,  # 订单簿深度检查档位数
    'thin_book_threshold': 0.3,  # 薄订单簿阈值（相对历史）
    'gap_size_threshold': 2,  # 价格档位间隙阈值
    'normal_spread_multiplier': 2.0,  # 正常价差倍数（检测异常价差）
    
    # 追踪止损参数
    'stop_loss_pct': 0.003,  # 止损百分比（0.3%）
    'trailing_stop_activation': 0.002,  # 追踪止损激活阈值（0.2%）
    'trailing_stop_distance': 0.001,  # 追踪止损距离（0.1%）
    
    # 交易执行参数
    'jump_trade_size_ratio': 0.6,  # 跳跃交易数量比例（占可用仓位）
    'gap_trade_size_ratio': 0.4,  # 缺口交易数量比例
    'quick_profit_target': 0.002,  # 快速获利目标（0.2%）
    'max_position_holding_time': 10,  # 最大持仓时间（循环数）
    
    # 动态参数调整
    'adaptive_threshold_window': 20,  # 适应性阈值窗口
    'adaptive_factor': 0.7,  # 适应性因子
}

class Trader:
    def __init__(self):
        # 初始化参数
        self.position_limit = PARAMS['position_limit']
        self.conversion_limit = PARAMS['conversion_limit']
        
        # 价格历史
        self.price_history = []
        self.volume_history = []
        self.spread_history = []
        self.mid_price_history = []
        
        # 流动性历史
        self.buy_depth_history = []  # 买盘深度历史
        self.sell_depth_history = []  # 卖盘深度历史
        
        # 仓位管理
        self.position_entry_price = 0  # 入场价格
        self.position_entry_time = 0   # 入场时间
        self.current_time = 0           # 当前时间
        self.trailing_stop_price = 0    # 追踪止损价格
        self.trailing_stop_activated = False  # 追踪止损是否激活
        
        # 状态跟踪
        self.last_jump_direction = 0    # 上次跳跃方向
        self.last_jump_time = 0         # 上次跳跃时间
        self.consecutive_jumps = 0      # 连续跳跃次数
        
        # 波动率计算
        self.volatility_short = 0       # 短期波动率
        self.volatility_window = 10     # 波动率窗口
    
    def update_time(self):
        """更新内部时间计数器"""
        self.current_time += 1
    
    def calculate_volatility(self) -> float:
        """计算价格波动率"""
        if len(self.price_history) < self.volatility_window:
            return 0.001  # 默认低波动率
        
        # 使用最近n个价格计算波动率
        recent_prices = self.price_history[-self.volatility_window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0.001
        return max(volatility, 0.001)  # 确保波动率不为零
    
    def detect_price_jump(self, current_price: float, current_volume: float) -> int:
        """
        检测价格跳跃，返回跳跃方向
        1: 上跳, -1: 下跳, 0: 无跳跃
        """
        if len(self.price_history) < PARAMS['jump_lookback'] + 1:
            return 0
        
        # 计算价格变化
        prev_price = self.price_history[-1]
        price_change = current_price - prev_price
        price_change_pct = abs(price_change / prev_price)
        
        # 计算当前波动率
        volatility = self.calculate_volatility()
        
        # 动态跳跃阈值 = 基础阈值 * 波动率调整
        dynamic_threshold = PARAMS['jump_threshold_base'] * (
            1 + PARAMS['jump_volatility_multiplier'] * volatility / 0.01  # 相对于1%基准波动率调整
        )
        
        # 成交量条件：当前成交量是否明显高于最近平均
        volume_condition = False
        if len(self.volume_history) >= 3:
            avg_recent_volume = sum(self.volume_history[-3:]) / 3
            volume_condition = current_volume > avg_recent_volume * PARAMS['jump_volume_threshold']
        
        # 检测跳跃
        if price_change_pct > dynamic_threshold:
            # 如果是连续同向跳跃，提高阈值要求
            if self.last_jump_direction * np.sign(price_change) > 0 and self.current_time - self.last_jump_time < 3:
                self.consecutive_jumps += 1
                # 连续跳跃要求更高阈值
                if price_change_pct > dynamic_threshold * (1 + 0.3 * self.consecutive_jumps):
                    self.last_jump_time = self.current_time
                    self.last_jump_direction = 1 if price_change > 0 else -1
                    return self.last_jump_direction
            else:
                # 新方向的跳跃或间隔足够长
                self.consecutive_jumps = 1
                self.last_jump_time = self.current_time
                self.last_jump_direction = 1 if price_change > 0 else -1
                return self.last_jump_direction
        
        # 无跳跃或不满足条件
        if self.current_time - self.last_jump_time > 5:
            self.consecutive_jumps = 0  # 重置连续跳跃计数
        return 0
    
    def detect_liquidity_gap(self, order_depth: OrderDepth) -> Tuple[int, float]:
        """
        检测流动性缺口，返回(方向, 流动性分数)
        方向: 1=买方缺口(卖盘薄), -1=卖方缺口(买盘薄), 0=无明显缺口
        流动性分数: 0-1之间，越低表示缺口越大
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0, 0.0
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # 计算买卖盘深度
        buy_depth = 0
        sell_depth = 0
        
        # 买盘深度计算
        buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)[:PARAMS['liquidity_depth_threshold']]
        for price in buy_prices:
            buy_depth += abs(order_depth.buy_orders[price])
        
        # 卖盘深度计算
        sell_prices = sorted(order_depth.sell_orders.keys())[:PARAMS['liquidity_depth_threshold']]
        for price in sell_prices:
            sell_depth += abs(order_depth.sell_orders[price])
        
        # 保存深度历史
        self.buy_depth_history.append(buy_depth)
        self.sell_depth_history.append(sell_depth)
        
        # 检查是否有足够的历史数据进行比较
        if len(self.buy_depth_history) < 5 or len(self.sell_depth_history) < 5:
            return 0, 0.5
        
        # 计算相对历史的深度比例
        avg_buy_depth = sum(self.buy_depth_history[-5:]) / 5
        avg_sell_depth = sum(self.sell_depth_history[-5:]) / 5
        
        buy_depth_ratio = buy_depth / avg_buy_depth if avg_buy_depth > 0 else 1.0
        sell_depth_ratio = sell_depth / avg_sell_depth if avg_sell_depth > 0 else 1.0
        
        # 检测价格级别间的缺口
        buy_gap = False
        sell_gap = False
        
        if len(buy_prices) >= 2:
            for i in range(len(buy_prices) - 1):
                if buy_prices[i] - buy_prices[i+1] > PARAMS['gap_size_threshold']:
                    buy_gap = True
                    break
        
        if len(sell_prices) >= 2:
            for i in range(len(sell_prices) - 1):
                if sell_prices[i+1] - sell_prices[i] > PARAMS['gap_size_threshold']:
                    sell_gap = True
                    break
        
        # 计算异常价差
        spread = best_ask - best_bid
        avg_spread = sum(self.spread_history[-5:]) / 5 if len(self.spread_history) >= 5 else spread
        abnormal_spread = spread > avg_spread * PARAMS['normal_spread_multiplier']
        
        # 确定流动性缺口方向和强度
        direction = 0
        liquidity_score = 0.5
        
        # 买方缺口（卖盘薄）
        if sell_depth_ratio < PARAMS['thin_book_threshold'] or (sell_gap and abnormal_spread):
            direction = 1
            liquidity_score = max(0.1, min(sell_depth_ratio, 0.5))
        
        # 卖方缺口（买盘薄）
        elif buy_depth_ratio < PARAMS['thin_book_threshold'] or (buy_gap and abnormal_spread):
            direction = -1
            liquidity_score = max(0.1, min(buy_depth_ratio, 0.5))
        
        return direction, liquidity_score
    
    def check_trailing_stop(self, current_price: float, current_position: int) -> bool:
        """
        检查追踪止损是否触发
        返回True表示应该平仓
        """
        if current_position == 0 or not self.trailing_stop_activated:
            return False
        
        # 根据持仓方向检查止损条件
        if current_position > 0:  # 多仓
            return current_price <= self.trailing_stop_price
        else:  # 空仓
            return current_price >= self.trailing_stop_price
    
    def update_trailing_stop(self, current_price: float, current_position: int):
        """更新追踪止损价格"""
        if current_position == 0:
            self.trailing_stop_activated = False
            return
        
        # 计算当前盈利百分比
        if self.position_entry_price == 0:
            return
            
        profit_pct = (current_price - self.position_entry_price) / self.position_entry_price
        if current_position < 0:  # 空仓反向计算
            profit_pct = -profit_pct
        
        # 激活追踪止损
        if profit_pct >= PARAMS['trailing_stop_activation'] and not self.trailing_stop_activated:
            self.trailing_stop_activated = True
            if current_position > 0:  # 多仓
                self.trailing_stop_price = current_price * (1 - PARAMS['trailing_stop_distance'])
            else:  # 空仓
                self.trailing_stop_price = current_price * (1 + PARAMS['trailing_stop_distance'])
        
        # 更新追踪止损价格
        elif self.trailing_stop_activated:
            if current_position > 0:  # 多仓，止损价格只上调不下调
                new_stop_price = current_price * (1 - PARAMS['trailing_stop_distance'])
                if new_stop_price > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop_price
            else:  # 空仓，止损价格只下调不上调
                new_stop_price = current_price * (1 + PARAMS['trailing_stop_distance'])
                if new_stop_price < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop_price
    
    def should_take_quick_profit(self, current_price: float, current_position: int) -> bool:
        """判断是否应该快速获利了结"""
        if current_position == 0 or self.position_entry_price == 0:
            return False
        
        # 计算当前收益率
        profit_pct = (current_price - self.position_entry_price) / self.position_entry_price
        if current_position < 0:  # 空仓反向计算
            profit_pct = -profit_pct
        
        # 达到快速获利目标
        return profit_pct >= PARAMS['quick_profit_target']
    
    def should_close_by_time(self, current_position: int) -> bool:
        """判断是否因持仓时间过长而平仓"""
        if current_position == 0 or self.position_entry_time == 0:
            return False
        
        holding_time = self.current_time - self.position_entry_time
        return holding_time >= PARAMS['max_position_holding_time']
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        result = {}
        conversions = 0
        self.update_time()  # 更新内部时间
        
        if "MAGNIFICENT_MACARONS" not in state.order_depths:
            return result, conversions, state.traderData
            
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        current_position = state.position.get("MAGNIFICENT_MACARONS", 0)
        
        # 如果订单簿为空，则不操作
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return result, conversions, state.traderData
            
        # 获取当前价格
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # 更新历史数据
        self.price_history.append(mid_price)
        self.mid_price_history.append(mid_price)
        self.spread_history.append(spread)
        
        # 模拟成交量（使用订单簿深度作为替代）
        current_volume = sum(abs(qty) for qty in order_depth.buy_orders.values()) + \
                         sum(abs(qty) for qty in order_depth.sell_orders.values())
        self.volume_history.append(current_volume)
        
        # 检测价格跳跃和流动性缺口
        jump_direction = self.detect_price_jump(mid_price, current_volume)
        gap_direction, liquidity_score = self.detect_liquidity_gap(order_depth)
        
        # 更新追踪止损
        self.update_trailing_stop(mid_price, current_position)
        
        orders = []
        trade_executed = False
        
        # 1. 首先检查是否需要平仓（止损/止盈/时间）
        if self.check_trailing_stop(mid_price, current_position):
            # 追踪止损触发，全部平仓
            if current_position > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -current_position))
            elif current_position < 0:
                orders.append(Order("MAGNIFICENT_MACARONS", best_ask, -current_position))
            trade_executed = True
            self.trailing_stop_activated = False
            self.position_entry_price = 0
            self.position_entry_time = 0
            
        elif self.should_take_quick_profit(mid_price, current_position):
            # 快速获利平仓
            if current_position > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -current_position))
            elif current_position < 0:
                orders.append(Order("MAGNIFICENT_MACARONS", best_ask, -current_position))
            trade_executed = True
            self.position_entry_price = 0
            self.position_entry_time = 0
            
        elif self.should_close_by_time(current_position):
            # 持仓时间过长平仓
            if current_position > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -current_position))
            elif current_position < 0:
                orders.append(Order("MAGNIFICENT_MACARONS", best_ask, -current_position))
            trade_executed = True
            self.position_entry_price = 0
            self.position_entry_time = 0
        
        # 2. 如果无需平仓，检查新交易信号
        if not trade_executed:
            # 价格跳跃信号
            if jump_direction != 0:
                available_position = 0
                trade_price = 0
                trade_size = 0
                
                if jump_direction > 0:  # 向上跳跃，做多
                    available_position = self.position_limit - current_position
                    if available_position > 0:
                        trade_price = best_ask  # 市价买入
                        trade_size = int(available_position * PARAMS['jump_trade_size_ratio'])
                        if trade_size > 0:
                            orders.append(Order("MAGNIFICENT_MACARONS", trade_price, trade_size))
                            trade_executed = True
                            # 记录入场信息
                            if current_position == 0:
                                self.position_entry_price = trade_price
                                self.position_entry_time = self.current_time
                
                elif jump_direction < 0:  # 向下跳跃，做空
                    available_position = self.position_limit + current_position
                    if available_position > 0:
                        trade_price = best_bid  # 市价卖出
                        trade_size = int(available_position * PARAMS['jump_trade_size_ratio'])
                        if trade_size > 0:
                            orders.append(Order("MAGNIFICENT_MACARONS", trade_price, -trade_size))
                            trade_executed = True
                            # 记录入场信息
                            if current_position == 0:
                                self.position_entry_price = trade_price
                                self.position_entry_time = self.current_time
            
            # 如果跳跃信号未执行交易，检查流动性缺口信号
            elif gap_direction != 0 and liquidity_score < 0.4:
                available_position = 0
                trade_price = 0
                trade_size = 0
                
                if gap_direction > 0:  # 卖盘薄，做多
                    available_position = self.position_limit - current_position
                    if available_position > 0:
                        trade_price = best_ask  # 市价买入
                        # 流动性越差，仓位越大
                        position_factor = 1.0 - liquidity_score  # 0.6-0.9
                        trade_size = int(available_position * PARAMS['gap_trade_size_ratio'] * position_factor)
                        if trade_size > 0:
                            orders.append(Order("MAGNIFICENT_MACARONS", trade_price, trade_size))
                            trade_executed = True
                            # 记录入场信息
                            if current_position == 0:
                                self.position_entry_price = trade_price
                                self.position_entry_time = self.current_time
                
                elif gap_direction < 0:  # 买盘薄，做空
                    available_position = self.position_limit + current_position
                    if available_position > 0:
                        trade_price = best_bid  # 市价卖出
                        # 流动性越差，仓位越大
                        position_factor = 1.0 - liquidity_score  # 0.6-0.9
                        trade_size = int(available_position * PARAMS['gap_trade_size_ratio'] * position_factor)
                        if trade_size > 0:
                            orders.append(Order("MAGNIFICENT_MACARONS", trade_price, -trade_size))
                            trade_executed = True
                            # 记录入场信息
                            if current_position == 0:
                                self.position_entry_price = trade_price
                                self.position_entry_time = self.current_time
        
        result["MAGNIFICENT_MACARONS"] = orders
        return result, conversions, state.traderData 