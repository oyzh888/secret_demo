import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# 全局参数配置 - 动量跟踪激进型
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
    
    # 动量参数 - 激进设置
    'ema_short': 3,  # 超短期指数移动平均线
    'ema_medium': 10,  # 中期指数移动平均线
    'ema_long': 20,  # 长期指数移动平均线
    'rsi_period': 5,  # RSI周期（短）
    'rsi_overbought': 70,  # RSI超买阈值
    'rsi_oversold': 30,  # RSI超卖阈值
    'momentum_period': 3,  # 动量计算周期
    'trend_strength_threshold': 0.5,  # 趋势强度阈值
    
    # 趋势判断参数
    'breakout_threshold': 1.0,  # 突破阈值
    'consolidation_bars': 3,  # 盘整期间数
    'pullback_threshold': 0.3,  # 回调阈值
    'acceleration_threshold': 0.2,  # 加速阈值
    
    # 价格调整参数
    'momentum_multiplier': 2.0,  # 动量修正因子（放大）
    'fair_price_trend_adj': 0.8,  # 趋势对公允价格的调整比例
    
    # 交易量分析
    'vol_lookback': 5,  # 交易量回看周期
    'vol_surge_threshold': 1.3,  # 交易量激增阈值
    
    # 交易执行参数
    'base_quantity': 15,  # 基础交易数量
    'scaling_factor': 0.8,  # 基于趋势强度的缩放因子
    'max_trade_quantity': 25,  # 最大交易数量
    'trend_position_scale': 0.9,  # 趋势下的仓位利用
    'mean_reversion_scale': 0.5,  # 反转模式下的仓位利用
    
    # 风险管理参数
    'stop_loss_atr_multiple': 1.2,  # 止损为ATR的倍数
    'take_profit_atr_multiple': 2.0,  # 止盈为ATR的倍数
    'atr_period': 5,  # ATR计算周期
    'max_drawdown': 0.05,  # 最大回撤限制
    
    # 价格目标参数
    'price_target_atr_multiple': 3.0,  # 目标价为ATR的倍数
    'trailing_stop_activation': 0.5,  # 移动止损激活点数（ATR的倍数）
}

class Trader:
    def __init__(self):
        # 初始化参数
        self.position_limit = PARAMS['position_limit']
        self.conversion_limit = PARAMS['conversion_limit']
        
        # 价格历史数据
        self.price_history = []
        self.mid_price_history = []
        self.ema_short_history = []
        self.ema_medium_history = []
        self.ema_long_history = []
        self.rsi_history = []
        self.atr_history = []
        self.volume_history = []
        self.dir_movement_history = []
        
        # 订单簿数据
        self.spread_history = []
        self.depth_history = []
        
        # 趋势状态
        self.trend_state = 'neutral'  # 'uptrend', 'downtrend', 'neutral'
        self.trend_strength = 0.0
        self.last_breakout_price = None
        self.last_pivot_high = None
        self.last_pivot_low = None
        
        # 仓位管理
        self.position_history = []
        self.entry_price = None
        self.initial_stop_loss = None
        self.trailing_stop = None
        self.price_target = None
        self.current_drawdown = 0.0
        
        # 回归模型系数
        self.reg_coefs = PARAMS['regression_coefs']
    
    def update_indicators(self, price: float, volume: float = 0):
        """更新所有技术指标"""
        # 更新价格历史
        self.mid_price_history.append(price)
        
        # 更新交易量历史
        self.volume_history.append(volume)
        
        # 计算EMA
        self.update_ema(price)
        
        # 计算RSI
        self.update_rsi()
        
        # 计算ATR
        self.update_atr()
        
        # 计算方向运动
        self.update_directional_movement()
        
        # 评估趋势状态
        self.evaluate_trend_state()
    
    def update_ema(self, price: float):
        """更新各周期EMA"""
        # 短期EMA
        if not self.ema_short_history:
            self.ema_short_history.append(price)
        else:
            alpha = 2 / (PARAMS['ema_short'] + 1)
            ema_short = price * alpha + self.ema_short_history[-1] * (1 - alpha)
            self.ema_short_history.append(ema_short)
            
        # 中期EMA
        if not self.ema_medium_history:
            self.ema_medium_history.append(price)
        else:
            alpha = 2 / (PARAMS['ema_medium'] + 1)
            ema_medium = price * alpha + self.ema_medium_history[-1] * (1 - alpha)
            self.ema_medium_history.append(ema_medium)
            
        # 长期EMA
        if not self.ema_long_history:
            self.ema_long_history.append(price)
        else:
            alpha = 2 / (PARAMS['ema_long'] + 1)
            ema_long = price * alpha + self.ema_long_history[-1] * (1 - alpha)
            self.ema_long_history.append(ema_long)
            
        # 限制历史记录长度
        max_history = 100
        self.ema_short_history = self.ema_short_history[-max_history:]
        self.ema_medium_history = self.ema_medium_history[-max_history:]
        self.ema_long_history = self.ema_long_history[-max_history:]
    
    def update_rsi(self):
        """更新RSI指标"""
        period = PARAMS['rsi_period']
        if len(self.mid_price_history) <= period:
            self.rsi_history.append(50)  # 默认初始值
            return
            
        price_changes = np.diff(self.mid_price_history[-period-1:])
        gains = np.sum([max(0, change) for change in price_changes])
        losses = np.sum([abs(min(0, change)) for change in price_changes])
        
        if losses == 0:
            rsi = 100
        else:
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
            
        self.rsi_history.append(rsi)
        self.rsi_history = self.rsi_history[-100:]  # 限制长度
    
    def update_atr(self):
        """更新ATR (Average True Range)"""
        period = PARAMS['atr_period']
        if len(self.mid_price_history) <= 1:
            self.atr_history.append(0.5)  # 默认初始值
            return
            
        # 简化：使用高低价差作为范围的估计
        if len(self.mid_price_history) >= period + 1:
            price_series = self.mid_price_history[-period-1:]
            true_ranges = []
            
            for i in range(1, len(price_series)):
                true_range = abs(price_series[i] - price_series[i-1])
                true_ranges.append(true_range)
                
            atr = np.mean(true_ranges)
        else:
            atr = abs(self.mid_price_history[-1] - self.mid_price_history[-2])
            
        # EMA滤波ATR
        if not self.atr_history:
            self.atr_history.append(atr)
        else:
            alpha = 2 / (period + 1)
            smoothed_atr = atr * alpha + self.atr_history[-1] * (1 - alpha)
            self.atr_history.append(smoothed_atr)
            
        self.atr_history = self.atr_history[-100:]  # 限制长度
    
    def update_directional_movement(self):
        """更新方向运动指标"""
        if len(self.mid_price_history) <= 1:
            self.dir_movement_history.append(0)
            return
            
        price = self.mid_price_history[-1]
        prev_price = self.mid_price_history[-2]
        
        # 简单方向：+1(上涨)，-1(下跌)，0(横盘)
        if price > prev_price:
            self.dir_movement_history.append(1)
        elif price < prev_price:
            self.dir_movement_history.append(-1)
        else:
            self.dir_movement_history.append(0)
            
        self.dir_movement_history = self.dir_movement_history[-100:]
    
    def evaluate_trend_state(self):
        """评估当前趋势状态"""
        if len(self.ema_short_history) < 2 or len(self.ema_medium_history) < 2:
            self.trend_state = 'neutral'
            self.trend_strength = 0.0
            return
            
        # 判断EMAs的关系
        short_above_medium = self.ema_short_history[-1] > self.ema_medium_history[-1]
        medium_above_long = (len(self.ema_long_history) >= 2 and 
                           self.ema_medium_history[-1] > self.ema_long_history[-1])
        
        # 趋势方向变化
        short_cross_medium = ((self.ema_short_history[-1] > self.ema_medium_history[-1] and 
                             self.ema_short_history[-2] < self.ema_medium_history[-2]) or
                            (self.ema_short_history[-1] < self.ema_medium_history[-1] and 
                             self.ema_short_history[-2] > self.ema_medium_history[-2]))
        
        # 计算趋势强度（使用短期和中期EMA的距离）
        ema_distance = abs(self.ema_short_history[-1] - self.ema_medium_history[-1])
        if len(self.atr_history) > 0 and self.atr_history[-1] > 0:
            # 距离相对于ATR的比例
            self.trend_strength = min(1.0, ema_distance / self.atr_history[-1])
        else:
            self.trend_strength = 0.5
            
        # 短期方向性
        recent_moves = self.dir_movement_history[-PARAMS['momentum_period']:]
        directional_bias = sum(recent_moves) / max(1, len(recent_moves))
        
        # 判断趋势状态
        if short_above_medium and medium_above_long and directional_bias > 0:
            self.trend_state = 'uptrend'
            
            # 检查突破
            if self.last_pivot_high and self.mid_price_history[-1] > self.last_pivot_high:
                self.last_breakout_price = self.mid_price_history[-1]
                
        elif not short_above_medium and not medium_above_long and directional_bias < 0:
            self.trend_state = 'downtrend'
            
            # 检查突破
            if self.last_pivot_low and self.mid_price_history[-1] < self.last_pivot_low:
                self.last_breakout_price = self.mid_price_history[-1]
                
        else:
            if abs(directional_bias) < 0.2:
                self.trend_state = 'neutral'
            else:
                # 保持前一状态，但强度降低
                self.trend_strength *= 0.8
                
        # 识别价格关键点（高点和低点）
        if len(self.mid_price_history) >= 3:
            # 简单高点：前一价格高于两边
            if (self.mid_price_history[-2] > self.mid_price_history[-3] and 
                self.mid_price_history[-2] > self.mid_price_history[-1]):
                self.last_pivot_high = self.mid_price_history[-2]
                
            # 简单低点：前一价格低于两边
            if (self.mid_price_history[-2] < self.mid_price_history[-3] and 
                self.mid_price_history[-2] < self.mid_price_history[-1]):
                self.last_pivot_low = self.mid_price_history[-2]
    
    def estimate_fair_price(self, state: TradingState) -> float:
        """使用回归模型估计公平价格，考虑趋势调整"""
        sugar_price = getattr(state.observations, 'sugarPrice', 0)
        sunlight_index = getattr(state.observations, 'sunlightIndex', 0)
        import_tariff = getattr(state.observations, 'importTariff', 0)
        export_tariff = getattr(state.observations, 'exportTariff', 0)
        transport_fee = getattr(state.observations, 'transportFee', 0)
        
        # 基础公允价格（回归模型）
        base_fair_price = (
            self.reg_coefs['intercept'] +
            self.reg_coefs['sunlight'] * sunlight_index +
            self.reg_coefs['sugar_price'] * sugar_price +
            self.reg_coefs['transport_fee'] * transport_fee +
            self.reg_coefs['export_tariff'] * export_tariff +
            self.reg_coefs['import_tariff'] * import_tariff
        )
        
        # 基于趋势调整公允价格
        if len(self.atr_history) > 0 and len(self.mid_price_history) > 0:
            current_atr = self.atr_history[-1]
            
            # 趋势方向调整
            if self.trend_state == 'uptrend':
                # 上涨趋势中，增加公允价格
                trend_adjustment = current_atr * self.trend_strength * PARAMS['momentum_multiplier']
                adjusted_fair_price = base_fair_price + trend_adjustment * PARAMS['fair_price_trend_adj']
            elif self.trend_state == 'downtrend':
                # 下跌趋势中，降低公允价格
                trend_adjustment = current_atr * self.trend_strength * PARAMS['momentum_multiplier']
                adjusted_fair_price = base_fair_price - trend_adjustment * PARAMS['fair_price_trend_adj']
            else:
                adjusted_fair_price = base_fair_price
        else:
            adjusted_fair_price = base_fair_price
            
        return adjusted_fair_price
    
    def calculate_momentum_score(self) -> float:
        """计算动量得分，范围[-1, 1]"""
        if len(self.mid_price_history) < PARAMS['momentum_period'] + 1:
            return 0
            
        # 计算短期动量
        price_changes = np.diff(self.mid_price_history[-PARAMS['momentum_period']-1:])
        
        # 指数加权（更重视最近变化）
        weights = np.exp(np.linspace(0, 1, len(price_changes)))
        weights = weights / np.sum(weights)
        
        weighted_momentum = np.sum(price_changes * weights)
        
        # 标准化
        if len(self.atr_history) > 0 and self.atr_history[-1] > 0:
            normalized_momentum = weighted_momentum / self.atr_history[-1]
            # 限制到[-1, 1]范围
            normalized_momentum = max(-1, min(1, normalized_momentum))
        else:
            normalized_momentum = 0
            
        return normalized_momentum
    
    def detect_acceleration(self) -> float:
        """检测价格加速度"""
        if len(self.mid_price_history) < PARAMS['momentum_period'] + 2:
            return 0
            
        # 计算最近两段的动量变化率
        recent_changes = np.diff(self.mid_price_history[-PARAMS['momentum_period']-1:])
        earlier_changes = np.diff(self.mid_price_history[-PARAMS['momentum_period']-2:-1])
        
        recent_momentum = np.mean(recent_changes)
        earlier_momentum = np.mean(earlier_changes)
        
        # 动量增加为正加速度，减少为负加速度
        acceleration = recent_momentum - earlier_momentum
        
        # 标准化
        if len(self.atr_history) > 0 and self.atr_history[-1] > 0:
            normalized_acceleration = acceleration / self.atr_history[-1]
            # 限制范围
            normalized_acceleration = max(-1, min(1, normalized_acceleration))
        else:
            normalized_acceleration = 0
            
        return normalized_acceleration
    
    def detect_volume_trend(self) -> float:
        """检测交易量趋势"""
        if len(self.volume_history) < PARAMS['vol_lookback']:
            return 0
            
        recent_volume = np.mean(self.volume_history[-3:])
        earlier_volume = np.mean(self.volume_history[-PARAMS['vol_lookback']:-3])
        
        if earlier_volume > 0:
            volume_change = recent_volume / earlier_volume - 1
        else:
            volume_change = 0
            
        return volume_change
    
    def should_take_profit_or_stop_loss(self, current_price: float, position: int) -> bool:
        """检查是否应该止盈止损"""
        if self.entry_price is None or position == 0:
            return False
            
        # 检查止损
        if self.initial_stop_loss is not None:
            if (position > 0 and current_price < self.initial_stop_loss) or \
               (position < 0 and current_price > self.initial_stop_loss):
                return True
                
        # 检查移动止损
        if self.trailing_stop is not None:
            if (position > 0 and current_price < self.trailing_stop) or \
               (position < 0 and current_price > self.trailing_stop):
                return True
                
        # 检查目标价格
        if self.price_target is not None:
            if (position > 0 and current_price > self.price_target) or \
               (position < 0 and current_price < self.price_target):
                return True
                
        # 检查最大回撤
        price_change = current_price - self.entry_price if position > 0 else self.entry_price - current_price
        if price_change < 0:
            drawdown = abs(price_change) / self.entry_price
            if drawdown > PARAMS['max_drawdown']:
                return True
                
        return False
    
    def calculate_position_size(self, current_position: int, is_buy: bool) -> int:
        """计算仓位大小"""
        # 确定可用仓位空间
        if is_buy:  # 买入
            available_position = self.position_limit - current_position
        else:  # 卖出
            available_position = self.position_limit + current_position
            
        # 基础交易数量
        base_quantity = PARAMS['base_quantity']
        
        # 根据趋势强度调整
        if (is_buy and self.trend_state == 'uptrend') or (not is_buy and self.trend_state == 'downtrend'):
            # 顺势加仓
            adjusted_quantity = base_quantity * (1 + self.trend_strength * PARAMS['scaling_factor'])
            # 使用更高的仓位上限
            position_scale = PARAMS['trend_position_scale']
        else:
            # 逆势减仓
            adjusted_quantity = base_quantity * (1 - self.trend_strength * PARAMS['scaling_factor'])
            # 使用更低的仓位上限
            position_scale = PARAMS['mean_reversion_scale']
            
        # 确保不超过设定的上限
        adjusted_quantity = min(adjusted_quantity, PARAMS['max_trade_quantity'])
        
        # 确保不超过仓位限制
        adjusted_quantity = min(adjusted_quantity, available_position * position_scale)
        
        return max(1, int(adjusted_quantity))
    
    def set_stop_and_target(self, position: int, entry_price: float):
        """设置止损和目标价格"""
        if len(self.atr_history) == 0:
            return
            
        current_atr = self.atr_history[-1]
        
        # 设置初始止损
        stop_distance = current_atr * PARAMS['stop_loss_atr_multiple']
        if position > 0:  # 多仓
            self.initial_stop_loss = entry_price - stop_distance
            self.trailing_stop = self.initial_stop_loss
            self.price_target = entry_price + current_atr * PARAMS['price_target_atr_multiple']
        elif position < 0:  # 空仓
            self.initial_stop_loss = entry_price + stop_distance
            self.trailing_stop = self.initial_stop_loss
            self.price_target = entry_price - current_atr * PARAMS['price_target_atr_multiple']
            
    def update_trailing_stop(self, current_price: float, position: int):
        """更新移动止损"""
        if self.trailing_stop is None or position == 0:
            return
            
        # 激活条件：价格移动超过ATR的一定倍数
        activation_distance = self.atr_history[-1] * PARAMS['trailing_stop_activation'] if len(self.atr_history) > 0 else 0
        
        if position > 0:  # 多仓
            # 价格上涨时提高止损
            potential_stop = current_price - activation_distance
            if potential_stop > self.trailing_stop:
                self.trailing_stop = potential_stop
        elif position < 0:  # 空仓
            # 价格下跌时降低止损
            potential_stop = current_price + activation_distance
            if potential_stop < self.trailing_stop:
                self.trailing_stop = potential_stop
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        result = {}
        conversions = 0
        
        if "MAGNIFICENT_MACARONS" not in state.order_depths:
            return result, conversions, state.traderData
            
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        current_position = state.position.get("MAGNIFICENT_MACARONS", 0)
        
        # 更新仓位历史
        self.position_history.append(current_position)
        
        # 获取市场价格
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return result, conversions, state.traderData
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        # 计算交易量
        total_volume = sum(abs(q) for q in order_depth.buy_orders.values()) + sum(abs(q) for q in order_depth.sell_orders.values())
        
        # 更新技术指标
        self.update_indicators(mid_price, total_volume)
        
        # 更新移动止损
        if self.entry_price is not None and current_position != 0:
            self.update_trailing_stop(mid_price, current_position)
        
        # 估算公平价格（考虑趋势调整）
        fair_price = self.estimate_fair_price(state)
        
        # 计算动量和加速度得分
        momentum_score = self.calculate_momentum_score()
        acceleration = self.detect_acceleration()
        volume_trend = self.detect_volume_trend()
        
        # 初始化订单列表
        orders = []
        
        # 止盈止损检查
        if self.should_take_profit_or_stop_loss(mid_price, current_position):
            if current_position > 0:
                # 平多仓
                sell_price = best_bid
                sell_quantity = current_position
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                self.entry_price = None
                self.initial_stop_loss = None
                self.trailing_stop = None
                self.price_target = None
                return {"MAGNIFICENT_MACARONS": orders}, conversions, state.traderData
            elif current_position < 0:
                # 平空仓
                buy_price = best_ask
                buy_quantity = abs(current_position)
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                self.entry_price = None
                self.initial_stop_loss = None
                self.trailing_stop = None
                self.price_target = None
                return {"MAGNIFICENT_MACARONS": orders}, conversions, state.traderData
        
        # 趋势跟踪和动量交易决策
        
        # 上涨趋势信号
        uptrend_signal = (
            (self.trend_state == 'uptrend' and self.trend_strength > PARAMS['trend_strength_threshold']) or 
            (momentum_score > PARAMS['breakout_threshold']) or
            (acceleration > PARAMS['acceleration_threshold'] and momentum_score > 0)
        )
        
        # 下跌趋势信号
        downtrend_signal = (
            (self.trend_state == 'downtrend' and self.trend_strength > PARAMS['trend_strength_threshold']) or 
            (momentum_score < -PARAMS['breakout_threshold']) or
            (acceleration < -PARAMS['acceleration_threshold'] and momentum_score < 0)
        )
        
        # 交易决策
        price_diff = fair_price - mid_price
        price_diff_pct = price_diff / mid_price
        
        # 买入条件
        buy_signal = (
            # 趋势条件
            uptrend_signal or 
            # 均值回归条件：低于公允价格且动量不是强烈下跌
            (price_diff_pct > 0.01 and momentum_score > -PARAMS['breakout_threshold'])
        )
        
        # 卖出条件
        sell_signal = (
            # 趋势条件
            downtrend_signal or 
            # 均值回归条件：高于公允价格且动量不是强烈上涨
            (price_diff_pct < -0.01 and momentum_score < PARAMS['breakout_threshold'])
        )
        
        # 强化动量指标权重
        if volume_trend > PARAMS['vol_surge_threshold']:
            if momentum_score > 0:
                buy_signal = True
            elif momentum_score < 0:
                sell_signal = True
        
        # 执行交易
        if buy_signal and not sell_signal:
            # 买入信号
            available_buy = self.position_limit - current_position
            if available_buy > 0:
                buy_price = best_ask
                buy_quantity = self.calculate_position_size(current_position, True)
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                
                # 如果是新仓位，设置止损和目标
                if current_position <= 0:
                    self.entry_price = buy_price
                    self.set_stop_and_target(buy_quantity, buy_price)
                
        elif sell_signal and not buy_signal:
            # 卖出信号
            available_sell = self.position_limit + current_position
            if available_sell > 0:
                sell_price = best_bid
                sell_quantity = self.calculate_position_size(current_position, False)
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                
                # 如果是新仓位，设置止损和目标
                if current_position >= 0:
                    self.entry_price = sell_price
                    self.set_stop_and_target(-sell_quantity, sell_price)
        
        # 如果当前持仓与趋势方向相反且趋势强，考虑平仓
        elif (current_position > 0 and self.trend_state == 'downtrend' and self.trend_strength > 0.7) or \
             (current_position < 0 and self.trend_state == 'uptrend' and self.trend_strength > 0.7):
            
            if current_position > 0:
                sell_price = best_bid
                sell_quantity = current_position
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                self.entry_price = None
                self.initial_stop_loss = None
                self.trailing_stop = None
                self.price_target = None
            else:
                buy_price = best_ask
                buy_quantity = abs(current_position)
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                self.entry_price = None
                self.initial_stop_loss = None
                self.trailing_stop = None
                self.price_target = None
        
        result["MAGNIFICENT_MACARONS"] = orders
        return result, conversions, state.traderData 