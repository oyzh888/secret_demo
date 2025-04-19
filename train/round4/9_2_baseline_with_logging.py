import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation, Listing, Observation, Symbol, Trade, ProsperityEncoder
import json

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()

# 全局参数配置 - 变种2：高频全仓交易策略
PARAMS = {
    # 基础参数
    'position_limit': 75,  # 仓位限制
    'conversion_limit': 10,  # 转换限制
    
    # 超短线技术指标参数
    'ma_ultra_short_window': 3,  # 超短期移动平均窗口
    'ma_short_window': 5,   # 短期移动平均窗口
    'momentum_window': 2,   # 动量计算窗口
    
    # 高频交易参数
    'price_change_threshold': 0.001,  # 价格变动阈值（仅需0.1%变动）
    'tick_threshold': 1,    # 最小价格变动单位阈值
    'signal_cooldown': 0,   # 信号冷却时间（几乎无冷却）
    
    # 交易规模参数
    'default_trade_ratio': 0.7,  # 默认交易比例（占可用仓位）
    'strong_signal_ratio': 1.0,  # 强信号交易比例（满仓）
    'micro_profit_target': 0.001,  # 微利目标（0.1%）
    
    # 订单簿分析参数
    'depth_threshold': 2,  # 超浅层订单簿分析
    'imbalance_threshold': 0.05,  # 极小的不平衡阈值
    'spread_threshold': 0.01,  # 极小的价差阈值
    
    # 订单分批参数
    'batch_size': 5,       # 每批订单数量
    'batch_interval_ms': 0,  # 批次间隔时间（毫秒）
    
    # 波动率参数
    'volatility_window': 5,  # 波动率计算窗口（极短）
    'volatility_threshold_multiplier': 0.5,  # 波动率阈值乘数（降低阈值）
}

class Trader:
    def __init__(self):
        # 初始化参数
        self.position_limit = PARAMS['position_limit']
        self.conversion_limit = PARAMS['conversion_limit']
        
        # 技术指标参数
        self.ma_ultra_short_window = PARAMS['ma_ultra_short_window']
        self.ma_short_window = PARAMS['ma_short_window']
        self.momentum_window = PARAMS['momentum_window']
        
        # 高频交易参数
        self.price_change_threshold = PARAMS['price_change_threshold']
        self.tick_threshold = PARAMS['tick_threshold']
        self.signal_cooldown = PARAMS['signal_cooldown']
        
        # 历史数据
        self.price_history = []
        self.volume_history = []
        self.trade_times = []  # 记录交易时间
        self.position_history = []  # 记录仓位历史
        self.last_signal_time = 0  # 上次信号时间
        self.current_time = 0  # 当前时间（模拟）
        
        # 移动平均缓存
        self.ma_ultra_short = None
        self.ma_short = None
        
        # 波动率计算
        self.volatility_window = PARAMS['volatility_window']
        self.volatility_history = []
    
    def calculate_ma(self, prices: List[float], window: int) -> float:
        """计算简单移动平均"""
        if len(prices) < window:
            return prices[-1] if prices else 0
        return sum(prices[-window:]) / window
    
    def calculate_momentum(self, prices: List[float], window: int) -> float:
        """计算动量指标"""
        if len(prices) < window + 1:
            return 0
        return prices[-1] - prices[-window-1]
    
    def calculate_volatility(self) -> float:
        """计算短期波动率"""
        if len(self.price_history) < self.volatility_window:
            return 0.0
        
        recent_prices = self.price_history[-self.volatility_window:]
        price_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                         for i in range(1, len(recent_prices))]
        
        return sum(price_changes) / len(price_changes) if price_changes else 0.0
    
    def detect_price_jump(self, current_price: float) -> int:
        """检测价格跳跃，返回方向（1=上涨，-1=下跌，0=无跳跃）"""
        if len(self.price_history) < 2:
            return 0
        
        price_change = current_price - self.price_history[-1]
        price_change_pct = abs(price_change / self.price_history[-1])
        
        # 获取当前波动率
        volatility = self.calculate_volatility()
        # 动态阈值：基于当前波动率
        dynamic_threshold = max(
            PARAMS['price_change_threshold'],
            volatility * PARAMS['volatility_threshold_multiplier']
        )
        
        if price_change_pct > dynamic_threshold:
            return 1 if price_change > 0 else -1
        return 0
    
    def analyze_order_book_pressure(self, order_depth: OrderDepth) -> float:
        """分析订单簿压力，返回值在-1到1之间，负值表示卖压大，正值表示买压大"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        
        # 计算买卖盘前N档总量
        buy_volume = 0
        sell_volume = 0
        
        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True)[:PARAMS['depth_threshold']]:
            buy_volume += abs(quantity)
            
        for price, quantity in sorted(order_depth.sell_orders.items())[:PARAMS['depth_threshold']]:
            sell_volume += abs(quantity)
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0
            
        # 压力指标：买卖盘差额占比
        return (buy_volume - sell_volume) / total_volume
    
    def calculate_order_imbalance(self, order_depth: OrderDepth) -> float:
        """计算订单簿不平衡度，正值表示买方力量大，负值表示卖方力量大"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        # 计算加权买卖量
        buy_pressure = 0
        sell_pressure = 0
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        for price, quantity in order_depth.buy_orders.items():
            # 越接近最高买价，权重越大
            weight = 1.0 - 0.2 * (best_bid - price) / best_bid if best_bid else 1.0
            buy_pressure += abs(quantity) * weight
            
        for price, quantity in order_depth.sell_orders.items():
            # 越接近最低卖价，权重越大
            weight = 1.0 - 0.2 * (price - best_ask) / best_ask if best_ask else 1.0
            sell_pressure += abs(quantity) * weight
        
        total_pressure = buy_pressure + sell_pressure
        if total_pressure == 0:
            return 0
            
        return (buy_pressure - sell_pressure) / total_pressure
    
    def should_trade_now(self) -> bool:
        """检查是否应该现在交易（冷却时间检查）"""
        if self.signal_cooldown == 0:
            return True
            
        time_since_last_signal = self.current_time - self.last_signal_time
        return time_since_last_signal >= self.signal_cooldown
    
    def update_time(self):
        """更新内部时间计数器"""
        self.current_time += 1
        
    def determine_trade_size(self, available_size: int, signal_strength: float) -> int:
        """根据信号强度确定交易数量"""
        if signal_strength >= 0.8:  # 强信号
            ratio = PARAMS['strong_signal_ratio']
        else:  # 一般信号
            ratio = PARAMS['default_trade_ratio'] * max(0.5, signal_strength)
            
        return max(1, min(int(available_size * ratio), available_size))
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        result = {}
        conversions = 0
        self.update_time()  # 更新内部时间
        
        if "MAGNIFICENT_MACARONS" not in state.order_depths:
            logger.flush(state, result, conversions, "")
            return result, conversions, state.traderData
            
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        current_position = state.position.get("MAGNIFICENT_MACARONS", 0)
        self.position_history.append(current_position)
        
        # 获取当前价格
        if not order_depth.buy_orders or not order_depth.sell_orders:
            logger.flush(state, result, conversions, "")
            return result, conversions, state.traderData
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        
        self.price_history.append(mid_price)
        
        # 记录基础市场数据
        logger.print(f"Timestamp: {state.timestamp}")
        logger.print(f"Current Position: {current_position}")
        logger.print(f"Best Bid: {best_bid}, Best Ask: {best_ask}")
        logger.print(f"Mid Price: {mid_price}, Spread: {spread}")
        
        # 计算短期指标
        ma_ultra_short = self.calculate_ma(self.price_history, self.ma_ultra_short_window)
        ma_short = self.calculate_ma(self.price_history, self.ma_short_window)
        momentum = self.calculate_momentum(self.price_history, self.momentum_window)
        
        # 保存指标
        self.ma_ultra_short = ma_ultra_short
        self.ma_short = ma_short
        
        # 记录技术指标
        logger.print(f"Technical Indicators - MA Ultra Short: {ma_ultra_short:.2f}, MA Short: {ma_short:.2f}, Momentum: {momentum:.2f}")
        
        # 计算订单簿指标
        pressure = self.analyze_order_book_pressure(order_depth)
        imbalance = self.calculate_order_imbalance(order_depth)
        
        # 记录订单簿指标
        logger.print(f"Order Book Metrics - Pressure: {pressure:.2f}, Imbalance: {imbalance:.2f}")
        
        # 计算价格跳跃
        jump_direction = self.detect_price_jump(mid_price)
        
        # 计算波动率
        volatility = self.calculate_volatility()
        self.volatility_history.append(volatility)
        
        # 记录波动率和价格跳跃
        logger.print(f"Volatility: {volatility:.4f}, Price Jump Direction: {jump_direction}")
        
        # 交易信号
        buy_signal = False
        sell_signal = False
        signal_strength = 0.0
        
        # 1. 超短期均线交叉信号
        if self.ma_ultra_short > self.ma_short:
            buy_signal = True
            signal_strength += 0.3
        elif self.ma_ultra_short < self.ma_short:
            sell_signal = True
            signal_strength += 0.3
            
        # 2. 价格动量信号
        if momentum > 0:
            buy_signal = buy_signal or True
            signal_strength += 0.2 * min(1.0, abs(momentum) / (volatility if volatility > 0 else 0.001))
        elif momentum < 0:
            sell_signal = sell_signal or True
            signal_strength += 0.2 * min(1.0, abs(momentum) / (volatility if volatility > 0 else 0.001))
            
        # 3. 订单簿压力信号
        if pressure > PARAMS['imbalance_threshold']:
            buy_signal = buy_signal or True
            signal_strength += 0.3 * min(1.0, abs(pressure) / 0.3)
        elif pressure < -PARAMS['imbalance_threshold']:
            sell_signal = sell_signal or True
            signal_strength += 0.3 * min(1.0, abs(pressure) / 0.3)
            
        # 4. 价格跳跃信号
        if jump_direction > 0:
            buy_signal = True
            signal_strength += 0.4
        elif jump_direction < 0:
            sell_signal = True
            signal_strength += 0.4
        
        # 信号强度归一化
        signal_strength = min(1.0, signal_strength)
        
        # 记录交易信号
        logger.print(f"Trading Signals - Buy: {buy_signal}, Sell: {sell_signal}, Signal Strength: {signal_strength:.2f}")
        
        # 确认是否可以交易（冷却时间）
        can_trade = self.should_trade_now()
        logger.print(f"Can Trade: {can_trade}")
        
        orders = []
        
        # 执行交易
        if can_trade and (buy_signal or sell_signal):
            self.last_signal_time = self.current_time
            
            if buy_signal and not sell_signal:
                # 买入信号
                available_buy = self.position_limit - current_position
                if available_buy > 0:
                    buy_quantity = self.determine_trade_size(available_buy, signal_strength)
                    buy_price = best_ask  # 市价单，直接吃卖单
                    orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                    logger.print(f"Placing Buy Order - Price: {buy_price}, Quantity: {buy_quantity}")
                    
            elif sell_signal and not buy_signal:
                # 卖出信号
                available_sell = self.position_limit + current_position
                if available_sell > 0:
                    sell_quantity = self.determine_trade_size(available_sell, signal_strength)
                    sell_price = best_bid  # 市价单，直接吃买单
                    orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                    logger.print(f"Placing Sell Order - Price: {sell_price}, Quantity: {sell_quantity}")
            
            # 如果买卖信号都有但不够强，考虑平仓
            elif buy_signal and sell_signal and signal_strength < 0.5:
                if current_position > 0:
                    # 持有多仓，平仓一部分
                    sell_quantity = min(current_position, max(1, int(current_position * 0.5)))
                    orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -sell_quantity))
                    logger.print(f"Partially Closing Long Position - Price: {best_bid}, Quantity: {sell_quantity}")
                elif current_position < 0:
                    # 持有空仓，平仓一部分
                    buy_quantity = min(abs(current_position), max(1, int(abs(current_position) * 0.5)))
                    orders.append(Order("MAGNIFICENT_MACARONS", best_ask, buy_quantity))
                    logger.print(f"Partially Closing Short Position - Price: {best_ask}, Quantity: {buy_quantity}")
        
        # 微利止盈
        elif current_position != 0 and len(self.price_history) > 2:
            if current_position > 0:
                # 多仓止盈
                last_price_change = (mid_price - self.price_history[-2]) / self.price_history[-2]
                if last_price_change > PARAMS['micro_profit_target']:
                    sell_quantity = min(current_position, max(1, int(current_position * 0.3)))
                    orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -sell_quantity))
                    logger.print(f"Taking Profit on Long Position - Price: {best_bid}, Quantity: {sell_quantity}, Price Change: {last_price_change:.4f}")
            else:
                # 空仓止盈
                last_price_change = (self.price_history[-2] - mid_price) / self.price_history[-2]
                if last_price_change > PARAMS['micro_profit_target']:
                    buy_quantity = min(abs(current_position), max(1, int(abs(current_position) * 0.3)))
                    orders.append(Order("MAGNIFICENT_MACARONS", best_ask, buy_quantity))
                    logger.print(f"Taking Profit on Short Position - Price: {best_ask}, Quantity: {buy_quantity}, Price Change: {last_price_change:.4f}")
        
        result["MAGNIFICENT_MACARONS"] = orders
        logger.flush(state, result, conversions, "")
        return result, conversions, state.traderData 