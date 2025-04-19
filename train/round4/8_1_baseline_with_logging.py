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

# 全局参数配置
PARAMS = {
    # 基础参数
    'position_limit': 75,  # 仓位限制
    'conversion_limit': 10,  # 转换限制
    
    # 策略1（基本面+技术面）参数
    'sugar_weight': 0.4,  # 糖价权重
    'sunlight_weight': 0.4,  # 阳光指数权重
    'tariff_weight': 0.2,  # 关税权重
    'ma_short_window': 10,  # 短期移动平均窗口
    'ma_long_window': 50,   # 长期移动平均窗口
    
    # 策略2（市场微观结构）参数
    'depth_threshold': 5,  # 订单簿深度阈值
    'spread_threshold': 0.1,  # 价差阈值
    'volume_threshold': 3,  # 交易量阈值
    'pressure_window': 10,  # 压力计算窗口
    'pressure_threshold': 0.7,  # 压力阈值
    
    # 策略选择参数
    'strategy_window': 10,  # 策略评估窗口
    'strategy_weight_threshold': 0.6,  # 策略权重阈值
    
    # 交易参数
    'max_trade_quantity': 15,  # 最大交易数量
    'imbalance_threshold': 0.2,  # 订单簿不平衡阈值
    'price_score_neutral': 0.5,  # 价格得分中性值
    'price_score_threshold': 0.1,  # 价格得分阈值
    'spread_multiplier': 1.5,  # 价差倍数
    'flow_ratio': 1.5,  # 订单流比率
    'imbalance_ratio': 0.3,  # 订单簿不平衡比率
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
        
        # 策略选择参数
        self.strategy_window = PARAMS['strategy_window']
        self.strategy_scores = []
        
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
            logger.flush(state, result, conversions, "")
            return result, conversions, state.traderData
            
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        current_position = state.position.get("MAGNIFICENT_MACARONS", 0)
        
        # 获取当前价格
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.price_history.append(mid_price)
            
            # 记录日志
            logger.print(f"Timestamp: {state.timestamp}")
            logger.print(f"Current Position: {current_position}")
            logger.print(f"Best Bid: {best_bid}, Best Ask: {best_ask}")
            logger.print(f"Mid Price: {mid_price}")
        else:
            logger.flush(state, result, conversions, "")
            return result, conversions, state.traderData
            
        # 计算策略权重
        strategy1_weight, strategy2_weight = self.evaluate_strategy_performance()
        logger.print(f"Strategy Weights - Strategy1: {strategy1_weight:.2f}, Strategy2: {strategy2_weight:.2f}")
        
        # 策略1：基本面+技术面分析
        price_score = self.calculate_price_score(state)
        ma_short = self.calculate_ma(self.price_history, self.ma_short_window)
        ma_long = self.calculate_ma(self.price_history, self.ma_long_window)
        logger.print(f"Price Score: {price_score:.2f}, MA Short: {ma_short:.2f}, MA Long: {ma_long:.2f}")
        
        # 策略2：市场微观结构分析
        buy_pressure, sell_pressure = self.calculate_market_pressure(order_depth)
        self.buy_pressure_history.append(buy_pressure)
        self.sell_pressure_history.append(sell_pressure)
        
        buy_flow, sell_flow = self.analyze_order_flow(order_depth)
        spread, relative_spread = self.calculate_spread_metrics(order_depth)
        self.spread_history.append(relative_spread)
        
        imbalance = self.analyze_order_book_imbalance(order_depth)
        self.imbalance_history.append(imbalance)
        
        logger.print(f"Market Pressure - Buy: {buy_pressure:.2f}, Sell: {sell_pressure:.2f}")
        logger.print(f"Order Flow - Buy: {buy_flow:.2f}, Sell: {sell_flow:.2f}")
        logger.print(f"Spread - Absolute: {spread:.2f}, Relative: {relative_spread:.2f}")
        logger.print(f"Order Book Imbalance: {imbalance:.2f}")
        
        # 确定交易方向
        orders = []
        
        # 策略1信号
        strategy1_buy = (price_score > 0.6 and ma_short > ma_long)
        strategy1_sell = (price_score < 0.4 and ma_short < ma_long)
        
        # 策略2信号
        strategy2_buy = (sell_pressure > self.pressure_threshold and
                        relative_spread < self.spread_threshold and
                        imbalance < -PARAMS['imbalance_ratio'] and
                        sell_flow > buy_flow * PARAMS['flow_ratio'])
        
        strategy2_sell = (buy_pressure > self.pressure_threshold and
                         relative_spread < self.spread_threshold and
                         imbalance > PARAMS['imbalance_ratio'] and
                         buy_flow > sell_flow * PARAMS['flow_ratio'])
        
        logger.print(f"Strategy Signals - Strategy1 Buy: {strategy1_buy}, Strategy1 Sell: {strategy1_sell}")
        logger.print(f"Strategy Signals - Strategy2 Buy: {strategy2_buy}, Strategy2 Sell: {strategy2_sell}")
        
        # 综合两个策略的信号
        if (strategy1_buy and strategy1_weight > PARAMS['strategy_weight_threshold']) or \
           (strategy2_buy and strategy2_weight > PARAMS['strategy_weight_threshold']):
            # 买入信号
            available_buy = self.position_limit - current_position
            if available_buy > 0:
                buy_price = best_ask
                buy_quantity = min(available_buy, PARAMS['max_trade_quantity'])
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                logger.print(f"Placing Buy Order - Price: {buy_price}, Quantity: {buy_quantity}")
                
        elif (strategy1_sell and strategy1_weight > PARAMS['strategy_weight_threshold']) or \
             (strategy2_sell and strategy2_weight > PARAMS['strategy_weight_threshold']):
            # 卖出信号
            available_sell = self.position_limit + current_position
            if available_sell > 0:
                sell_price = best_bid
                sell_quantity = min(available_sell, PARAMS['max_trade_quantity'])
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                logger.print(f"Placing Sell Order - Price: {sell_price}, Quantity: {sell_quantity}")
                
        # 平仓逻辑
        elif (abs(imbalance) < PARAMS['imbalance_threshold'] and
              (relative_spread > self.spread_threshold * PARAMS['spread_multiplier'] or
               abs(price_score - PARAMS['price_score_neutral']) < PARAMS['price_score_threshold'])):
            
            if current_position > 0:
                sell_price = best_bid
                sell_quantity = min(current_position, PARAMS['max_trade_quantity'])
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                logger.print(f"Closing Long Position - Price: {sell_price}, Quantity: {sell_quantity}")
            elif current_position < 0:
                buy_price = best_ask
                buy_quantity = min(abs(current_position), PARAMS['max_trade_quantity'])
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                logger.print(f"Closing Short Position - Price: {buy_price}, Quantity: {buy_quantity}")
        
        result["MAGNIFICENT_MACARONS"] = orders
        logger.flush(state, result, conversions, "")
        return result, conversions, state.traderData 