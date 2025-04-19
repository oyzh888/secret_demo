import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# 全局参数配置 - 变种2：专注于市场微观结构
PARAMS = {
    # 基础参数
    'position_limit': 75,  # 仓位限制
    'conversion_limit': 10,  # 转换限制
    
    # 策略1（基本面+技术面）参数 - 大幅降低权重
    'sugar_weight': 0.3,  # 糖价权重 - 降低权重因为我们更专注于市场微观结构
    'sunlight_weight': 0.2,  # 阳光指数权重 - 降低权重
    'tariff_weight': 0.5,  # 关税权重 - 增加因为它会造成价格冲击
    'ma_short_window': 3,  # 短期移动平均窗口 - 短窗口更敏感
    'ma_long_window': 10,   # 长期移动平均窗口 - 短窗口
    
    # 策略2（市场微观结构）参数 - 大幅强化
    'depth_threshold': 5,  # 订单簿深度阈值 - 加大深度分析
    'spread_threshold': 0.25,  # 价差阈值 - 专注于宽价差交易
    'volume_threshold': 1,  # 交易量阈值 - 最小值，更敏感
    'pressure_window': 3,  # 压力计算窗口 - 更短窗口反应更快
    'pressure_threshold': 0.52,  # 压力阈值 - 降低以更频繁交易
    
    # 策略选择参数
    'strategy_window': 5,  # 策略评估窗口
    'strategy_weight_threshold': 0.4,  # 策略权重阈值 - 降低以更频繁采用策略2
    
    # 交易参数
    'max_trade_quantity': 25,  # 最大交易数量 - 适中的交易规模
    'imbalance_threshold': 0.08,  # 订单簿不平衡阈值 - 非常低的阈值以捕捉细微不平衡
    'price_score_neutral': 0.5,  # 价格得分中性值
    'price_score_threshold': 0.12,  # 价格得分阈值
    'spread_multiplier': 1.15,  # 价差倍数
    'flow_ratio': 1.05,  # 订单流比率 - 更低以更敏感地检测流向
    'imbalance_ratio': 0.12,  # 订单簿不平衡比率 - 更低以更敏感
    
    # 新增参数
    'storage_cost': 0.1,  # 存储成本
    'min_profit_margin': 0.12,  # 最小利润边际 - 更低的利润要求
    'max_spread': 9.0,  # 最大价差
    'volatility_window': 8,  # 波动率计算窗口
    'volatility_threshold': 0.003,  # 波动率阈值
    'order_book_depth_weight': 0.7,  # 订单簿深度权重 - 这是新的参数以强调深度分析
    'order_imbalance_weight': 0.8,  # 订单不平衡权重 - 这是新的参数以强调不平衡分析
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
        self.order_book_depth_history = []
        
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
        
        # 强调关税变化
        tariff_change = 0
        if len(self.import_tariff_history) > 1:
            tariff_change = import_tariff - self.import_tariff_history[-2]
            
        tariff_impact = 0.3 * np.sign(tariff_change) * min(1.0, abs(tariff_change) * 5)
        
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
        
        # 加权计算 - 越靠近盘口的订单权重越大
        if order_depth.buy_orders:
            prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            for i, price in enumerate(prices):
                quantity = order_depth.buy_orders[price]
                # 使用指数衰减权重
                weight = np.exp(-0.5 * i)
                buy_pressure += price * quantity * weight
            
        if order_depth.sell_orders:
            prices = sorted(order_depth.sell_orders.keys())
            for i, price in enumerate(prices):
                quantity = order_depth.sell_orders[price]
                # 使用指数衰减权重
                weight = np.exp(-0.5 * i)
                sell_pressure += price * quantity * weight
            
        total_pressure = buy_pressure + sell_pressure
        if total_pressure > 0:
            buy_pressure = buy_pressure / total_pressure
            sell_pressure = sell_pressure / total_pressure
            
        return buy_pressure, sell_pressure
    
    def analyze_order_flow(self, order_depth: OrderDepth) -> tuple[float, float]:
        """分析订单流（策略2）"""
        buy_flow = 0
        sell_flow = 0
        
        # 加权计算订单流
        if order_depth.buy_orders:
            prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            for i, price in enumerate(prices):
                quantity = order_depth.buy_orders[price]
                # 使用指数衰减权重
                weight = np.exp(-0.3 * i)
                buy_flow += price * quantity * weight
                
        if order_depth.sell_orders:
            prices = sorted(order_depth.sell_orders.keys())
            for i, price in enumerate(prices):
                quantity = order_depth.sell_orders[price]
                # 使用指数衰减权重
                weight = np.exp(-0.3 * i)
                sell_flow += price * quantity * weight
                
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
        """分析订单簿不平衡度（策略2）- 增强版"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        buy_volume = 0
        sell_volume = 0
        
        # 计算买卖深度和平均价格
        buy_depth = len(order_depth.buy_orders)
        sell_depth = len(order_depth.sell_orders)
        
        # 记录订单簿深度
        self.order_book_depth_history.append((buy_depth, sell_depth))
        
        # 使用所有挂单而不只是前几个
        for price, quantity in order_depth.buy_orders.items():
            buy_volume += price * quantity
            
        for price, quantity in order_depth.sell_orders.items():
            sell_volume += price * quantity
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0
            
        # 订单簿不平衡计算
        imbalance = (buy_volume - sell_volume) / total_volume
        
        # 加入深度不平衡
        depth_imbalance = 0
        if buy_depth + sell_depth > 0:
            depth_imbalance = (buy_depth - sell_depth) / (buy_depth + sell_depth)
            
        # 综合不平衡指标
        combined_imbalance = imbalance * (1 - PARAMS['order_book_depth_weight']) + depth_imbalance * PARAMS['order_book_depth_weight']
        
        return combined_imbalance
    
    def evaluate_strategy_performance(self) -> tuple[float, float]:
        """评估两个策略的表现 - 偏向策略2"""
        if len(self.price_history) < PARAMS['strategy_window']:
            # 初始时偏向策略2
            return 0.3, 0.7
            
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
            
        # 标准化得分，但偏向策略2
        total_score = abs(strategy1_score) + abs(strategy2_score) * 1.5  # 策略2得分加权1.5倍
        if total_score > 0:
            strategy1_weight = abs(strategy1_score) / total_score
            strategy2_weight = abs(strategy2_score) * 1.5 / total_score
        else:
            strategy1_weight = 0.3  # 默认给策略1较低权重
            strategy2_weight = 0.7  # 默认给策略2较高权重
            
        # 确保权重总和为1
        total_weight = strategy1_weight + strategy2_weight
        if total_weight > 0:
            strategy1_weight /= total_weight
            strategy2_weight /= total_weight
            
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
        return total_cost < storage_cost * 1.1  # 较低的阈值使得转换更频繁

    def analyze_order_book_structure(self, order_depth: OrderDepth) -> dict:
        """分析订单簿结构 - 新增方法，更深入分析市场微观结构"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return {
                "liquidity_score": 0,
                "depth_score": 0,
                "price_levels": 0,
                "concentration": 0
            }
            
        # 计算订单簿的流动性、深度和集中度
        
        # 价格水平数量
        buy_levels = len(order_depth.buy_orders)
        sell_levels = len(order_depth.sell_orders)
        total_levels = buy_levels + sell_levels
        
        # 计算总体流动性
        total_buy_volume = sum(order_depth.buy_orders.values())
        total_sell_volume = abs(sum(order_depth.sell_orders.values()))
        total_volume = total_buy_volume + total_sell_volume
        
        # 计算集中度 - 前2个价格水平占总量的比例
        top_buy_volume = sum([v for k, v in sorted(order_depth.buy_orders.items(), reverse=True)[:2]])
        top_sell_volume = abs(sum([v for k, v in sorted(order_depth.sell_orders.items())[:2]]))
        concentration = (top_buy_volume + top_sell_volume) / total_volume if total_volume > 0 else 0
        
        # 深度评分 - 考虑价格水平数和每个水平的平均体量
        depth_score = total_levels * np.log1p(total_volume / total_levels) if total_levels > 0 else 0
        
        # 综合流动性评分
        liquidity_score = np.log1p(total_volume) * (1 - concentration)
        
        return {
            "liquidity_score": liquidity_score,
            "depth_score": depth_score,
            "price_levels": total_levels,
            "concentration": concentration
        }

    def detect_price_pressure(self, order_depth: OrderDepth) -> float:
        """检测价格压力 - 新增方法，专注于价格压力分析"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # 计算买方和卖方在最优价格附近的累积订单量
        buy_near_best = 0
        sell_near_best = 0
        
        for price, quantity in order_depth.buy_orders.items():
            # 考虑距离最优买价10%范围内的订单
            if price >= best_bid * 0.9:
                buy_near_best += quantity
                
        for price, quantity in order_depth.sell_orders.items():
            # 考虑距离最优卖价10%范围内的订单
            if price <= best_ask * 1.1:
                sell_near_best += abs(quantity)
                
        # 计算买卖压力比
        if buy_near_best + sell_near_best > 0:
            pressure = (buy_near_best - sell_near_best) / (buy_near_best + sell_near_best)
        else:
            pressure = 0
            
        return pressure
                
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
                # 高波动率时增加交易量 - 适合微观结构策略
                adjusted_max_trade = int(PARAMS['max_trade_quantity'] * 1.3)
                adjusted_spread_threshold = PARAMS['spread_threshold'] * 0.9 # 降低以更频繁交易
            else:
                # 低波动率时正常交易
                adjusted_max_trade = PARAMS['max_trade_quantity']
                adjusted_spread_threshold = PARAMS['spread_threshold']
        else:
            return result, conversions, state.traderData
            
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
        
        # 计算策略权重 - 偏向于策略2
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
        
        # 新增：详细分析订单簿结构
        book_structure = self.analyze_order_book_structure(order_depth)
        price_pressure = self.detect_price_pressure(order_depth)
        
        # 计算趋势指标
        price_trend = 0
        if len(self.price_history) >= 3:
            price_trend = np.sign(self.price_history[-1] - self.price_history[-3])
        
        # 确定交易方向
        orders = []
        
        # 策略1信号 - 基本的信号
        strategy1_buy = (price_score > 0.55 and ma_short > ma_long)
        strategy1_sell = (price_score < 0.45 and ma_short < ma_long)
        
        # 策略2信号 - 增强版，包含多种微观结构指标
        strategy2_buy = ((sell_pressure > PARAMS['pressure_threshold'] or 
                          price_pressure < -0.2 or
                          (imbalance < -PARAMS['imbalance_ratio'] and book_structure["liquidity_score"] > 1)) and
                         (relative_spread < adjusted_spread_threshold or book_structure["concentration"] > 0.7) and
                         (sell_flow > buy_flow * PARAMS['flow_ratio'] or market_trend > 0))
        
        strategy2_sell = ((buy_pressure > PARAMS['pressure_threshold'] or 
                           price_pressure > 0.2 or
                           (imbalance > PARAMS['imbalance_ratio'] and book_structure["liquidity_score"] > 1)) and
                          (relative_spread < adjusted_spread_threshold or book_structure["concentration"] > 0.7) and
                          (buy_flow > sell_flow * PARAMS['flow_ratio'] or market_trend < 0))
        
        # 综合信号 - 策略2权重更大
        should_buy = ((strategy1_buy and strategy1_weight > PARAMS['strategy_weight_threshold']) or 
                        (strategy2_buy and strategy2_weight > 0.3))  # 降低策略2的触发门槛
        
        should_sell = ((strategy1_sell and strategy1_weight > PARAMS['strategy_weight_threshold']) or 
                         (strategy2_sell and strategy2_weight > 0.3))  # 降低策略2的触发门槛
        
        # 执行买入
        if should_buy:
            available_buy = PARAMS['position_limit'] - current_position
            if available_buy > 0:
                buy_price = best_ask
                
                # 根据信号强度决定交易量
                signal_strength = max(
                    strategy1_weight if strategy1_buy else 0,
                    strategy2_weight if strategy2_buy else 0
                )
                
                # 交易量随信号强度变化
                buy_quantity = min(available_buy, int(adjusted_max_trade * (0.5 + 0.5 * signal_strength)))
                
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                self.last_trade_side = "BUY"
                self.last_trade_price = buy_price
                self.consecutive_trades = self.consecutive_trades + 1 if self.last_trade_side == "BUY" else 1
                
        # 执行卖出
        elif should_sell:
            available_sell = PARAMS['position_limit'] + current_position
            if available_sell > 0:
                sell_price = best_bid
                
                # 根据信号强度决定交易量
                signal_strength = max(
                    strategy1_weight if strategy1_sell else 0,
                    strategy2_weight if strategy2_sell else 0
                )
                
                # 交易量随信号强度变化
                sell_quantity = min(available_sell, int(adjusted_max_trade * (0.5 + 0.5 * signal_strength)))
                
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                self.last_trade_side = "SELL"
                self.last_trade_price = sell_price
                self.consecutive_trades = self.consecutive_trades + 1 if self.last_trade_side == "SELL" else 1
                
        # 平仓逻辑 - 基于微观结构指标
        elif (abs(imbalance) < PARAMS['imbalance_threshold'] * 0.5 and  # 更严格的不平衡阈值
              (relative_spread > adjusted_spread_threshold * PARAMS['spread_multiplier'] or 
               book_structure["concentration"] < 0.4 or  # 低集中度可能表示方向不明确
               abs(price_score - PARAMS['price_score_neutral']) < PARAMS['price_score_threshold'])):
            
            if current_position > 0:
                sell_price = best_bid
                # 根据持仓量决定平仓比例，持仓越大平仓比例越大
                position_ratio = min(1.0, current_position / 30)
                sell_quantity = min(current_position, int(adjusted_max_trade * (0.5 + 0.5 * position_ratio)))
                orders.append(Order("MAGNIFICENT_MACARONS", sell_price, -sell_quantity))
                self.last_trade_side = "SELL"
                self.last_trade_price = sell_price
                self.consecutive_trades = 1
                
            elif current_position < 0:
                buy_price = best_ask
                # 根据持仓量决定平仓比例，持仓越大平仓比例越大
                position_ratio = min(1.0, abs(current_position) / 30)
                buy_quantity = min(abs(current_position), int(adjusted_max_trade * (0.5 + 0.5 * position_ratio)))
                orders.append(Order("MAGNIFICENT_MACARONS", buy_price, buy_quantity))
                self.last_trade_side = "BUY"
                self.last_trade_price = buy_price
                self.consecutive_trades = 1
        
        # 检查是否需要转换
        if self.should_convert(state, "MAGNIFICENT_MACARONS"):
            conversions = min(abs(current_position), PARAMS['conversion_limit'])
            if current_position < 0:
                conversions = -conversions
        
        result["MAGNIFICENT_MACARONS"] = orders
        return result, conversions, str(state.timestamp)  # 使用时间戳作为trader data 