import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# 全局参数配置 - 变种4：订单簿不平衡与跨市场套利策略
PARAMS = {
    # 基础参数
    'position_limit': 75,  # 仓位限制
    'conversion_limit': 10,  # 转换限制
    
    # 订单簿不平衡参数
    'imbalance_depth_threshold': 5,  # 订单簿深度检查档位数
    'strong_imbalance_threshold': 0.15,  # 强不平衡阈值（比率）
    'mild_imbalance_threshold': 0.08,  # 轻微不平衡阈值（比率）
    'weight_decay': 0.8,  # 越远离最佳价格，权重递减
    
    # 套利参数
    'min_arb_profit': 0.0,  # 最小套利利润（理论上可以接受0利润）
    'min_spread_threshold': 0.001,  # 最小价差阈值（0.1%）
    'conversion_cooldown': 0,  # 转换冷却时间（几乎无冷却）
    
    # 交易执行参数
    'imbalance_trade_size_strong': 0.7,  # 强不平衡交易比例
    'imbalance_trade_size_mild': 0.4,  # 轻微不平衡交易比例
    'market_making_spread': 0.002,  # 做市商价差（0.2%）
    'aggressive_threshold': 0.3,  # 激进阈值
    
    # 动态调整参数
    'price_impact_factor': 0.8,  # 价格影响因子
    'imbalance_history_window': 10,  # 不平衡历史窗口
}

class Trader:
    def __init__(self):
        # 初始化参数
        self.position_limit = PARAMS['position_limit']
        self.conversion_limit = PARAMS['conversion_limit']
        
        # 历史数据
        self.price_history = []
        self.imbalance_history = []
        self.spread_history = []
        self.conversion_history = []
        
        # 套利相关
        self.last_conversion_time = 0
        self.current_time = 0
        self.consecutive_conversions = 0
        
        # 市场数据
        self.fair_price_estimate = None
        self.market_volatility = 0.01  # 初始波动率估计
    
    def update_time(self):
        """更新内部时间计数器"""
        self.current_time += 1
    
    def calculate_order_book_imbalance(self, order_depth: OrderDepth) -> Tuple[float, float, float]:
        """
        计算订单簿不平衡度
        返回：(总体不平衡度, 买盘权重, 卖盘权重)
        不平衡度在-1到1之间，正值表示买方占优，负值表示卖方占优
        """
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0.0, 0.0, 0.0
        
        # 获取最佳价格
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # 计算加权订单量
        buy_volume_weighted = 0
        sell_volume_weighted = 0
        
        # 买盘加权计算（越接近最高买价权重越高）
        buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)[:PARAMS['imbalance_depth_threshold']]
        for i, price in enumerate(buy_prices):
            weight = PARAMS['weight_decay'] ** i  # 权重随距离衰减
            buy_volume_weighted += abs(order_depth.buy_orders[price]) * weight * price
        
        # 卖盘加权计算（越接近最低卖价权重越高）
        sell_prices = sorted(order_depth.sell_orders.keys())[:PARAMS['imbalance_depth_threshold']]
        for i, price in enumerate(sell_prices):
            weight = PARAMS['weight_decay'] ** i  # 权重随距离衰减
            sell_volume_weighted += abs(order_depth.sell_orders[price]) * weight * price
        
        # 计算总权重量
        total_weighted = buy_volume_weighted + sell_volume_weighted
        
        if total_weighted == 0:
            return 0.0, 0.0, 0.0
        
        # 不平衡比率（-1到1）
        imbalance = (buy_volume_weighted - sell_volume_weighted) / total_weighted
        
        # 计算买卖盘权重
        buy_weight = buy_volume_weighted / total_weighted
        sell_weight = sell_volume_weighted / total_weighted
        
        return imbalance, buy_weight, sell_weight
    
    def check_conversion_arbitrage(self, state: TradingState) -> Tuple[bool, int, float]:
        """
        检查是否存在套利机会
        返回：(是否有套利, 套利方向(1:买入岛内卖出外部, -1:买入外部卖出岛内), 预期利润)
        """
        # 获取交易所订单簿
        if "MAGNIFICENT_MACARONS" not in state.order_depths:
            return False, 0, 0
            
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        
        # 确保订单簿有买卖盘
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return False, 0, 0
            
        # 获取岛内最佳价格
        island_best_bid = max(order_depth.buy_orders.keys())
        island_best_ask = min(order_depth.sell_orders.keys())
        
        # 获取外部厨师报价
        external_bid = 0
        external_ask = 0
        
        # 确保有转换观察数据
        chef_observation = getattr(state.observations, 'pristineCuisineConversion', None)
        if not chef_observation:
            return False, 0, 0
        
        # 从观察对象获取厨师报价
        if isinstance(chef_observation, ConversionObservation):
            external_bid = chef_observation.bid or 0
            external_ask = chef_observation.ask or 0
        else:
            return False, 0, 0
        
        # 获取费用信息
        import_tariff = getattr(state.observations, 'importTariff', 0)
        transport_fee = getattr(state.observations, 'transportFee', 0)
        
        # 计算有效外部价格（考虑费用）
        effective_external_ask = external_ask + transport_fee + import_tariff  # 从外部买入的有效价格
        effective_external_bid = external_bid - transport_fee - import_tariff  # 卖给外部的有效价格
        
        # 检查套利机会
        arb_direction = 0
        expected_profit = 0
        
        # 买岛内卖外部
        if island_best_ask < effective_external_bid:
            arb_direction = 1
            expected_profit = effective_external_bid - island_best_ask
        
        # 买外部卖岛内
        elif effective_external_ask < island_best_bid:
            arb_direction = -1
            expected_profit = island_best_bid - effective_external_ask
        
        # 判断是否有足够利润
        if expected_profit > PARAMS['min_arb_profit']:
            return True, arb_direction, expected_profit
        
        return False, 0, 0
    
    def execute_conversion_arbitrage(self, state: TradingState, direction: int, profit: float) -> Tuple[List[Order], int]:
        """
        执行套利交易
        返回：(交易订单列表, 转换次数)
        """
        orders = []
        conversions = 0
        
        # 获取当前仓位
        current_position = state.position.get("MAGNIFICENT_MACARONS", 0)
        
        # 获取订单簿
        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]
        
        # 判断冷却时间
        if self.current_time - self.last_conversion_time < PARAMS['conversion_cooldown']:
            return orders, conversions
        
        # 方向1：买岛内卖外部
        if direction == 1:
            # 确定可以买入的数量（考虑仓位限制和转换限制）
            available_to_buy = min(
                self.position_limit - current_position,  # 仓位限制
                self.conversion_limit  # 单次转换限制
            )
            
            if available_to_buy <= 0:
                return orders, conversions
            
            # 买入岛内
            best_ask = min(order_depth.sell_orders.keys())
            ask_volume = abs(order_depth.sell_orders[best_ask])
            trade_size = min(available_to_buy, ask_volume)
            
            if trade_size > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", best_ask, trade_size))
                conversions = trade_size  # 所有买入都将被转换
                self.last_conversion_time = self.current_time
                self.consecutive_conversions += 1
        
        # 方向-1：买外部卖岛内
        elif direction == -1:
            # 确定可以卖出的数量（考虑仓位限制和转换限制）
            available_to_sell = min(
                self.position_limit + current_position,  # 仓位限制
                self.conversion_limit  # 单次转换限制
            )
            
            if available_to_sell <= 0:
                return orders, conversions
            
            # 先转换（买入外部），然后卖出岛内
            best_bid = max(order_depth.buy_orders.keys())
            bid_volume = abs(order_depth.buy_orders[best_bid])
            trade_size = min(available_to_sell, bid_volume)
            
            if trade_size > 0:
                orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -trade_size))
                conversions = -trade_size  # 负数表示将被转换的"买入"数量
                self.last_conversion_time = self.current_time
                self.consecutive_conversions += 1
        
        return orders, conversions
    
    def trade_on_imbalance(self, order_depth: OrderDepth, current_position: int, imbalance: float) -> List[Order]:
        """根据订单簿不平衡交易"""
        orders = []
        
        # 没有明显不平衡
        if abs(imbalance) < PARAMS['mild_imbalance_threshold']:
            return orders
        
        # 获取最佳价格
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # 强烈不平衡 - 使用更激进的仓位
        if abs(imbalance) > PARAMS['strong_imbalance_threshold']:
            # 买方力量强（卖盘将被吃掉）
            if imbalance > 0:
                # 如果我们有能力，顺势做多
                available_buy = self.position_limit - current_position
                if available_buy > 0:
                    # 确定交易量
                    trade_size = int(available_buy * PARAMS['imbalance_trade_size_strong'])
                    if trade_size > 0:
                        # 市价单吃掉卖盘
                        orders.append(Order("MAGNIFICENT_MACARONS", best_ask, trade_size))
            # 卖方力量强（买盘将被吃掉）
            else:
                # 如果我们有能力，顺势做空
                available_sell = self.position_limit + current_position
                if available_sell > 0:
                    # 确定交易量
                    trade_size = int(available_sell * PARAMS['imbalance_trade_size_strong'])
                    if trade_size > 0:
                        # 市价单吃掉买盘
                        orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -trade_size))
        
        # 轻微不平衡 - 使用较保守的仓位
        else:
            # 买方轻微占优
            if imbalance > 0:
                available_buy = self.position_limit - current_position
                if available_buy > 0:
                    trade_size = int(available_buy * PARAMS['imbalance_trade_size_mild'])
                    if trade_size > 0:
                        orders.append(Order("MAGNIFICENT_MACARONS", best_ask, trade_size))
            # 卖方轻微占优
            else:
                available_sell = self.position_limit + current_position
                if available_sell > 0:
                    trade_size = int(available_sell * PARAMS['imbalance_trade_size_mild'])
                    if trade_size > 0:
                        orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -trade_size))
        
        return orders
    
    def estimate_fair_price(self, state: TradingState, order_depth: OrderDepth) -> float:
        """估算公平价格，考虑外部因素和订单簿"""
        # 获取中间价
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return self.fair_price_estimate if self.fair_price_estimate else 0
            
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2
        
        # 获取相关因素
        sugar_price = getattr(state.observations, 'sugarPrice', 0)
        sunlight_index = getattr(state.observations, 'sunlightIndex', 0)
        import_tariff = getattr(state.observations, 'importTariff', 0)
        
        # 获取厨师报价
        chef_observation = getattr(state.observations, 'pristineCuisineConversion', None)
        external_mid = 0
        
        if chef_observation and isinstance(chef_observation, ConversionObservation):
            external_bid = chef_observation.bid or 0
            external_ask = chef_observation.ask or 0
            if external_bid > 0 and external_ask > 0:
                external_mid = (external_bid + external_ask) / 2
        
        # 简单的公平价值模型
        if not self.fair_price_estimate:
            # 第一次初始化，使用中间价
            self.fair_price_estimate = mid_price
            return mid_price
            
        # 组合各种信息源估算公平价格
        # 权重分配：市场中间价(50%), 外部价格(30%), 基本面因素(20%)
        market_weight = 0.5
        external_weight = 0.3
        fundamental_weight = 0.2
        
        # 基本面估值（简单的线性组合）
        fundamental_price = (
            sugar_price * 0.4 +  # 糖价影响
            sunlight_index * 0.4 -  # 阳光指数影响（负相关）
            import_tariff * 0.2    # 关税影响（负相关）
        )
        
        # 如果没有外部价格，调整权重
        if external_mid == 0:
            market_weight = 0.7
            external_weight = 0
            fundamental_weight = 0.3
        
        # 计算加权平均
        fair_price = (
            mid_price * market_weight +
            (external_mid * external_weight if external_mid else 0) +
            fundamental_price * fundamental_weight
        )
        
        # 平滑更新
        self.fair_price_estimate = self.fair_price_estimate * 0.7 + fair_price * 0.3
        
        return self.fair_price_estimate
    
    def make_market_orders(self, order_depth: OrderDepth, current_position: int, fair_price: float) -> List[Order]:
        """基于估计的公平价格，提供做市商挂单"""
        orders = []
        
        # 如果没有有效的公平价格或订单簿为空，返回空列表
        if not fair_price or not order_depth.buy_orders or not order_depth.sell_orders:
            return orders
            
        # 获取最佳价格
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # 计算市场价差
        market_spread = best_ask - best_bid
        relative_spread = market_spread / best_bid
        
        # 如果市场价差已经很窄，不做市
        if relative_spread < PARAMS['min_spread_threshold']:
            return orders
        
        # 确定做市价差（基于波动率动态调整）
        making_spread = fair_price * max(PARAMS['market_making_spread'], self.market_volatility * 2)
        
        # 确定我们的挂单价格
        our_bid = fair_price - making_spread / 2
        our_ask = fair_price + making_spread / 2
        
        # 如果我们的价格优于市场最佳价格，调整到略优于市场
        if our_bid >= best_bid:
            our_bid = best_bid + 0.01  # 略高于最佳买价
        
        if our_ask <= best_ask:
            our_ask = best_ask - 0.01  # 略低于最佳卖价
        
        # 确定挂单数量（基于当前仓位）
        bid_size = 0
        ask_size = 0
        
        # 当仓位偏空时，买单量更大；当仓位偏多时，卖单量更大
        position_ratio = current_position / self.position_limit if self.position_limit > 0 else 0
        
        if position_ratio < 0:  # 负仓位（空头）
            # 更倾向于买入平仓
            bid_size = int(min(5, self.position_limit + current_position))
            ask_size = int(min(2, self.position_limit - current_position))
        elif position_ratio > 0:  # 正仓位（多头）
            # 更倾向于卖出平仓
            bid_size = int(min(2, self.position_limit + current_position))
            ask_size = int(min(5, self.position_limit - current_position))
        else:  # 中性仓位
            # 买卖均衡
            bid_size = int(min(3, self.position_limit + current_position))
            ask_size = int(min(3, self.position_limit - current_position))
        
        # 添加订单
        if bid_size > 0:
            orders.append(Order("MAGNIFICENT_MACARONS", our_bid, bid_size))
            
        if ask_size > 0:
            orders.append(Order("MAGNIFICENT_MACARONS", our_ask, -ask_size))
        
        return orders
    
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
        
        # 更新历史价格
        self.price_history.append(mid_price)
        
        # 计算波动率
        if len(self.price_history) >= 5:
            recent_returns = np.diff(self.price_history[-5:]) / self.price_history[-6:-1]
            self.market_volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0.01
        
        # 计算订单簿不平衡度
        imbalance, buy_weight, sell_weight = self.calculate_order_book_imbalance(order_depth)
        self.imbalance_history.append(imbalance)
        
        # 估算公平价格
        fair_price = self.estimate_fair_price(state, order_depth)
        
        # 优先级1：检查是否有套利机会
        arb_exists, arb_direction, arb_profit = self.check_conversion_arbitrage(state)
        
        orders = []
        
        if arb_exists:
            # 如果有套利机会，优先执行套利
            arb_orders, conv = self.execute_conversion_arbitrage(state, arb_direction, arb_profit)
            orders.extend(arb_orders)
            conversions = conv
        else:
            # 优先级2：根据订单簿不平衡交易
            imbalance_orders = self.trade_on_imbalance(order_depth, current_position, imbalance)
            orders.extend(imbalance_orders)
            
            # 如果不平衡交易未产生订单，考虑做市
            if not imbalance_orders:
                # 优先级3：做市商挂单
                market_making_orders = self.make_market_orders(order_depth, current_position, fair_price)
                orders.extend(market_making_orders)
        
        result["MAGNIFICENT_MACARONS"] = orders
        return result, conversions, state.traderData 