from typing import Dict, List, Tuple, Optional, Set
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np

# 产品限制
LIMIT = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75
}

# 产品分类
PRODUCT_CATEGORIES = {
    # 火山岩及其期权 - 使用方向性策略
    "volcanic": {
        "VOLCANIC_ROCK", 
        "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
        "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"
    },
    # 高表现产品 - 使用激进做市策略
    "high_performers": {
        "RAINFOREST_RESIN", "CROISSANTS", "MAGNIFICENT_MACARONS"
    },
    # 中等表现产品 - 使用标准做市策略
    "medium_performers": {
        "KELP", "PICNIC_BASKET2", "JAMS"
    },
    # 低表现产品 - 使用保守做市策略或避开
    "low_performers": {
        "SQUID_INK", "PICNIC_BASKET1", "DJEMBES"
    }
}

# 参数设置
PARAM = {
    # 基础参数
    "base_spread": 2,
    "vol_factor": 1.0,
    "position_limit_pct": 0.8,  # 使用80%的仓位限制
    
    # 产品类别参数
    "categories": {
        "volcanic": {
            "spread_multiplier": 1.0,
            "size_limit": 0.5,
            "aggr_take": True,
            "trend_threshold": 0.65,
            "signal_strength_threshold": 2.0
        },
        "high_performers": {
            "spread_multiplier": 0.8,  # 更窄的价差
            "size_limit": 0.7,         # 更大的规模
            "aggr_take": True
        },
        "medium_performers": {
            "spread_multiplier": 1.2,
            "size_limit": 0.5,
            "aggr_take": True
        },
        "low_performers": {
            "spread_multiplier": 1.5,  # 更宽的价差
            "size_limit": 0.3,         # 更小的规模
            "aggr_take": False
        }
    },
    
    # 技术指标参数
    "trend_window_short": 10,
    "trend_window_medium": 20,
    "trend_window_long": 40,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "vol_window": 15
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.returns = defaultdict(list)  # 价格回报率
        self.position_history = defaultdict(list)  # 仓位历史
        self.fair_values = {}  # 产品公平价值
        self.trends = defaultdict(int)  # 1=上升, -1=下降, 0=中性
        self.signal_strength = defaultdict(float)  # 信号强度
        self.rsi_values = defaultdict(float)  # RSI值
        self.product_performance = defaultdict(float)  # 产品表现评分
        self.last_mid_price = {}  # 上一个中间价
        self.option_deltas = defaultdict(float)  # 期权Delta值
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < PARAM["vol_window"]:
            return 2  # 默认中等波动率
        return statistics.stdev(h[-PARAM["vol_window"]:]) * PARAM["vol_factor"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def get_product_category(self, product: str) -> str:
        """获取产品所属类别"""
        for category, products in PRODUCT_CATEGORIES.items():
            if product in products:
                return category
        return "medium_performers"  # 默认为中等表现产品
    
    def update_technical_indicators(self, state: TradingState):
        """更新技术指标"""
        for p, depth in state.order_depths.items():
            mid = self._mid(depth)
            if mid is None:
                continue
                
            # 记录价格
            self.prices[p].append(mid)
            self.fair_values[p] = mid
            
            # 记录仓位
            self.position_history[p].append(state.position.get(p, 0))
            
            # 计算回报率
            if p in self.last_mid_price:
                ret = (mid / self.last_mid_price[p]) - 1
                self.returns[p].append(ret)
            
            # 更新上一个中间价
            self.last_mid_price[p] = mid
            
            # 保持价格和回报率历史记录在合理范围内
            if len(self.prices[p]) > 100:
                self.prices[p].pop(0)
            if len(self.returns[p]) > 100:
                self.returns[p].pop(0)
            
            # 计算趋势
            self.trends[p] = self.detect_trend(p)
            
            # 计算RSI
            self.calculate_rsi(p)
            
            # 计算信号强度
            self.calculate_signal_strength(p)
            
            # 如果是火山岩期权，计算Delta值
            if p.startswith("VOLCANIC_ROCK_VOUCHER"):
                self.calculate_option_delta(p)
    
    def calculate_rsi(self, product: str):
        """计算相对强弱指数(RSI)"""
        if len(self.returns[product]) < PARAM["rsi_period"]:
            self.rsi_values[product] = 50  # 默认中性值
            return
            
        # 计算上涨和下跌的平均值
        gains = [max(0, ret) for ret in self.returns[product][-PARAM["rsi_period"]:]]
        losses = [max(0, -ret) for ret in self.returns[product][-PARAM["rsi_period"]:]]
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # 计算相对强度
        if avg_loss == 0:
            rs = 100
        else:
            rs = avg_gain / avg_loss
            
        # 计算RSI
        self.rsi_values[product] = 100 - (100 / (1 + rs))
    
    def calculate_option_delta(self, option: str):
        """计算期权的Delta值"""
        if "VOLCANIC_ROCK" not in self.fair_values:
            self.option_deltas[option] = 0.5  # 默认中性Delta
            return
            
        # 从期权名称中提取行权价
        try:
            strike = int(option.split("_")[-1])
            underlying_price = self.fair_values["VOLCANIC_ROCK"]
            
            # 简化的Delta计算
            if underlying_price > strike:  # 实值期权
                delta = 0.8
            elif underlying_price < strike - 200:  # 深度虚值期权
                delta = 0.2
            else:  # 平值附近的期权
                delta = 0.5
            
            self.option_deltas[option] = delta
        except (ValueError, KeyError):
            self.option_deltas[option] = 0.5  # 默认中性Delta
    
    def detect_trend(self, product: str) -> int:
        """检测市场趋势"""
        prices = self.prices[product]
        
        # 如果数据不足，返回中性
        if len(prices) < PARAM["trend_window_long"]:
            return 0
            
        # 计算短期、中期和长期趋势
        short_trend = self.calculate_trend_direction(prices, PARAM["trend_window_short"])
        medium_trend = self.calculate_trend_direction(prices, PARAM["trend_window_medium"])
        long_trend = self.calculate_trend_direction(prices, PARAM["trend_window_long"])
        
        # 综合趋势判断
        if short_trend == medium_trend == long_trend:
            # 三个时间框架趋势一致，强信号
            return short_trend
        elif short_trend == medium_trend:
            # 短期和中期趋势一致，中等信号
            return short_trend
        elif medium_trend == long_trend:
            # 中期和长期趋势一致，中等信号
            return medium_trend
        else:
            # 趋势不一致，弱信号或中性
            return short_trend if abs(short_trend) > abs(medium_trend) else medium_trend
    
    def calculate_trend_direction(self, prices: List[float], window: int) -> int:
        """计算指定窗口的趋势方向"""
        if len(prices) < window:
            return 0
            
        recent_prices = prices[-window:]
        
        # 计算上涨和下跌的比例
        up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
        
        up_ratio = up_moves / (len(recent_prices) - 1)
        down_ratio = down_moves / (len(recent_prices) - 1)
        
        category_params = PARAM["categories"]["volcanic"]
        threshold = category_params["trend_threshold"]
        
        if up_ratio > threshold:
            return 1  # 上升趋势
        elif down_ratio > threshold:
            return -1  # 下降趋势
        return 0  # 中性
    
    def calculate_signal_strength(self, product: str):
        """计算信号强度"""
        # 初始信号强度为0
        strength = 0
        
        # 1. 趋势信号
        trend = self.trends[product]
        strength += trend * 1.0  # 趋势贡献
        
        # 2. RSI信号
        rsi = self.rsi_values[product]
        if rsi > PARAM["rsi_overbought"]:
            strength -= 1.0  # 超买，看跌信号
        elif rsi < PARAM["rsi_oversold"]:
            strength += 1.0  # 超卖，看涨信号
        
        # 3. 价格动量
        if len(self.returns[product]) >= 5:
            momentum = sum(self.returns[product][-5:])
            if momentum > 0:
                strength += 0.5  # 正动量，看涨信号
            else:
                strength -= 0.5  # 负动量，看跌信号
        
        self.signal_strength[product] = strength
    
    def update_product_performance(self, state: TradingState):
        """更新产品表现评分"""
        for p, pos in state.position.items():
            if p not in self.prices or len(self.prices[p]) < 2:
                continue
                
            # 计算价格变动
            current_price = self.prices[p][-1]
            previous_price = self.prices[p][-2]
            price_change = (current_price - previous_price) / previous_price
            
            # 计算仓位价值变化
            position_value_change = pos * price_change
            
            # 更新表现评分 (指数移动平均)
            alpha = 0.3  # 平滑因子
            self.product_performance[p] = (alpha * position_value_change + 
                                          (1 - alpha) * self.product_performance.get(p, 0))
    
    def volcanic_directional_strategy(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """火山岩方向性策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 获取类别参数
        category_params = PARAM["categories"]["volcanic"]
        
        # 计算波动率
        vol = self._vol(p)
        
        # 获取信号强度
        strength = self.signal_strength.get(p, 0)
        
        # 如果信号强度不够强，使用常规做市策略
        if abs(strength) < category_params["signal_strength_threshold"]:
            return self.standard_market_making(p, depth, pos, "volcanic")
        
        # 确定交易方向
        direction = 1 if strength > 0 else -1
        
        # 计算价差
        spread = int(PARAM["base_spread"] + vol * category_params["spread_multiplier"])
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 根据方向调整价格和规模
        max_position = int(LIMIT[p] * PARAM["position_limit_pct"] * category_params["size_limit"])
        
        if direction == 1:  # 看涨
            # 更积极地买入
            buy_px = int(mid - spread * 0.5)  # 提高买入价格
            sell_px = int(mid + spread * 1.5)  # 提高卖出价格
            size_buy = int(max_position * 0.7)  # 增加买入规模
            size_sell = int(max_position * 0.3)  # 减少卖出规模
        else:  # 看跌
            # 更积极地卖出
            buy_px = int(mid - spread * 1.5)  # 降低买入价格
            sell_px = int(mid + spread * 0.5)  # 降低卖出价格
            size_buy = int(max_position * 0.3)  # 减少买入规模
            size_sell = int(max_position * 0.7)  # 增加卖出规模
        
        # 主动吃单
        if category_params["aggr_take"]:
            b, a = best_bid_ask(depth)
            if a is not None and b is not None:
                # 如果卖单价格低于我们的买入价，主动买入
                if direction == 1 and a < buy_px and pos < max_position:
                    qty = min(size_buy, max_position - pos, abs(depth.sell_orders[a]))
                    if qty > 0: orders.append(Order(p, a, qty))
                
                # 如果买单价格高于我们的卖出价，主动卖出
                if direction == -1 and b > sell_px and pos > -max_position:
                    qty = min(size_sell, max_position + pos, depth.buy_orders[b])
                    if qty > 0: orders.append(Order(p, b, -qty))
        
        # 常规做市订单
        if pos < max_position:
            orders.append(Order(p, buy_px, min(size_buy, max_position - pos)))
        if pos > -max_position:
            orders.append(Order(p, sell_px, -min(size_sell, max_position + pos)))
        
        return orders
    
    def standard_market_making(self, p: str, depth: OrderDepth, pos: int, category: str) -> List[Order]:
        """标准做市策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 获取类别参数
        category_params = PARAM["categories"][category]
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差
        spread = int(PARAM["base_spread"] + vol * category_params["spread_multiplier"])
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 根据趋势调整价格
        trend = self.trends.get(p, 0)
        if trend == 1:  # 上升趋势
            buy_px += 1  # 更积极地买入
            sell_px += 1  # 更保守地卖出
        elif trend == -1:  # 下降趋势
            buy_px -= 1  # 更保守地买入
            sell_px -= 1  # 更积极地卖出
        
        # 计算交易规模
        max_position = int(LIMIT[p] * PARAM["position_limit_pct"] * category_params["size_limit"])
        size = max(1, max_position // 4)  # 每次用1/4的允许仓位
        
        # 主动吃单
        if category_params["aggr_take"]:
            b, a = best_bid_ask(depth)
            if a is not None and b is not None:
                # 如果卖单价格低于我们的买入价，主动买入
                if a < buy_px and pos < max_position:
                    qty = min(size, max_position - pos, abs(depth.sell_orders[a]))
                    if qty > 0: orders.append(Order(p, a, qty))
                
                # 如果买单价格高于我们的卖出价，主动卖出
                if b > sell_px and pos > -max_position:
                    qty = min(size, max_position + pos, depth.buy_orders[b])
                    if qty > 0: orders.append(Order(p, b, -qty))
        
        # 常规做市订单
        if pos < max_position:
            orders.append(Order(p, buy_px, min(size, max_position - pos)))
        if pos > -max_position:
            orders.append(Order(p, sell_px, -min(size, max_position + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 更新技术指标
        self.update_technical_indicators(state)
        
        # 更新产品表现评分
        self.update_product_performance(state)
        
        # 对每个产品应用相应的策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                category = self.get_product_category(p)
                pos = state.position.get(p, 0)
                
                if category == "volcanic":
                    # 对火山岩及其期权使用方向性策略
                    result[p] = self.volcanic_directional_strategy(p, depth, pos)
                else:
                    # 对其他产品使用标准做市策略
                    result[p] = self.standard_market_making(p, depth, pos, category)
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
