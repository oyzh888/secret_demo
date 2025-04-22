from typing import Dict, List, Tuple, Optional
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

# 产品分类 - 根据波动性将产品分成不同类别，每类产品使用不同的RSI设置
PRODUCT_TIERS = {
    # 高波动性产品
    "high_vol": {
        "products": {"VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
                    "PICNIC_BASKET1", "PICNIC_BASKET2"},
        "rsi_period": 9,         # 短周期，快速反应
        "rsi_overbought": 75,    # 更灵敏的超买判断
        "rsi_oversold": 25       # 更灵敏的超卖判断
    },
    # 中波动性产品
    "medium_vol": {
        "products": {"MAGNIFICENT_MACARONS", "VOLCANIC_ROCK_VOUCHER_10000", 
                    "VOLCANIC_ROCK_VOUCHER_10250", "SQUID_INK", "DJEMBES"},
        "rsi_period": 14,        # 中等周期
        "rsi_overbought": 70,    # 标准超买判断
        "rsi_oversold": 30       # 标准超卖判断
    },
    # 低波动性产品
    "low_vol": {
        "products": {"VOLCANIC_ROCK_VOUCHER_10500", "CROISSANTS", "JAMS", 
                    "KELP", "RAINFOREST_RESIN"},
        "rsi_period": 21,        # 长周期，减少假信号
        "rsi_overbought": 65,    # 更宽松的超买判断
        "rsi_oversold": 35       # 更宽松的超卖判断
    }
}

# 参数 - RSI敏感型设置
PARAM = {
    "trend_window": 30,       # 趋势检测窗口
    "trend_threshold": 0.7,   # 趋势检测阈值
    "reversal_window": 5,     # 反转检测窗口
    "reversal_threshold": 0.8, # 反转检测阈值
    "position_limit_pct": 0.6, # 仓位限制百分比
    "mm_size_frac": 0.15,     # 做市规模
    "counter_size_frac": 0.25, # 反趋势交易规模
    "min_spread": 2,          # 最小价差
    "vol_scale": 1.2,         # 波动率缩放因子
    "rsi_divergence_window": 5,  # RSI背离检测窗口
    "rsi_trend_confirm": True,  # 是否需要趋势确认RSI信号
    "rsi_max_cooldown": 15,     # RSI信号最大冷却期
    "rsi_scalar": 0.01,        # RSI偏离中值(50)程度对订单大小的影响因子
    "adaptive_thresholds": True # 是否使用自适应RSI阈值
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
        self.rsi_values = defaultdict(list)  # RSI值历史
        self.rsi_avg = defaultdict(float)   # RSI平均值
        self.rsi_std = defaultdict(float)   # RSI标准差
        self.last_counter_trade = defaultdict(int)  # 上次反向交易的时间戳
        self.product_tier_map = {}          # 产品到其分类的映射
        
        # 构建产品到分类的映射
        for tier, config in PRODUCT_TIERS.items():
            for product in config["products"]:
                self.product_tier_map[product] = tier
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) * PARAM["vol_scale"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def get_rsi_settings(self, p: str) -> dict:
        """获取产品的RSI设置"""
        tier = self.product_tier_map.get(p, "medium_vol")  # 默认为中等波动性
        return PRODUCT_TIERS[tier]
    
    def calculate_rsi(self, p: str):
        """计算相对强弱指数(RSI)"""
        rsi_settings = self.get_rsi_settings(p)
        rsi_period = rsi_settings["rsi_period"]
        
        if len(self.prices[p]) < rsi_period + 1:
            self.rsi_values[p].append(50)  # 默认中性值
            return 50
            
        # 计算价格变动
        price_changes = [self.prices[p][i] - self.prices[p][i-1] for i in range(1, len(self.prices[p]))]
        
        # 只使用最近的价格变动
        price_changes = price_changes[-rsi_period:]
        
        # 计算上涨和下跌的平均值
        gains = [max(0, change) for change in price_changes]
        losses = [max(0, -change) for change in price_changes]
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # 计算相对强度
        if avg_loss == 0:
            rs = 100
        else:
            rs = avg_gain / avg_loss
            
        # 计算RSI
        rsi = 100 - (100 / (1 + rs))
        self.rsi_values[p].append(rsi)
        
        # 更新RSI统计
        if len(self.rsi_values[p]) > rsi_period:
            self.rsi_avg[p] = sum(self.rsi_values[p][-rsi_period:]) / rsi_period
            if len(self.rsi_values[p]) > rsi_period + 1:
                self.rsi_std[p] = statistics.stdev(self.rsi_values[p][-rsi_period:])
            else:
                self.rsi_std[p] = 10  # 默认标准差
                
        return rsi
    
    def get_adaptive_rsi_thresholds(self, p: str) -> Tuple[float, float]:
        """获取自适应RSI阈值"""
        if not PARAM["adaptive_thresholds"] or p not in self.rsi_std:
            # 如果不使用自适应阈值或没有足够数据，使用静态阈值
            rsi_settings = self.get_rsi_settings(p)
            return rsi_settings["rsi_overbought"], rsi_settings["rsi_oversold"]
        
        # 使用RSI的均值和标准差计算动态阈值
        avg = self.rsi_avg[p]
        std = self.rsi_std[p]
        
        # 确保标准差在合理范围内
        std = max(5, min(20, std))
        
        # 计算自适应阈值
        overbought = min(85, avg + 1.5 * std)
        oversold = max(15, avg - 1.5 * std)
        
        return overbought, oversold
    
    def detect_rsi_divergence(self, p: str) -> int:
        """检测RSI背离"""
        if len(self.prices[p]) < PARAM["rsi_divergence_window"] or len(self.rsi_values[p]) < PARAM["rsi_divergence_window"]:
            return 0  # 数据不足
            
        # 获取最近的价格和RSI
        recent_prices = self.prices[p][-PARAM["rsi_divergence_window"]:]
        recent_rsi = self.rsi_values[p][-PARAM["rsi_divergence_window"]:]
        
        # 检查价格新高但RSI较低的情况 (顶背离)
        if recent_prices[-1] > max(recent_prices[:-1]) and recent_rsi[-1] < max(recent_rsi[:-1]):
            return -1  # 看跌背离
            
        # 检查价格新低但RSI较高的情况 (底背离)
        if recent_prices[-1] < min(recent_prices[:-1]) and recent_rsi[-1] > min(recent_rsi[:-1]):
            return 1  # 看涨背离
            
        return 0  # 无背离
    
    def detect_trend(self, p: str) -> int:
        """检测市场趋势"""
        prices = self.prices[p]
        if len(prices) < PARAM["trend_window"]:
            return 0  # 数据不足
            
        recent_prices = prices[-PARAM["trend_window"]:]  
        up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
        
        up_ratio = up_moves / (len(recent_prices) - 1)
        down_ratio = down_moves / (len(recent_prices) - 1)
        
        if up_ratio > PARAM["trend_threshold"]:
            return 1  # 上升趋势
        elif down_ratio > PARAM["trend_threshold"]:
            return -1  # 下降趋势
        return 0  # 中性
    
    def detect_reversal(self, p: str) -> int:
        """检测市场反转"""
        prices = self.prices[p]
        if len(prices) < PARAM["reversal_window"] + 5:
            return 0  # 数据不足
            
        # 检查之前的趋势
        prev_trend = self.detect_trend(p)
        if prev_trend == 0:
            return 0  # 没有明确趋势，无法判断反转
            
        # 检查最近的价格变动
        recent_prices = prices[-PARAM["reversal_window"]:]
        prev_prices = prices[-PARAM["reversal_window"]-5:-PARAM["reversal_window"]]
        
        if prev_trend == 1:  # 之前是上升趋势
            # 检查是否开始下跌
            down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
            down_ratio = down_moves / (len(recent_prices) - 1)
            
            if down_ratio > PARAM["reversal_threshold"]:
                return -1  # 上升趋势反转为下降
        
        elif prev_trend == -1:  # 之前是下降趋势
            # 检查是否开始上涨
            up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
            up_ratio = up_moves / (len(recent_prices) - 1)
            
            if up_ratio > PARAM["reversal_threshold"]:
                return 1  # 下降趋势反转为上升
                
        return 0  # 没有检测到反转
    
    def calculate_rsi_based_size(self, p: str, base_size: int) -> int:
        """根据RSI计算订单大小"""
        if not self.rsi_values[p]:
            return base_size
            
        current_rsi = self.rsi_values[p][-1]
        
        # RSI偏离中值(50)的程度
        rsi_deviation = abs(current_rsi - 50)
        
        # RSI越偏离50，订单规模越大
        size_multiplier = 1 + (rsi_deviation * PARAM["rsi_scalar"])
        
        # 计算调整后的规模
        adjusted_size = int(base_size * size_multiplier)
        
        return max(1, adjusted_size)
    
    def rsi_countertrend_strategy(self, p: str, depth: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        """基于RSI的反趋势交易策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 记录仓位
        self.position_history[p].append(pos)
        
        # 计算RSI
        current_rsi = self.calculate_rsi(p)
        
        # 获取自适应RSI阈值
        rsi_overbought, rsi_oversold = self.get_adaptive_rsi_thresholds(p)
        
        # 检测RSI背离
        rsi_divergence = self.detect_rsi_divergence(p)
        
        # 检测趋势和反转
        trend = self.detect_trend(p)
        reversal = self.detect_reversal(p)
        
        # 计算价差
        vol = self._vol(p)
        spread = max(PARAM["min_spread"], int(vol))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模
        max_position = int(LIMIT[p] * PARAM["position_limit_pct"])
        mm_size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"]))
        counter_size_base = max(1, int(LIMIT[p] * PARAM["counter_size_frac"]))
        
        # 根据RSI调整反趋势交易规模
        counter_size = self.calculate_rsi_based_size(p, counter_size_base)
        
        # 检查交易冷却期
        cooldown_active = timestamp - self.last_counter_trade.get(p, 0) < PARAM["rsi_max_cooldown"]
        
        # RSI信号交易逻辑
        if not cooldown_active:
            # RSI超买信号
            if current_rsi > rsi_overbought:
                # 如果需要趋势确认且趋势为上升，或者不需要趋势确认
                if not PARAM["rsi_trend_confirm"] or trend == 1:
                    # 强烈上升趋势，反向做空
                    sell_px = int(mid - 1)  # 降低卖出价格以确保成交
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders  # 只做反向交易，不做常规做市
                    
            # RSI超卖信号
            elif current_rsi < rsi_oversold:
                # 如果需要趋势确认且趋势为下降，或者不需要趋势确认
                if not PARAM["rsi_trend_confirm"] or trend == -1:
                    # 强烈下降趋势，反向做多
                    buy_px = int(mid + 1)  # 提高买入价格以确保成交
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders  # 只做反向交易，不做常规做市
            
            # RSI背离信号
            elif rsi_divergence != 0:
                if rsi_divergence == 1:  # 看涨背离
                    buy_px = int(mid + 1)
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders
                    
                elif rsi_divergence == -1:  # 看跌背离
                    sell_px = int(mid - 1)
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders
                    
            # 检查是否有明确的反转信号
            elif reversal != 0:
                if reversal == 1:  # 下降趋势反转为上升
                    # 积极买入
                    buy_px = int(mid + 1)
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders
                    
                elif reversal == -1:  # 上升趋势反转为下降
                    # 积极卖出
                    sell_px = int(mid - 1)
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders
        
        # 如果没有反趋势交易信号，执行常规做市
        # 根据趋势调整价格
        if trend == 1:  # 上升趋势
            buy_px += 1
            sell_px += 1
        elif trend == -1:  # 下降趋势
            buy_px -= 1
            sell_px -= 1
        
        # 常规做市订单
        if pos < max_position:
            orders.append(Order(p, buy_px, min(mm_size, max_position - pos)))
        if pos > -max_position:
            orders.append(Order(p, sell_px, -min(mm_size, max_position + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 对每个产品应用RSI驱动的反趋势策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.rsi_countertrend_strategy(
                    p, 
                    depth, 
                    state.position.get(p, 0),
                    state.timestamp
                )
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData 