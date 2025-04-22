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

# 高波动性产品 - 缩小范围，只对真正高波动的产品应用反趋势策略
HIGH_VOL_PRODUCTS = {
    "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750"
}

# 参数 - 更保守的设置
PARAM = {
    "trend_window": 40,       # 延长趋势检测窗口，更确保趋势
    "trend_threshold": 0.75,  # 提高趋势检测阈值，要求更明确的趋势
    "reversal_window": 8,     # 延长反转检测窗口，需要更明确的反转信号
    "reversal_threshold": 0.85, # 提高反转检测阈值，要求更明确的反转
    "position_limit_pct": 0.4, # 降低仓位限制百分比，减少风险
    "mm_size_frac": 0.2,      # 适中的做市规模
    "counter_size_frac": 0.15, # 降低反趋势交易规模，降低风险
    "min_spread": 3,          # 增加最小价差，增加利润空间
    "vol_scale": 1.0,         # 降低波动率缩放因子，更保守
    "rsi_period": 21,         # 延长RSI周期，减少假信号
    "rsi_overbought": 80,     # 提高RSI超买阈值，只在极端情况交易
    "rsi_oversold": 20        # 降低RSI超卖阈值，只在极端情况交易
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
        self.rsi_values = defaultdict(float)  # RSI值
        self.last_counter_trade = defaultdict(int)  # 上次反向交易的时间戳
        self.trade_cooldown = 20  # 添加交易冷却期，避免频繁交易
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) * PARAM["vol_scale"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def calculate_rsi(self, p: str):
        """计算相对强弱指数(RSI)"""
        if len(self.prices[p]) < PARAM["rsi_period"] + 1:
            self.rsi_values[p] = 50  # 默认中性值
            return
            
        # 计算价格变动
        price_changes = [self.prices[p][i] - self.prices[p][i-1] for i in range(1, len(self.prices[p]))]
        
        # 只使用最近的价格变动
        price_changes = price_changes[-PARAM["rsi_period"]:]
        
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
        self.rsi_values[p] = 100 - (100 / (1 + rs))
    
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
    
    def check_position_safety(self, p: str, pos: int, new_pos: int) -> bool:
        """检查新仓位是否安全"""
        # 确保新的仓位不会超过最大允许仓位的绝对值
        max_position = int(LIMIT[p] * PARAM["position_limit_pct"])
        return abs(new_pos) <= max_position
    
    def countertrend_strategy(self, p: str, depth: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        """反趋势交易策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 记录仓位
        self.position_history[p].append(pos)
        
        # 计算RSI
        self.calculate_rsi(p)
        
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
        counter_size = max(1, int(LIMIT[p] * PARAM["counter_size_frac"]))
        
        # 检查交易冷却期
        cooldown_active = timestamp - self.last_counter_trade.get(p, 0) < self.trade_cooldown
        
        # 反趋势交易逻辑
        if p in HIGH_VOL_PRODUCTS and not cooldown_active:  # 只对高波动性产品应用反趋势策略，且不在冷却期内
            # 检查是否有强烈趋势和极端RSI值
            if trend == 1 and self.rsi_values[p] > PARAM["rsi_overbought"]:
                # 确保新仓位安全
                if self.check_position_safety(p, pos, pos - counter_size):
                    # 强烈上升趋势，反向做空
                    sell_px = int(mid - 1)  # 降低卖出价格以确保成交
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders  # 只做反向交易，不做常规做市
                
            elif trend == -1 and self.rsi_values[p] < PARAM["rsi_oversold"]:
                # 确保新仓位安全
                if self.check_position_safety(p, pos, pos + counter_size):
                    # 强烈下降趋势，反向做多
                    buy_px = int(mid + 1)  # 提高买入价格以确保成交
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders  # 只做反向交易，不做常规做市
                
            # 检查是否有明确的反转信号
            elif reversal != 0:
                if reversal == 1 and self.check_position_safety(p, pos, pos + counter_size):  # 下降趋势反转为上升
                    # 积极买入
                    buy_px = int(mid + 1)
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders
                    
                elif reversal == -1 and self.check_position_safety(p, pos, pos - counter_size):  # 上升趋势反转为下降
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
        
        # 对每个产品应用反趋势策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.countertrend_strategy(
                    p, 
                    depth, 
                    state.position.get(p, 0),
                    state.timestamp
                )
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData 