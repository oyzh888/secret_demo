from typing import Dict, List, Tuple, Optional
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np
import math

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

# 所有产品默认都可以进行反趋势交易，根据波动率动态调整
# 不再使用静态的高波动性产品列表

# 参数
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
    "rsi_period": 14,         # RSI周期
    "rsi_overbought": 70,     # RSI超买阈值
    "rsi_oversold": 30,       # RSI超卖阈值
    
    # 波动率适应性参数
    "vol_window": 20,         # 波动率计算窗口
    "vol_history_window": 100, # 历史波动率窗口大小
    "vol_threshold_multiplier": 1.5, # 波动率阈值倍数
    "high_vol_multiplier": 1.5, # 高波动环境订单大小倍数
    "low_vol_multiplier": 0.7,  # 低波动环境订单大小倍数
    "vol_trend_window": 5,      # 波动率趋势检测窗口
    "vol_adjust_alpha": 0.25,   # 波动率调整平滑系数
    "price_jump_threshold": 0.02 # 价格跳跃阈值(百分比)
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
        
        # 波动率相关
        self.volatility = defaultdict(list)  # 波动率历史
        self.volatility_avg = defaultdict(float)  # 平均波动率
        self.volatility_std = defaultdict(float)  # 波动率标准差
        self.volatility_trend = defaultdict(float)  # 波动率趋势
        self.is_high_volatility = defaultdict(bool)  # 是否高波动环境
        self.last_price_jump = defaultdict(int)  # 上次价格跳跃的时间戳
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def update_volatility(self, p: str):
        """更新波动率指标"""
        prices = self.prices[p]
        if len(prices) < PARAM["vol_window"] + 1:
            return
            
        # 计算最近窗口的价格变动率
        returns = [(prices[i] / prices[i-1] - 1) for i in range(-1, -PARAM["vol_window"]-1, -1)]
        
        # 计算波动率 (标准差)
        current_vol = statistics.stdev(returns) * math.sqrt(252)  # 年化
        
        # 更新波动率历史
        self.volatility[p].append(current_vol)
        if len(self.volatility[p]) > PARAM["vol_history_window"]:
            self.volatility[p].pop(0)
            
        # 更新波动率统计
        if len(self.volatility[p]) > 5:
            self.volatility_avg[p] = sum(self.volatility[p]) / len(self.volatility[p])
            self.volatility_std[p] = statistics.stdev(self.volatility[p]) if len(self.volatility[p]) > 1 else 0
            
            # 检测波动率趋势
            if len(self.volatility[p]) >= PARAM["vol_trend_window"]:
                recent_vols = self.volatility[p][-PARAM["vol_trend_window"]:]
                vol_trend = (recent_vols[-1] / recent_vols[0]) - 1
                # 使用平滑因子更新趋势
                self.volatility_trend[p] = PARAM["vol_adjust_alpha"] * vol_trend + (1 - PARAM["vol_adjust_alpha"]) * self.volatility_trend.get(p, 0)
                
            # 确定是否是高波动环境
            threshold = self.volatility_avg[p] + PARAM["vol_threshold_multiplier"] * self.volatility_std[p]
            self.is_high_volatility[p] = current_vol > threshold
            
        return current_vol
            
    def detect_price_jump(self, p: str, timestamp: int) -> bool:
        """检测价格跳跃"""
        prices = self.prices[p]
        if len(prices) < 2:
            return False
            
        # 计算最近的价格变动率
        price_change = abs(prices[-1] / prices[-2] - 1)
        
        # 如果价格变动超过阈值，记录为跳跃
        if price_change > PARAM["price_jump_threshold"]:
            self.last_price_jump[p] = timestamp
            return True
            
        return False
    
    def get_volatility_adjusted_size(self, p: str, base_size: int) -> int:
        """根据波动率调整订单大小"""
        # 默认不调整
        if p not in self.volatility_avg or p not in self.volatility_trend:
            return base_size
            
        size_multiplier = 1.0
        
        # 根据波动率环境调整
        if self.is_high_volatility[p]:
            # 高波动环境下
            if self.volatility_trend[p] > 0:
                # 波动率上升，更加谨慎
                size_multiplier = PARAM["high_vol_multiplier"] * 0.8
            else:
                # 波动率下降，增加头寸
                size_multiplier = PARAM["high_vol_multiplier"]
        else:
            # 低波动环境下
            if self.volatility_trend[p] < 0:
                # 波动率下降，更加保守
                size_multiplier = PARAM["low_vol_multiplier"] * 0.8
            else:
                # 波动率上升，增加头寸
                size_multiplier = PARAM["low_vol_multiplier"]
                
        # 计算调整后的订单大小
        adjusted_size = int(base_size * size_multiplier)
        
        # 确保至少为1
        return max(1, adjusted_size)
    
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
    
    def get_volatility_adjusted_params(self, p: str) -> dict:
        """获取根据波动率调整的参数"""
        adjusted_params = PARAM.copy()
        
        # 如果没有足够波动率历史，使用默认参数
        if p not in self.volatility_avg:
            return adjusted_params
            
        # 根据波动率环境调整参数
        if self.is_high_volatility[p]:
            # 高波动环境下：更快的反应，更严格的信号过滤
            adjusted_params["trend_window"] = max(15, int(PARAM["trend_window"] * 0.7))
            adjusted_params["trend_threshold"] = min(0.8, PARAM["trend_threshold"] * 1.1)
            adjusted_params["reversal_window"] = max(3, int(PARAM["reversal_window"] * 0.8))
            adjusted_params["reversal_threshold"] = min(0.85, PARAM["reversal_threshold"] * 1.05)
            adjusted_params["position_limit_pct"] = max(0.4, PARAM["position_limit_pct"] * 0.9)
            adjusted_params["rsi_period"] = max(7, int(PARAM["rsi_period"] * 0.8))
            adjusted_params["rsi_overbought"] = min(75, PARAM["rsi_overbought"] + 2)
            adjusted_params["rsi_oversold"] = max(25, PARAM["rsi_oversold"] - 2)
        else:
            # 低波动环境下：更慢的反应，更宽松的信号过滤
            adjusted_params["trend_window"] = min(40, int(PARAM["trend_window"] * 1.2))
            adjusted_params["trend_threshold"] = max(0.65, PARAM["trend_threshold"] * 0.95)
            adjusted_params["reversal_window"] = min(8, int(PARAM["reversal_window"] * 1.2))
            adjusted_params["reversal_threshold"] = max(0.75, PARAM["reversal_threshold"] * 0.95)
            adjusted_params["position_limit_pct"] = min(0.75, PARAM["position_limit_pct"] * 1.1)
            adjusted_params["rsi_period"] = min(21, int(PARAM["rsi_period"] * 1.2))
            adjusted_params["rsi_overbought"] = max(65, PARAM["rsi_overbought"] - 2)
            adjusted_params["rsi_oversold"] = min(35, PARAM["rsi_oversold"] + 2)
            
        # 波动率趋势影响
        if self.volatility_trend[p] > 0.1:
            # 波动率快速上升，更加谨慎
            adjusted_params["position_limit_pct"] = max(0.4, adjusted_params["position_limit_pct"] * 0.9)
        elif self.volatility_trend[p] < -0.1:
            # 波动率快速下降，更加激进
            adjusted_params["position_limit_pct"] = min(0.75, adjusted_params["position_limit_pct"] * 1.1)
            
        return adjusted_params
    
    def volatility_adjusted_countertrend_strategy(self, p: str, depth: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        """波动率适应性反趋势交易策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 记录仓位
        self.position_history[p].append(pos)
        
        # 更新波动率
        self.update_volatility(p)
        
        # 检测价格跳跃
        price_jump = self.detect_price_jump(p, timestamp)
        
        # 获取根据波动率调整的参数
        params = self.get_volatility_adjusted_params(p)
        
        # 计算RSI
        self.calculate_rsi(p)
        
        # 检测趋势和反转
        trend = self.detect_trend(p)
        reversal = self.detect_reversal(p)
        
        # 波动率调整后的价差
        vol_value = self.volatility_avg.get(p, 0.01)
        spread = max(params["min_spread"], int(vol_value * 100))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模
        max_position = int(LIMIT[p] * params["position_limit_pct"])
        mm_size_base = max(1, int(LIMIT[p] * params["mm_size_frac"]))
        counter_size_base = max(1, int(LIMIT[p] * params["counter_size_frac"]))
        
        # 根据波动率调整交易规模
        mm_size = self.get_volatility_adjusted_size(p, mm_size_base)
        counter_size = self.get_volatility_adjusted_size(p, counter_size_base)
        
        # 检查是否应该使用反趋势策略
        # 根据波动率动态决定，而不是静态列表
        should_use_countertrend = self.is_high_volatility.get(p, False) or price_jump
        
        # 反趋势交易逻辑
        if should_use_countertrend:
            # 检查是否有强烈趋势
            if trend == 1 and self.rsi_values[p] > params["rsi_overbought"]:
                # 强烈上升趋势，反向做空
                # 更积极地卖出
                sell_px = int(mid - 1)  # 降低卖出价格以确保成交
                orders.append(Order(p, sell_px, -counter_size))
                self.last_counter_trade[p] = timestamp
                return orders  # 只做反向交易，不做常规做市
                
            elif trend == -1 and self.rsi_values[p] < params["rsi_oversold"]:
                # 强烈下降趋势，反向做多
                # 更积极地买入
                buy_px = int(mid + 1)  # 提高买入价格以确保成交
                orders.append(Order(p, buy_px, counter_size))
                self.last_counter_trade[p] = timestamp
                return orders  # 只做反向交易，不做常规做市
                
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
        
        # 对每个产品应用波动率适应性反趋势策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.volatility_adjusted_countertrend_strategy(
                    p, 
                    depth, 
                    state.position.get(p, 0),
                    state.timestamp
                )
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData 