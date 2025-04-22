from typing import Dict, List, Tuple, Optional
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np
import math

# 产品限制
POSITION_LIMITS = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75
}

# 产品特性配置
PRODUCT_CONFIG = {
    # 高波动性产品 - 更激进的RSI阈值
    "VOLCANIC_ROCK": {"rsi_period": 10, "rsi_ob": 75, "rsi_os": 25, "vol_scale": 1.4, "size": 0.3},
    "VOLCANIC_ROCK_VOUCHER_9500": {"rsi_period": 12, "rsi_ob": 75, "rsi_os": 25, "vol_scale": 1.3, "size": 0.25},
    "VOLCANIC_ROCK_VOUCHER_9750": {"rsi_period": 12, "rsi_ob": 75, "rsi_os": 25, "vol_scale": 1.3, "size": 0.25},
    "VOLCANIC_ROCK_VOUCHER_10000": {"rsi_period": 12, "rsi_ob": 75, "rsi_os": 25, "vol_scale": 1.3, "size": 0.25},
    "VOLCANIC_ROCK_VOUCHER_10250": {"rsi_period": 12, "rsi_ob": 75, "rsi_os": 25, "vol_scale": 1.3, "size": 0.25},
    "VOLCANIC_ROCK_VOUCHER_10500": {"rsi_period": 12, "rsi_ob": 75, "rsi_os": 25, "vol_scale": 1.3, "size": 0.25},
    
    # 低/中波动性产品 - 更保守的RSI阈值
    "PICNIC_BASKET1": {"rsi_period": 14, "rsi_ob": 72, "rsi_os": 28, "vol_scale": 1.2, "size": 0.3},
    "PICNIC_BASKET2": {"rsi_period": 14, "rsi_ob": 72, "rsi_os": 28, "vol_scale": 1.2, "size": 0.3},
    "MAGNIFICENT_MACARONS": {"rsi_period": 14, "rsi_ob": 70, "rsi_os": 30, "vol_scale": 1.0, "size": 0.25},
    
    # 默认配置 - 用于其他产品
    "DEFAULT": {"rsi_period": 14, "rsi_ob": 70, "rsi_os": 30, "vol_scale": 1.0, "size": 0.2}
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """获取最优买卖价和数量"""
    bid_p = ask_p = bid_q = ask_q = None
    if depth.buy_orders:
        bid_p = max(depth.buy_orders.keys())
        bid_q = depth.buy_orders[bid_p]
    if depth.sell_orders:
        ask_p = min(depth.sell_orders.keys())
        ask_q = depth.sell_orders[ask_p]
    return bid_p, bid_q, ask_p, ask_q

def mid_price(depth: OrderDepth) -> Optional[float]:
    """计算中间价"""
    bid, _, ask, _ = best_bid_ask(depth)
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return None

def get_product_config(product: str) -> dict:
    """获取产品配置或默认配置"""
    return PRODUCT_CONFIG.get(product, PRODUCT_CONFIG["DEFAULT"])

class Trader:
    def __init__(self):
        # 价格历史
        self.prices = defaultdict(list)
        # 波动率历史
        self.volatility = defaultdict(float)
        # RSI历史
        self.rsi_values = defaultdict(float)
        # 价格变动历史
        self.price_changes = defaultdict(list)
        # 平均真实波动率
        self.atr = defaultdict(float)
        # 动量指标
        self.momentum = defaultdict(float)
        # 交易历史
        self.trades = defaultdict(list)
        # 上一次交易的时间戳
        self.last_trade_timestamp = defaultdict(int)
        # 市场状态（1=上升趋势，-1=下降趋势，0=中性）
        self.market_state = defaultdict(int)
        # 动态阈值调整
        self.rsi_threshold_adj = defaultdict(float)
        # 总体市场波动率
        self.market_volatility = 0.0
        # 趋势强度
        self.trend_strength = defaultdict(float)
        
    def calculate_rsi(self, product: str, config: dict):
        """计算相对强弱指数 (RSI)"""
        price_changes = self.price_changes[product]
        
        if len(price_changes) < config["rsi_period"]:
            self.rsi_values[product] = 50  # 默认中性值
            return
        
        # 只使用最近的价格变动
        recent_changes = price_changes[-config["rsi_period"]:]
        
        # 计算上涨和下跌的平均值
        gains = [max(0, change) for change in recent_changes]
        losses = [max(0, -change) for change in recent_changes]
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # 计算相对强度
        if avg_loss == 0:
            rs = 100
        else:
            rs = avg_gain / avg_loss
            
        # 计算RSI
        self.rsi_values[product] = 100 - (100 / (1 + rs))
        
    def calculate_volatility(self, product: str, window: int = 20):
        """计算价格波动率"""
        prices = self.prices[product]
        if len(prices) < window:
            return 0.0
        
        # 使用最近的价格
        recent_prices = prices[-window:]
        
        # 计算波动率（标准差/均值）
        try:
            vol = statistics.stdev(recent_prices) / statistics.mean(recent_prices)
            return vol
        except:
            return 0.0
    
    def calculate_atr(self, product: str, depth: OrderDepth, period: int = 14):
        """计算平均真实波动率 (ATR)"""
        prices = self.prices[product]
        if len(prices) < 2:
            return
        
        # 当前价格
        current_price = mid_price(depth)
        if current_price is None:
            return
        
        # 上一个价格
        previous_price = prices[-1]
        
        # 真实波动率计算
        true_range = abs(current_price - previous_price)
        
        # 更新ATR（指数移动平均）
        alpha = 2 / (period + 1)
        if self.atr[product] == 0:
            self.atr[product] = true_range
        else:
            self.atr[product] = (1 - alpha) * self.atr[product] + alpha * true_range
    
    def calculate_momentum(self, product: str, period: int = 10):
        """计算价格动量"""
        prices = self.prices[product]
        if len(prices) <= period:
            return 0
        
        # 计算当前价格相对于n期前价格的百分比变化
        current = prices[-1]
        previous = prices[-period]
        
        momentum = (current - previous) / previous if previous != 0 else 0
        self.momentum[product] = momentum
        return momentum
    
    def detect_market_state(self, product: str, window: int = 20):
        """检测市场状态（趋势）"""
        prices = self.prices[product]
        if len(prices) < window:
            return 0  # 数据不足
        
        # 使用简单线性回归判断趋势
        y = np.array(prices[-window:])
        x = np.arange(len(y))
        
        # 计算趋势线斜率
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # 计算趋势强度 - 归一化斜率
        strength = abs(m * window / np.mean(y)) if np.mean(y) != 0 else 0
        self.trend_strength[product] = min(1.0, strength * 5)  # 限制在0-1范围内
        
        # 判断趋势方向
        if m > 0.0001:  # 正斜率 - 上升趋势
            return 1
        elif m < -0.0001:  # 负斜率 - 下降趋势
            return -1
        else:  # 无明显趋势
            return 0
        
    def adaptive_rsi_thresholds(self, product: str, base_ob: float, base_os: float):
        """根据市场状态调整RSI阈值"""
        trend = self.market_state[product]
        strength = self.trend_strength[product]
        vol = self.volatility[product]
        
        # 基于波动率和趋势强度的动态调整
        vol_factor = min(1.0, vol * 20)  # 波动率影响因子（最大1.0）
        
        # 在强趋势中，调整RSI阈值更难以触发反趋势信号
        if trend == 1:  # 上升趋势
            ob_adj = base_ob + 5 * strength  # 提高超买阈值
            os_adj = base_os + 5 * strength  # 提高超卖阈值（减少在上升趋势中的买入）
        elif trend == -1:  # 下降趋势
            ob_adj = base_ob - 5 * strength  # 降低超买阈值（减少在下降趋势中的卖出）
            os_adj = base_os - 5 * strength  # 降低超卖阈值
        else:
            ob_adj = base_ob
            os_adj = base_os
        
        # 高波动时调整阈值，使信号更极端（避免过早交易）
        ob_adj += 5 * vol_factor
        os_adj -= 5 * vol_factor
        
        # 确保阈值在合理范围内
        ob_adj = min(85, max(65, ob_adj))
        os_adj = min(35, max(15, os_adj))
        
        return ob_adj, os_adj
    
    def should_trade(self, product: str, timestamp: int, config: dict):
        """判断是否应该交易（冷却时间和其他条件）"""
        # 检查冷却时间
        cool_down = 50  # 冷却周期
        if timestamp - self.last_trade_timestamp[product] < cool_down:
            return False
        
        # 检查趋势强度 - 极端趋势时避免反向交易
        if self.trend_strength[product] > 0.8:
            return False
            
        # 检查整体市场波动性 - 市场极度波动时减少交易
        if self.market_volatility > 0.03:
            # 随机决定是否交易 - 高波动时只有30%机会交易
            return np.random.random() < 0.3
            
        return True
    
    def position_sizing(self, product: str, signal: int, depth: OrderDepth, config: dict):
        """智能仓位管理"""
        limit = POSITION_LIMITS[product]
        base_size = config["size"]
        
        # 基于信号强度的规模调整
        signal_strength = abs(self.rsi_values[product] - 50) / 50  # 0-1范围的信号强度
        
        # 基于波动率的风险调整
        vol_factor = max(0.5, min(1.5, 1.0 / (self.volatility[product] * 10 + 0.5)))
        
        # 基于趋势与信号一致性的调整
        trend = self.market_state[product]
        trend_agreement = 1.0
        if (signal > 0 and trend < 0) or (signal < 0 and trend > 0):
            # 信号与趋势相反时减小规模
            trend_agreement = 0.8
        
        # 综合计算规模
        size_pct = base_size * signal_strength * vol_factor * trend_agreement
        
        # 确保规模在合理范围内
        size_pct = min(0.5, max(0.1, size_pct))
        
        # 计算实际订单数量
        size = max(1, int(limit * size_pct))
        
        return size
    
    def adaptive_rsi_strategy(self, product: str, timestamp: int, depth: OrderDepth, position: int):
        """自适应RSI策略"""
        # 获取产品配置
        config = get_product_config(product)
        
        # 获取中间价
        price = mid_price(depth)
        if price is None:
            return []
        
        # 更新价格历史
        if self.prices[product]:
            # 计算价格变化
            prev_price = self.prices[product][-1]
            price_change = price - prev_price
            self.price_changes[product].append(price_change)
        
        self.prices[product].append(price)
        
        # 计算技术指标
        self.calculate_rsi(product, config)
        self.calculate_atr(product, depth)
        vol = self.calculate_volatility(product)
        self.volatility[product] = vol
        self.calculate_momentum(product)
        
        # 检测市场状态
        self.market_state[product] = self.detect_market_state(product)
        
        # 调整RSI阈值
        rsi_ob, rsi_os = self.adaptive_rsi_thresholds(
            product, config["rsi_ob"], config["rsi_os"]
        )
        
        # 获取当前RSI值
        rsi = self.rsi_values[product]
        
        # 检查是否应该交易
        if not self.should_trade(product, timestamp, config):
            return []
        
        # 获取当前价格
        bid, bid_q, ask, ask_q = best_bid_ask(depth)
        if bid is None or ask is None:
            return []
        
        # 计算动态价差
        spread = max(1, int(self.atr[product] * config["vol_scale"]))
        
        # 智能交易逻辑
        orders = []
        limit = POSITION_LIMITS[product]
        
        # 超买情况 - 做空信号
        if rsi > rsi_ob:
            # 计算仓位规模
            size = self.position_sizing(product, -1, depth, config)
            
            # 确保不超过仓位限制
            size = min(size, limit + position) if position < 0 else min(size, limit - position)
            
            if size > 0:
                # 更激进地卖出 - 使用买一价接近的价格
                sell_price = bid  # 使用买一价
                orders.append(Order(product, sell_price, -size))
                self.last_trade_timestamp[product] = timestamp
        
        # 超卖情况 - 做多信号
        elif rsi < rsi_os:
            # 计算仓位规模
            size = self.position_sizing(product, 1, depth, config)
            
            # 确保不超过仓位限制
            size = min(size, limit - position) if position > 0 else min(size, limit + position)
            
            if size > 0:
                # 更激进地买入 - 使用卖一价接近的价格
                buy_price = ask  # 使用卖一价
                orders.append(Order(product, buy_price, size))
                self.last_trade_timestamp[product] = timestamp
        
        # 中性区域 - 考虑做市
        else:
            # 在中性区域添加做市订单
            mm_size = max(1, int(limit * 0.1))  # 做市规模
            
            # 防止过度下单
            if abs(position) < limit * 0.7:  # 仓位小于限制的70%
                # 买单
                if position < limit * 0.3:  # 多头敞口较小时才做买单
                    orders.append(Order(product, bid - spread, mm_size))
                
                # 卖单
                if position > -limit * 0.3:  # 空头敞口较小时才做卖单
                    orders.append(Order(product, ask + spread, -mm_size))
        
        return orders
    
    def estimate_market_volatility(self, state: TradingState):
        """估计整体市场波动性"""
        vol_samples = []
        for product in POSITION_LIMITS:
            if product in self.volatility and self.volatility[product] > 0:
                vol_samples.append(self.volatility[product])
        
        if vol_samples:
            # 使用加权平均，给予较高波动性更大的权重
            weighted_vol = sum(vol**2 for vol in vol_samples) / len(vol_samples)
            # 平滑处理
            self.market_volatility = 0.8 * self.market_volatility + 0.2 * weighted_vol if self.market_volatility > 0 else weighted_vol
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result = {}
        
        # 估计整体市场波动性
        self.estimate_market_volatility(state)
        
        # 处理每个产品
        for product, depth in state.order_depths.items():
            # 跳过没有限制的产品
            if product not in POSITION_LIMITS:
                continue
                
            # 获取当前仓位
            position = state.position.get(product, 0)
            
            # 应用自适应RSI策略
            orders = self.adaptive_rsi_strategy(
                product, state.timestamp, depth, position
            )
            
            if orders:
                result[product] = orders
        
        return result, 0, state.traderData 