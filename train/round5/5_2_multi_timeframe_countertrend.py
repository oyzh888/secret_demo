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

# 高波动性产品
HIGH_VOL_PRODUCTS = {
    "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
    "PICNIC_BASKET1", "PICNIC_BASKET2"
}

# 中等波动性产品
MEDIUM_VOL_PRODUCTS = {
    "MAGNIFICENT_MACARONS", "VOLCANIC_ROCK_VOUCHER_10000", 
    "VOLCANIC_ROCK_VOUCHER_10250", "DJEMBES", "JAMS"
}

# 低波动性产品
LOW_VOL_PRODUCTS = {
    "CROISSANTS", "KELP", "VOLCANIC_ROCK_VOUCHER_10500", "RAINFOREST_RESIN", "SQUID_INK"
}

# 参数
PARAM = {
    # 多时间框架参数
    "timeframes": {
        "ultra_short": 5,    # 超短期窗口
        "short": 15,         # 短期窗口
        "medium": 30,        # 中期窗口
        "long": 60           # 长期窗口
    },
    "timeframe_weights": {   # 各时间框架权重
        "ultra_short": 0.1,
        "short": 0.3,
        "medium": 0.4,
        "long": 0.2
    },
    
    # 趋势和反转参数
    "trend_threshold": 0.65,  # 趋势检测阈值
    "reversal_threshold": 0.8, # 反转检测阈值
    "divergence_threshold": 0.5, # 背离检测阈值
    
    # 交易参数
    "position_limit_pct": 0.7, # 仓位限制百分比
    "mm_size_frac": 0.15,     # 做市规模
    "counter_size_frac": {    # 反趋势交易规模
        "strong": 0.3,        # 强信号
        "medium": 0.2,        # 中等信号
        "weak": 0.1           # 弱信号
    },
    "reversal_cooldown": 8,   # 反转交易冷却期
    
    # 价格参数
    "min_spread": 2,          # 最小价差
    "vol_scale": 1.2,         # 波动率缩放因子
    "price_aggression": {     # 价格激进程度
        "strong": 3,          # 强信号
        "medium": 2,          # 中等信号
        "weak": 1             # 弱信号
    },
    
    # 技术指标参数
    "rsi_period": 14,         # RSI周期
    "rsi_overbought": 70,     # RSI超买阈值
    "rsi_oversold": 30,       # RSI超卖阈值
    "macd_fast": 12,          # MACD快线
    "macd_slow": 26,          # MACD慢线
    "macd_signal": 9,         # MACD信号线
    
    # 风险控制参数
    "stop_loss_pct": 0.03,    # 止损百分比
    "take_profit_pct": 0.05,  # 止盈百分比
    "max_drawdown": 200,      # 最大回撤
    "max_trades_per_day": 5   # 每天最大交易次数
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
        self.rsi_values = defaultdict(dict)  # 各时间框架的RSI值
        self.macd_values = defaultdict(dict)  # 各时间框架的MACD值
        self.trends = defaultdict(dict)  # 各时间框架的趋势
        self.last_counter_trade = defaultdict(int)  # 上次反向交易的时间戳
        self.trade_count = defaultdict(int)  # 交易计数
        self.entry_prices = defaultdict(float)  # 入场价格
        self.max_position_values = defaultdict(float)  # 最大仓位价值
        self.signal_strength = defaultdict(str)  # 信号强度：strong, medium, weak
        self.divergence_detected = defaultdict(bool)  # 是否检测到背离
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) * PARAM["vol_scale"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def calculate_rsi(self, p: str, timeframe: str):
        """计算指定时间框架的RSI"""
        period = PARAM["timeframes"][timeframe]
        if len(self.prices[p]) < period + 1:
            self.rsi_values[p][timeframe] = 50  # 默认中性值
            return
            
        # 使用指定时间框架的价格
        prices = self.prices[p][-period-1:]
        
        # 计算价格变动
        price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
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
        self.rsi_values[p][timeframe] = 100 - (100 / (1 + rs))
    
    def calculate_macd(self, p: str, timeframe: str):
        """计算指定时间框架的MACD"""
        if len(self.prices[p]) < PARAM["macd_slow"]:
            self.macd_values[p][timeframe] = {"macd": 0, "signal": 0, "histogram": 0}
            return
            
        # 使用指定时间框架的价格
        prices = self.prices[p][-PARAM["timeframes"][timeframe]:]
        
        # 计算EMA
        ema_fast = self.calculate_ema(prices, PARAM["macd_fast"])
        ema_slow = self.calculate_ema(prices, PARAM["macd_slow"])
        
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线
        macd_values = [self.macd_values[p].get(timeframe, {}).get("macd", 0)] * (PARAM["macd_signal"] - 1)
        macd_values.append(macd_line)
        signal_line = self.calculate_ema(macd_values, PARAM["macd_signal"])
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        self.macd_values[p][timeframe] = {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    def calculate_ema(self, values: List[float], period: int) -> float:
        """计算指数移动平均线(EMA)"""
        if len(values) < period:
            return sum(values) / len(values)
            
        # 使用最近的period个值
        recent_values = values[-period:]
        
        # 计算EMA
        multiplier = 2 / (period + 1)
        ema = recent_values[0]
        
        for value in recent_values[1:]:
            ema = (value * multiplier) + (ema * (1 - multiplier))
            
        return ema
    
    def detect_trend(self, p: str, timeframe: str) -> int:
        """检测指定时间框架的市场趋势"""
        period = PARAM["timeframes"][timeframe]
        prices = self.prices[p]
        if len(prices) < period:
            return 0  # 数据不足
            
        recent_prices = prices[-period:]  
        up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
        
        up_ratio = up_moves / (len(recent_prices) - 1)
        down_ratio = down_moves / (len(recent_prices) - 1)
        
        if up_ratio > PARAM["trend_threshold"]:
            return 1  # 上升趋势
        elif down_ratio > PARAM["trend_threshold"]:
            return -1  # 下降趋势
        return 0  # 中性
    
    def update_all_timeframes(self, p: str):
        """更新所有时间框架的指标"""
        for timeframe in PARAM["timeframes"]:
            # 更新RSI
            self.calculate_rsi(p, timeframe)
            
            # 更新MACD
            self.calculate_macd(p, timeframe)
            
            # 更新趋势
            self.trends[p][timeframe] = self.detect_trend(p, timeframe)
    
    def detect_divergence(self, p: str) -> bool:
        """检测价格与指标之间的背离"""
        # 检查价格与RSI的背离
        if len(self.prices[p]) < PARAM["timeframes"]["medium"]:
            return False
            
        # 获取中期时间框架的价格和RSI
        prices = self.prices[p][-PARAM["timeframes"]["medium"]:]
        rsi = self.rsi_values[p].get("medium", 50)
        
        # 检查价格新高但RSI未创新高（顶背离）
        if len(prices) >= 3 and prices[-1] > max(prices[:-1]) and rsi < PARAM["rsi_overbought"]:
            return True
            
        # 检查价格新低但RSI未创新低（底背离）
        if len(prices) >= 3 and prices[-1] < min(prices[:-1]) and rsi > PARAM["rsi_oversold"]:
            return True
            
        # 检查MACD背离
        macd = self.macd_values[p].get("medium", {}).get("histogram", 0)
        prev_macd = self.macd_values[p].get("medium", {}).get("histogram", 0)
        
        # 价格上涨但MACD下降
        if len(prices) >= 2 and prices[-1] > prices[-2] and macd < prev_macd:
            return True
            
        # 价格下跌但MACD上升
        if len(prices) >= 2 and prices[-1] < prices[-2] and macd > prev_macd:
            return True
            
        return False
    
    def calculate_weighted_trend(self, p: str) -> float:
        """计算加权趋势"""
        weighted_trend = 0
        
        for timeframe, weight in PARAM["timeframe_weights"].items():
            trend = self.trends[p].get(timeframe, 0)
            weighted_trend += trend * weight
            
        return weighted_trend
    
    def determine_signal_strength(self, p: str, weighted_trend: float) -> str:
        """确定信号强度"""
        # 检查是否有背离
        divergence = self.detect_divergence(p)
        self.divergence_detected[p] = divergence
        
        # 检查RSI超买超卖
        rsi_medium = self.rsi_values[p].get("medium", 50)
        rsi_signal = 0
        if rsi_medium > PARAM["rsi_overbought"]:
            rsi_signal = -1  # 超买，看跌信号
        elif rsi_medium < PARAM["rsi_oversold"]:
            rsi_signal = 1   # 超卖，看涨信号
            
        # 检查MACD信号
        macd_medium = self.macd_values[p].get("medium", {})
        macd_signal = 0
        if macd_medium.get("macd", 0) > macd_medium.get("signal", 0):
            macd_signal = 1  # MACD在信号线上方，看涨信号
        elif macd_medium.get("macd", 0) < macd_medium.get("signal", 0):
            macd_signal = -1 # MACD在信号线下方，看跌信号
            
        # 综合信号强度
        signal_count = sum(1 for s in [weighted_trend, rsi_signal, macd_signal] if s != 0)
        
        # 如果有背离，增强信号
        if divergence:
            if signal_count >= 2:
                return "strong"
            elif signal_count == 1:
                return "medium"
                
        # 根据信号数量确定强度
        if signal_count >= 3:
            return "strong"
        elif signal_count == 2:
            return "medium"
        elif signal_count == 1:
            return "weak"
            
        return "weak"
    
    def check_risk_controls(self, p: str, pos: int, current_price: float) -> bool:
        """检查风险控制，返回是否允许交易"""
        # 检查交易次数限制
        if self.trade_count[p] >= PARAM["max_trades_per_day"]:
            return False  # 达到每日交易次数限制
            
        # 如果没有入场价格，设置当前价格为入场价格
        if self.entry_prices[p] == 0 and pos != 0:
            self.entry_prices[p] = current_price
            self.max_position_values[p] = abs(pos * current_price)
            return True
        
        # 如果有持仓，检查止损和止盈
        if pos != 0 and self.entry_prices[p] != 0:
            # 计算当前仓位价值
            current_value = abs(pos * current_price)
            
            # 更新最大仓位价值
            if current_value > self.max_position_values[p]:
                self.max_position_values[p] = current_value
            
            # 计算价格变动百分比
            price_change_pct = (current_price - self.entry_prices[p]) / self.entry_prices[p]
            
            # 检查止损
            if (pos > 0 and price_change_pct < -PARAM["stop_loss_pct"]) or \
               (pos < 0 and price_change_pct > PARAM["stop_loss_pct"]):
                return False  # 触发止损，不允许继续同方向交易
            
            # 检查止盈
            if (pos > 0 and price_change_pct > PARAM["take_profit_pct"]) or \
               (pos < 0 and price_change_pct < -PARAM["take_profit_pct"]):
                return False  # 触发止盈，不允许继续同方向交易
            
            # 检查最大回撤
            drawdown = self.max_position_values[p] - current_value
            if drawdown > PARAM["max_drawdown"]:
                return False  # 触发最大回撤，不允许继续同方向交易
        
        return True  # 通过风险控制，允许交易
    
    def multi_timeframe_countertrend_strategy(self, p: str, depth: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        """多时间框架反趋势交易策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 记录仓位
        self.position_history[p].append(pos)
        
        # 更新所有时间框架的指标
        self.update_all_timeframes(p)
        
        # 计算加权趋势
        weighted_trend = self.calculate_weighted_trend(p)
        
        # 确定信号强度
        strength = self.determine_signal_strength(p, weighted_trend)
        self.signal_strength[p] = strength
        
        # 检查风险控制
        risk_ok = self.check_risk_controls(p, pos, mid)
        
        # 计算价差
        vol = self._vol(p)
        spread = max(PARAM["min_spread"], int(vol))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模
        max_position = int(LIMIT[p] * PARAM["position_limit_pct"])
        mm_size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"]))
        counter_size = max(1, int(LIMIT[p] * PARAM["counter_size_frac"][strength]))
        
        # 反趋势交易逻辑
        if p in HIGH_VOL_PRODUCTS or p in MEDIUM_VOL_PRODUCTS:  # 只对高波动性和中等波动性产品应用反趋势策略
            # 检查是否在冷却期
            cooldown_active = timestamp - self.last_counter_trade.get(p, 0) < PARAM["reversal_cooldown"]
            
            if not cooldown_active and risk_ok:
                # 获取中期RSI
                rsi_medium = self.rsi_values[p].get("medium", 50)
                
                # 获取中期MACD
                macd_medium = self.macd_values[p].get("medium", {})
                macd_hist = macd_medium.get("histogram", 0)
                
                # 反趋势交易条件
                counter_signal = 0
                
                # 条件1: 加权趋势明显 + RSI超买/超卖
                if abs(weighted_trend) > 0.5:
                    if weighted_trend > 0 and rsi_medium > PARAM["rsi_overbought"]:
                        counter_signal = -1  # 做空信号
                    elif weighted_trend < 0 and rsi_medium < PARAM["rsi_oversold"]:
                        counter_signal = 1   # 做多信号
                
                # 条件2: 检测到背离
                if self.divergence_detected[p]:
                    if weighted_trend > 0:
                        counter_signal = -1  # 做空信号
                    elif weighted_trend < 0:
                        counter_signal = 1   # 做多信号
                
                # 条件3: 超短期反转 + 中期趋势
                ultra_short_trend = self.trends[p].get("ultra_short", 0)
                medium_trend = self.trends[p].get("medium", 0)
                
                if ultra_short_trend != 0 and medium_trend != 0 and ultra_short_trend != medium_trend:
                    counter_signal = ultra_short_trend  # 使用超短期趋势方向
                
                # 执行反趋势交易
                if counter_signal != 0:
                    price_aggression = PARAM["price_aggression"][strength]
                    
                    if counter_signal > 0:  # 做多信号
                        # 积极买入
                        buy_px = int(mid + price_aggression)
                        orders.append(Order(p, buy_px, counter_size))
                        self.last_counter_trade[p] = timestamp
                        self.entry_prices[p] = buy_px
                        self.trade_count[p] += 1
                        return orders
                        
                    elif counter_signal < 0:  # 做空信号
                        # 积极卖出
                        sell_px = int(mid - price_aggression)
                        orders.append(Order(p, sell_px, -counter_size))
                        self.last_counter_trade[p] = timestamp
                        self.entry_prices[p] = sell_px
                        self.trade_count[p] += 1
                        return orders
        
        # 如果没有反趋势交易信号，执行常规做市
        # 根据加权趋势调整价格
        if weighted_trend > 0.3:  # 上升趋势
            buy_px += 1
            sell_px += 1
        elif weighted_trend < -0.3:  # 下降趋势
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
        
        # 对每个产品应用多时间框架反趋势策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.multi_timeframe_countertrend_strategy(
                    p, 
                    depth, 
                    state.position.get(p, 0),
                    state.timestamp
                )
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
