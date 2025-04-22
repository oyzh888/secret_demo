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

# 产品分类
PRODUCT_CATEGORIES = {
    # 高波动性产品
    "high_vol": {
        "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
        "PICNIC_BASKET1", "PICNIC_BASKET2"
    },
    # 中等波动性产品
    "medium_vol": {
        "MAGNIFICENT_MACARONS", "VOLCANIC_ROCK_VOUCHER_10000", 
        "VOLCANIC_ROCK_VOUCHER_10250", "DJEMBES", "JAMS"
    },
    # 低波动性产品
    "low_vol": {
        "CROISSANTS", "KELP", "VOLCANIC_ROCK_VOUCHER_10500", "RAINFOREST_RESIN", "SQUID_INK"
    }
}

# 参数
PARAM = {
    # 波动率参数
    "vol_window_short": 10,     # 短期波动率窗口
    "vol_window_medium": 30,    # 中期波动率窗口
    "vol_window_long": 60,      # 长期波动率窗口
    "vol_ratio_threshold": 1.5, # 波动率比率阈值
    "vol_percentile_high": 80,  # 高波动率百分位
    "vol_percentile_low": 20,   # 低波动率百分位
    
    # 趋势和反转参数
    "trend_window": 30,         # 趋势检测窗口
    "trend_threshold": 0.7,     # 趋势检测阈值
    "reversal_window": 5,       # 反转检测窗口
    "reversal_threshold": 0.8,  # 反转检测阈值
    
    # 交易参数
    "position_limit_pct": {     # 不同波动率环境的仓位限制
        "high_vol": 0.4,        # 高波动率环境
        "normal_vol": 0.6,      # 正常波动率环境
        "low_vol": 0.8          # 低波动率环境
    },
    "mm_size_frac": {           # 不同波动率环境的做市规模
        "high_vol": 0.1,
        "normal_vol": 0.15,
        "low_vol": 0.2
    },
    "counter_size_frac": {      # 不同波动率环境的反趋势交易规模
        "high_vol": 0.2,
        "normal_vol": 0.25,
        "low_vol": 0.3
    },
    "reversal_cooldown": {      # 不同波动率环境的反转交易冷却期
        "high_vol": 5,
        "normal_vol": 8,
        "low_vol": 12
    },
    
    # 价格参数
    "min_spread": {             # 不同波动率环境的最小价差
        "high_vol": 3,
        "normal_vol": 2,
        "low_vol": 1
    },
    "vol_scale": 1.2,           # 波动率缩放因子
    "price_aggression": {       # 不同波动率环境的价格激进程度
        "high_vol": 1,
        "normal_vol": 2,
        "low_vol": 3
    },
    
    # 技术指标参数
    "rsi_period": 14,           # RSI周期
    "rsi_overbought": {         # 不同波动率环境的RSI超买阈值
        "high_vol": 75,
        "normal_vol": 70,
        "low_vol": 65
    },
    "rsi_oversold": {           # 不同波动率环境的RSI超卖阈值
        "high_vol": 25,
        "normal_vol": 30,
        "low_vol": 35
    },
    "bollinger_period": 20,     # 布林带周期
    "bollinger_std": {          # 不同波动率环境的布林带标准差
        "high_vol": 2.5,
        "normal_vol": 2.0,
        "low_vol": 1.5
    },
    
    # 风险控制参数
    "stop_loss_pct": {          # 不同波动率环境的止损百分比
        "high_vol": 0.02,
        "normal_vol": 0.03,
        "low_vol": 0.04
    },
    "max_drawdown": 200,        # 最大回撤
    "max_trades_per_day": {     # 不同波动率环境的每日最大交易次数
        "high_vol": 3,
        "normal_vol": 5,
        "low_vol": 7
    },
    
    # 波动率交易参数
    "vol_expansion_threshold": 1.8,  # 波动率扩张阈值
    "vol_contraction_threshold": 0.6, # 波动率收缩阈值
    "vol_regime_update_freq": 5,     # 波动率环境更新频率
    "vol_trade_size_pct": 0.3        # 波动率交易规模百分比
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.returns = defaultdict(list)  # 价格回报率
        self.volatilities = defaultdict(dict)  # 不同时间框架的波动率
        self.vol_percentile = defaultdict(float)  # 波动率百分位
        self.position_history = defaultdict(list)  # 仓位历史
        self.rsi_values = defaultdict(float)  # RSI值
        self.bollinger_bands = defaultdict(dict)  # 布林带
        self.last_counter_trade = defaultdict(int)  # 上次反向交易的时间戳
        self.trade_count = defaultdict(int)  # 交易计数
        self.entry_prices = defaultdict(float)  # 入场价格
        self.max_position_values = defaultdict(float)  # 最大仓位价值
        self.vol_regime = defaultdict(str)  # 波动率环境：high_vol, normal_vol, low_vol
        self.update_counter = 0  # 更新计数器
        self.vol_expansion_detected = defaultdict(bool)  # 波动率扩张检测
        self.vol_contraction_detected = defaultdict(bool)  # 波动率收缩检测
        
    def _vol(self, p: str, window: int = None) -> float:
        """计算产品波动率"""
        if window is None:
            window = PARAM["vol_window_medium"]
            
        h = self.prices[p]
        if len(h) < window:
            return 2  # 默认中等波动率
        return statistics.stdev(h[-window:]) * PARAM["vol_scale"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def get_product_category(self, product: str) -> str:
        """获取产品类别"""
        for category, products in PRODUCT_CATEGORIES.items():
            if product in products:
                return category
        return "medium_vol"  # 默认为中等波动性
    
    def update_volatility_metrics(self):
        """更新波动率相关指标"""
        # 增加计数器
        self.update_counter += 1
        
        # 只在特定频率更新
        if self.update_counter % PARAM["vol_regime_update_freq"] != 0:
            return
            
        # 计算每个产品的波动率
        for p, prices in self.prices.items():
            if len(prices) < PARAM["vol_window_long"]:
                continue
                
            # 计算不同时间框架的波动率
            short_vol = self._vol(p, PARAM["vol_window_short"])
            medium_vol = self._vol(p, PARAM["vol_window_medium"])
            long_vol = self._vol(p, PARAM["vol_window_long"])
            
            # 存储波动率
            self.volatilities[p] = {
                "short": short_vol,
                "medium": medium_vol,
                "long": long_vol
            }
            
            # 计算波动率比率
            short_medium_ratio = short_vol / medium_vol if medium_vol > 0 else 1
            medium_long_ratio = medium_vol / long_vol if long_vol > 0 else 1
            
            # 检测波动率扩张和收缩
            self.vol_expansion_detected[p] = short_medium_ratio > PARAM["vol_expansion_threshold"]
            self.vol_contraction_detected[p] = short_medium_ratio < PARAM["vol_contraction_threshold"]
            
            # 确定波动率环境
            if short_medium_ratio > PARAM["vol_ratio_threshold"] or medium_long_ratio > PARAM["vol_ratio_threshold"]:
                self.vol_regime[p] = "high_vol"
            elif short_medium_ratio < 1 / PARAM["vol_ratio_threshold"] or medium_long_ratio < 1 / PARAM["vol_ratio_threshold"]:
                self.vol_regime[p] = "low_vol"
            else:
                self.vol_regime[p] = "normal_vol"
    
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
    
    def calculate_bollinger_bands(self, p: str):
        """计算布林带"""
        if len(self.prices[p]) < PARAM["bollinger_period"]:
            self.bollinger_bands[p] = {"upper": None, "middle": None, "lower": None}
            return
            
        # 获取波动率环境
        vol_env = self.vol_regime.get(p, "normal_vol")
        
        # 获取布林带标准差
        std_dev = PARAM["bollinger_std"][vol_env]
        
        # 计算移动平均线
        recent_prices = self.prices[p][-PARAM["bollinger_period"]:]
        sma = sum(recent_prices) / len(recent_prices)
        
        # 计算标准差
        std = statistics.stdev(recent_prices)
        
        # 计算上下轨
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        self.bollinger_bands[p] = {
            "upper": upper,
            "middle": sma,
            "lower": lower
        }
    
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
    
    def check_risk_controls(self, p: str, pos: int, current_price: float) -> bool:
        """检查风险控制，返回是否允许交易"""
        # 获取波动率环境
        vol_env = self.vol_regime.get(p, "normal_vol")
        
        # 检查交易次数限制
        if self.trade_count[p] >= PARAM["max_trades_per_day"][vol_env]:
            return False  # 达到每日交易次数限制
            
        # 如果没有入场价格，设置当前价格为入场价格
        if self.entry_prices[p] == 0 and pos != 0:
            self.entry_prices[p] = current_price
            self.max_position_values[p] = abs(pos * current_price)
            return True
        
        # 如果有持仓，检查止损
        if pos != 0 and self.entry_prices[p] != 0:
            # 计算当前仓位价值
            current_value = abs(pos * current_price)
            
            # 更新最大仓位价值
            if current_value > self.max_position_values[p]:
                self.max_position_values[p] = current_value
            
            # 计算价格变动百分比
            price_change_pct = (current_price - self.entry_prices[p]) / self.entry_prices[p]
            
            # 检查止损
            stop_loss = PARAM["stop_loss_pct"][vol_env]
            if (pos > 0 and price_change_pct < -stop_loss) or \
               (pos < 0 and price_change_pct > stop_loss):
                return False  # 触发止损，不允许继续同方向交易
            
            # 检查最大回撤
            drawdown = self.max_position_values[p] - current_value
            if drawdown > PARAM["max_drawdown"]:
                return False  # 触发最大回撤，不允许继续同方向交易
        
        return True  # 通过风险控制，允许交易
    
    def volatility_trading_strategy(self, p: str, depth: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        """波动率交易策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 记录仓位
        self.position_history[p].append(pos)
        
        # 更新技术指标
        self.calculate_rsi(p)
        self.calculate_bollinger_bands(p)
        
        # 检测趋势和反转
        trend = self.detect_trend(p)
        reversal = self.detect_reversal(p)
        
        # 获取波动率环境
        vol_env = self.vol_regime.get(p, "normal_vol")
        
        # 检查风险控制
        risk_ok = self.check_risk_controls(p, pos, mid)
        
        # 计算价差
        vol = self._vol(p)
        min_spread = PARAM["min_spread"][vol_env]
        spread = max(min_spread, int(vol))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模
        max_position = int(LIMIT[p] * PARAM["position_limit_pct"][vol_env])
        mm_size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"][vol_env]))
        counter_size = max(1, int(LIMIT[p] * PARAM["counter_size_frac"][vol_env]))
        vol_trade_size = max(1, int(LIMIT[p] * PARAM["vol_trade_size_pct"]))
        
        # 获取产品类别
        category = self.get_product_category(p)
        
        # 波动率交易逻辑
        if category in ["high_vol", "medium_vol"]:  # 只对高波动性和中等波动性产品应用波动率策略
            # 检查是否在冷却期
            cooldown_active = timestamp - self.last_counter_trade.get(p, 0) < PARAM["reversal_cooldown"][vol_env]
            
            if not cooldown_active and risk_ok:
                # 获取RSI超买超卖阈值
                rsi_overbought = PARAM["rsi_overbought"][vol_env]
                rsi_oversold = PARAM["rsi_oversold"][vol_env]
                
                # 获取价格激进程度
                price_aggression = PARAM["price_aggression"][vol_env]
                
                # 波动率扩张策略
                if self.vol_expansion_detected[p]:
                    # 波动率扩张时，使用更宽的价差，更小的规模
                    spread = int(spread * 1.5)
                    buy_px = int(mid - spread)
                    sell_px = int(mid + spread)
                    
                    # 如果有明确趋势，可以顺势交易
                    if trend != 0:
                        if trend > 0:  # 上升趋势
                            # 更积极地买入
                            buy_px = int(mid + price_aggression)
                            orders.append(Order(p, buy_px, vol_trade_size))
                            self.last_counter_trade[p] = timestamp
                            self.entry_prices[p] = buy_px
                            self.trade_count[p] += 1
                            return orders
                        else:  # 下降趋势
                            # 更积极地卖出
                            sell_px = int(mid - price_aggression)
                            orders.append(Order(p, sell_px, -vol_trade_size))
                            self.last_counter_trade[p] = timestamp
                            self.entry_prices[p] = sell_px
                            self.trade_count[p] += 1
                            return orders
                
                # 波动率收缩策略
                elif self.vol_contraction_detected[p]:
                    # 波动率收缩时，使用更窄的价差，更大的规模
                    spread = max(1, int(spread * 0.7))
                    buy_px = int(mid - spread)
                    sell_px = int(mid + spread)
                    
                    # 如果有明确反转信号，可以反向交易
                    if reversal != 0:
                        if reversal > 0:  # 下降趋势反转为上升
                            # 积极买入
                            buy_px = int(mid + price_aggression)
                            orders.append(Order(p, buy_px, vol_trade_size))
                            self.last_counter_trade[p] = timestamp
                            self.entry_prices[p] = buy_px
                            self.trade_count[p] += 1
                            return orders
                        else:  # 上升趋势反转为下降
                            # 积极卖出
                            sell_px = int(mid - price_aggression)
                            orders.append(Order(p, sell_px, -vol_trade_size))
                            self.last_counter_trade[p] = timestamp
                            self.entry_prices[p] = sell_px
                            self.trade_count[p] += 1
                            return orders
                
                # 布林带策略
                bb = self.bollinger_bands[p]
                if bb["upper"] is not None and bb["lower"] is not None:
                    if mid > bb["upper"] and self.rsi_values[p] > rsi_overbought:
                        # 价格突破上轨 + RSI超买，反向做空
                        sell_px = int(mid - price_aggression)
                        orders.append(Order(p, sell_px, -counter_size))
                        self.last_counter_trade[p] = timestamp
                        self.entry_prices[p] = sell_px
                        self.trade_count[p] += 1
                        return orders
                    elif mid < bb["lower"] and self.rsi_values[p] < rsi_oversold:
                        # 价格突破下轨 + RSI超卖，反向做多
                        buy_px = int(mid + price_aggression)
                        orders.append(Order(p, buy_px, counter_size))
                        self.last_counter_trade[p] = timestamp
                        self.entry_prices[p] = buy_px
                        self.trade_count[p] += 1
                        return orders
                
                # 反趋势交易逻辑
                if trend == 1 and self.rsi_values[p] > rsi_overbought:
                    # 强烈上升趋势 + RSI超买，反向做空
                    sell_px = int(mid - price_aggression)
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    self.entry_prices[p] = sell_px
                    self.trade_count[p] += 1
                    return orders
                elif trend == -1 and self.rsi_values[p] < rsi_oversold:
                    # 强烈下降趋势 + RSI超卖，反向做多
                    buy_px = int(mid + price_aggression)
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    self.entry_prices[p] = buy_px
                    self.trade_count[p] += 1
                    return orders
        
        # 如果没有波动率交易信号，执行常规做市
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
        
        # 更新波动率指标
        self.update_volatility_metrics()
        
        # 对每个产品应用波动率交易策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.volatility_trading_strategy(
                    p, 
                    depth, 
                    state.position.get(p, 0),
                    state.timestamp
                )
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
