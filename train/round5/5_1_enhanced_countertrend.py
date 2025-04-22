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
    # 高波动性产品 - 适用完整反趋势策略
    "high_vol": {
        "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
        "PICNIC_BASKET1", "PICNIC_BASKET2"
    },
    # 中等波动性产品 - 适用部分反趋势策略
    "medium_vol": {
        "MAGNIFICENT_MACARONS", "VOLCANIC_ROCK_VOUCHER_10000", 
        "VOLCANIC_ROCK_VOUCHER_10250", "DJEMBES", "JAMS"
    },
    # 低波动性产品 - 主要做市策略
    "low_vol": {
        "CROISSANTS", "KELP", "VOLCANIC_ROCK_VOUCHER_10500", "RAINFOREST_RESIN", "SQUID_INK"
    }
}

# 参数
PARAM = {
    # 趋势检测参数
    "trend_window_short": 15,    # 短期趋势窗口
    "trend_window_medium": 30,   # 中期趋势窗口
    "trend_window_long": 50,     # 长期趋势窗口
    "trend_threshold": 0.7,      # 趋势检测阈值
    
    # 反转检测参数
    "reversal_window": 5,        # 反转检测窗口
    "reversal_threshold": 0.8,   # 反转检测阈值
    "reversal_cooldown": 10,     # 反转交易冷却期
    
    # 仓位管理参数
    "position_limit_pct": {      # 不同类别产品的仓位限制
        "high_vol": 0.5,         # 高波动性产品使用50%仓位
        "medium_vol": 0.7,       # 中等波动性产品使用70%仓位
        "low_vol": 0.9           # 低波动性产品使用90%仓位
    },
    "mm_size_frac": {            # 不同类别产品的做市规模
        "high_vol": 0.15,
        "medium_vol": 0.2,
        "low_vol": 0.25
    },
    "counter_size_frac": {       # 不同类别产品的反趋势交易规模
        "high_vol": 0.3,
        "medium_vol": 0.2,
        "low_vol": 0.1
    },
    
    # 价格参数
    "min_spread": {              # 不同类别产品的最小价差
        "high_vol": 3,
        "medium_vol": 2,
        "low_vol": 1
    },
    "vol_scale": 1.2,            # 波动率缩放因子
    "price_aggression": {        # 价格激进程度
        "high_vol": 2,
        "medium_vol": 1,
        "low_vol": 0
    },
    
    # 技术指标参数
    "rsi_period": 14,            # RSI周期
    "rsi_overbought": 75,        # RSI超买阈值
    "rsi_oversold": 25,          # RSI超卖阈值
    "bollinger_period": 20,      # 布林带周期
    "bollinger_std": 2.0,        # 布林带标准差
    
    # 风险控制参数
    "max_loss_per_trade": 50,    # 每笔交易最大损失
    "max_drawdown": 200,         # 最大回撤
    "stop_loss_pct": 0.03,       # 止损百分比
    
    # 订单簿分析参数
    "imbalance_threshold": 0.3,  # 订单簿不平衡阈值
    "depth_levels": 3            # 分析的订单簿深度
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.returns = defaultdict(list)  # 价格回报率
        self.volumes = defaultdict(list)  # 交易量估计
        self.position_history = defaultdict(list)  # 仓位历史
        self.rsi_values = defaultdict(float)  # RSI值
        self.bollinger_bands = defaultdict(dict)  # 布林带
        self.last_counter_trade = defaultdict(int)  # 上次反向交易的时间戳
        self.trade_profits = defaultdict(list)  # 交易盈亏
        self.entry_prices = defaultdict(float)  # 入场价格
        self.max_position_values = defaultdict(float)  # 最大仓位价值
        self.order_book_imbalance = defaultdict(float)  # 订单簿不平衡度
        self.trends = defaultdict(dict)  # 不同时间框架的趋势
        self.reversal_signals = defaultdict(int)  # 反转信号强度
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) * PARAM["vol_scale"] or 1
        
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
            
        # 计算移动平均线
        recent_prices = self.prices[p][-PARAM["bollinger_period"]:]
        sma = sum(recent_prices) / len(recent_prices)
        
        # 计算标准差
        std = statistics.stdev(recent_prices)
        
        # 计算上下轨
        upper = sma + PARAM["bollinger_std"] * std
        lower = sma - PARAM["bollinger_std"] * std
        
        self.bollinger_bands[p] = {
            "upper": upper,
            "middle": sma,
            "lower": lower
        }
    
    def calculate_order_book_imbalance(self, depth: OrderDepth) -> float:
        """计算订单簿不平衡度"""
        if not depth.buy_orders or not depth.sell_orders:
            return 0
            
        total_buy_volume = sum(abs(qty) for qty in depth.buy_orders.values())
        total_sell_volume = sum(abs(qty) for qty in depth.sell_orders.values())
        
        if total_buy_volume + total_sell_volume == 0:
            return 0
            
        # 计算不平衡度 (-1到1之间，正值表示买方压力大，负值表示卖方压力大)
        imbalance = (total_buy_volume - total_sell_volume) / (total_buy_volume + total_sell_volume)
        
        return imbalance
    
    def detect_trend(self, p: str, window: int) -> int:
        """检测指定窗口的市场趋势"""
        prices = self.prices[p]
        if len(prices) < window:
            return 0  # 数据不足
            
        recent_prices = prices[-window:]  
        up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
        
        up_ratio = up_moves / (len(recent_prices) - 1)
        down_ratio = down_moves / (len(recent_prices) - 1)
        
        if up_ratio > PARAM["trend_threshold"]:
            return 1  # 上升趋势
        elif down_ratio > PARAM["trend_threshold"]:
            return -1  # 下降趋势
        return 0  # 中性
    
    def update_trends(self, p: str):
        """更新不同时间框架的趋势"""
        self.trends[p] = {
            "short": self.detect_trend(p, PARAM["trend_window_short"]),
            "medium": self.detect_trend(p, PARAM["trend_window_medium"]),
            "long": self.detect_trend(p, PARAM["trend_window_long"])
        }
    
    def detect_reversal(self, p: str) -> int:
        """检测市场反转"""
        # 如果数据不足，无法检测反转
        if len(self.prices[p]) < PARAM["reversal_window"] + 5:
            return 0
            
        # 获取不同时间框架的趋势
        short_trend = self.trends[p]["short"]
        medium_trend = self.trends[p]["medium"]
        
        # 如果短期趋势与中期趋势相反，可能是反转信号
        if short_trend != 0 and medium_trend != 0 and short_trend != medium_trend:
            return short_trend  # 返回短期趋势方向作为反转方向
            
        # 检查最近的价格变动
        recent_prices = self.prices[p][-PARAM["reversal_window"]:]
        prev_prices = self.prices[p][-PARAM["reversal_window"]-5:-PARAM["reversal_window"]]
        
        # 如果中期趋势是上升，但最近价格开始下跌
        if medium_trend == 1:
            down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
            down_ratio = down_moves / (len(recent_prices) - 1)
            
            if down_ratio > PARAM["reversal_threshold"]:
                return -1  # 上升趋势反转为下降
        
        # 如果中期趋势是下降，但最近价格开始上涨
        elif medium_trend == -1:
            up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
            up_ratio = up_moves / (len(recent_prices) - 1)
            
            if up_ratio > PARAM["reversal_threshold"]:
                return 1  # 下降趋势反转为上升
                
        return 0  # 没有检测到反转
    
    def check_technical_signals(self, p: str, current_price: float) -> int:
        """检查技术指标信号"""
        signal = 0
        
        # RSI信号
        rsi = self.rsi_values[p]
        if rsi > PARAM["rsi_overbought"]:
            signal -= 1  # 超买，看跌信号
        elif rsi < PARAM["rsi_oversold"]:
            signal += 1  # 超卖，看涨信号
        
        # 布林带信号
        bb = self.bollinger_bands[p]
        if bb["upper"] is not None and bb["lower"] is not None:
            if current_price > bb["upper"]:
                signal -= 1  # 价格超过上轨，看跌信号
            elif current_price < bb["lower"]:
                signal += 1  # 价格低于下轨，看涨信号
        
        # 订单簿不平衡信号
        imbalance = self.order_book_imbalance[p]
        if abs(imbalance) > PARAM["imbalance_threshold"]:
            if imbalance > 0:
                signal += 1  # 买方压力大，看涨信号
            else:
                signal -= 1  # 卖方压力大，看跌信号
        
        return signal
    
    def check_risk_controls(self, p: str, pos: int, current_price: float) -> bool:
        """检查风险控制，返回是否允许交易"""
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
            if (pos > 0 and price_change_pct < -PARAM["stop_loss_pct"]) or \
               (pos < 0 and price_change_pct > PARAM["stop_loss_pct"]):
                return False  # 触发止损，不允许继续同方向交易
            
            # 检查最大回撤
            drawdown = self.max_position_values[p] - current_value
            if drawdown > PARAM["max_drawdown"]:
                return False  # 触发最大回撤，不允许继续同方向交易
        
        return True  # 通过风险控制，允许交易
    
    def enhanced_countertrend_strategy(self, p: str, depth: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        """增强型反趋势交易策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 记录仓位
        self.position_history[p].append(pos)
        
        # 计算订单簿不平衡度
        imbalance = self.calculate_order_book_imbalance(depth)
        self.order_book_imbalance[p] = imbalance
        
        # 更新技术指标
        self.calculate_rsi(p)
        self.calculate_bollinger_bands(p)
        
        # 更新趋势
        self.update_trends(p)
        
        # 检测反转
        reversal = self.detect_reversal(p)
        self.reversal_signals[p] = reversal
        
        # 检查技术指标信号
        tech_signal = self.check_technical_signals(p, mid)
        
        # 获取产品类别
        category = self.get_product_category(p)
        
        # 检查风险控制
        risk_ok = self.check_risk_controls(p, pos, mid)
        
        # 计算价差
        vol = self._vol(p)
        min_spread = PARAM["min_spread"][category]
        spread = max(min_spread, int(vol))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模
        max_position = int(LIMIT[p] * PARAM["position_limit_pct"][category])
        mm_size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"][category]))
        counter_size = max(1, int(LIMIT[p] * PARAM["counter_size_frac"][category]))
        
        # 反趋势交易逻辑
        if category in ["high_vol", "medium_vol"]:  # 只对高波动性和中等波动性产品应用反趋势策略
            # 检查是否在冷却期
            cooldown_active = timestamp - self.last_counter_trade.get(p, 0) < PARAM["reversal_cooldown"]
            
            if not cooldown_active and risk_ok:
                # 强烈的反转信号
                if reversal != 0 and abs(tech_signal) >= 2:
                    price_aggression = PARAM["price_aggression"][category]
                    
                    if reversal == 1:  # 下降趋势反转为上升
                        # 积极买入
                        buy_px = int(mid + price_aggression)
                        orders.append(Order(p, buy_px, counter_size))
                        self.last_counter_trade[p] = timestamp
                        self.entry_prices[p] = buy_px
                        return orders
                        
                    elif reversal == -1:  # 上升趋势反转为下降
                        # 积极卖出
                        sell_px = int(mid - price_aggression)
                        orders.append(Order(p, sell_px, -counter_size))
                        self.last_counter_trade[p] = timestamp
                        self.entry_prices[p] = sell_px
                        return orders
                
                # RSI超买超卖信号
                elif self.rsi_values[p] > PARAM["rsi_overbought"] and self.trends[p]["medium"] == 1:
                    # 超买 + 上升趋势，反向做空
                    sell_px = int(mid - PARAM["price_aggression"][category])
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    self.entry_prices[p] = sell_px
                    return orders
                    
                elif self.rsi_values[p] < PARAM["rsi_oversold"] and self.trends[p]["medium"] == -1:
                    # 超卖 + 下降趋势，反向做多
                    buy_px = int(mid + PARAM["price_aggression"][category])
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    self.entry_prices[p] = buy_px
                    return orders
                
                # 布林带突破信号
                elif self.bollinger_bands[p]["upper"] is not None and mid > self.bollinger_bands[p]["upper"]:
                    # 价格突破上轨，反向做空
                    sell_px = int(mid - PARAM["price_aggression"][category])
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    self.entry_prices[p] = sell_px
                    return orders
                    
                elif self.bollinger_bands[p]["lower"] is not None and mid < self.bollinger_bands[p]["lower"]:
                    # 价格突破下轨，反向做多
                    buy_px = int(mid + PARAM["price_aggression"][category])
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    self.entry_prices[p] = buy_px
                    return orders
        
        # 如果没有反趋势交易信号，执行常规做市
        # 根据趋势调整价格
        medium_trend = self.trends[p]["medium"]
        if medium_trend == 1:  # 上升趋势
            buy_px += 1
            sell_px += 1
        elif medium_trend == -1:  # 下降趋势
            buy_px -= 1
            sell_px -= 1
        
        # 根据订单簿不平衡调整价格
        if abs(imbalance) > PARAM["imbalance_threshold"]:
            if imbalance > 0:  # 买方压力大
                buy_px += 1
                sell_px += 1
            else:  # 卖方压力大
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
        
        # 对每个产品应用增强型反趋势策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.enhanced_countertrend_strategy(
                    p, 
                    depth, 
                    state.position.get(p, 0),
                    state.timestamp
                )
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
