from typing import Dict, List, Tuple, Optional, Set
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np

# 火山岩及其期权
VOLCANIC_PRODUCTS = {
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,  # 行权价9500的看涨期权
    "VOLCANIC_ROCK_VOUCHER_9750": 200,  # 行权价9750的看涨期权
    "VOLCANIC_ROCK_VOUCHER_10000": 200, # 行权价10000的看涨期权
    "VOLCANIC_ROCK_VOUCHER_10250": 200, # 行权价10250的看涨期权
    "VOLCANIC_ROCK_VOUCHER_10500": 200  # 行权价10500的看涨期权
}

# 其他产品（不会主动交易）
OTHER_PRODUCTS = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "MAGNIFICENT_MACARONS": 75
}

# 合并所有产品限制
LIMIT = {**VOLCANIC_PRODUCTS, **OTHER_PRODUCTS}

# 参数设置 - 增强型方向性期权
PARAM = {
    # 技术指标参数
    "trend_window_short": 8,     # 短期趋势窗口
    "trend_window_medium": 15,   # 中期趋势窋口
    "trend_window_long": 30,     # 长期趋势窗口
    "trend_threshold": 0.7,      # 趋势确认阈值
    "rsi_period": 14,            # RSI周期
    "rsi_overbought": 75,        # RSI超买阈值
    "rsi_oversold": 25,          # RSI超卖阈值
    "bollinger_period": 20,      # 布林带周期
    "bollinger_std": 2.0,        # 布林带标准差
    "macd_fast": 12,             # MACD快线
    "macd_slow": 26,             # MACD慢线
    "macd_signal": 9,            # MACD信号线
    
    # 交易参数
    "max_position_pct": 0.9,     # 最大仓位百分比
    "directional_size_pct": 0.6, # 方向性交易规模百分比
    "aggressive_entry": True,    # 激进进场
    "aggressive_exit": True,     # 激进出场
    "vol_scale": 1.2,            # 波动率缩放因子
    "vol_window": 15,            # 波动率窗口
    "max_dir_attempts": 10,      # 每个时间步最大方向性交易尝试次数
    "min_profit_target": 5,      # 最小利润目标
    
    # 风险控制参数
    "stop_loss_pct": 0.03,       # 止损阈值 (3%)
    "take_profit_pct": 0.05,     # 止盈阈值 (5%)
    "max_drawdown": 200,         # 最大回撤
    "cooldown_period": 5,        # 冷却期
    
    # 信号系统参数
    "signal_threshold": 3.0,     # 信号阈值
    "signal_weights": {          # 各指标权重
        "trend": 1.0,
        "rsi": 0.8,
        "bollinger": 0.7,
        "macd": 0.9,
        "volume": 0.6
    },
    "confirmation_count": 3      # 需要确认的指标数量
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.returns = defaultdict(list)  # 价格回报率
        self.volumes = defaultdict(list)  # 交易量
        self.position_history = defaultdict(list)  # 仓位历史
        self.fair_values = {}  # 产品公平价值
        self.trends = defaultdict(int)  # 1=上升, -1=下降, 0=中性
        self.signal_strength = defaultdict(float)  # 信号强度
        self.rsi_values = defaultdict(float)  # RSI值
        self.bollinger_bands = defaultdict(dict)  # 布林带
        self.macd = defaultdict(dict)  # MACD指标
        self.option_deltas = defaultdict(float)  # 期权Delta值
        self.stop_loss_active = defaultdict(bool)  # 止损激活状态
        self.take_profit_active = defaultdict(bool)  # 止盈激活状态
        self.cooldown_counter = defaultdict(int)  # 冷却计数器
        self.entry_prices = defaultdict(float)  # 入场价格
        self.max_drawdown_values = defaultdict(float)  # 最大回撤值
        self.confirmed_signals = defaultdict(set)  # 已确认的信号
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < PARAM["vol_window"]:
            return 2  # 默认中等波动率
        return statistics.stdev(h[-PARAM["vol_window"]:]) * PARAM["vol_scale"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def update_technical_indicators(self, state: TradingState):
        """更新技术指标"""
        for p, depth in state.order_depths.items():
            if p not in VOLCANIC_PRODUCTS:
                continue  # 只关注火山岩产品
                
            mid = self._mid(depth)
            if mid is None:
                continue
                
            # 记录价格
            self.prices[p].append(mid)
            self.fair_values[p] = mid
            
            # 记录仓位
            self.position_history[p].append(state.position.get(p, 0))
            
            # 估计交易量
            volume = sum(abs(qty) for qty in depth.buy_orders.values()) + sum(abs(qty) for qty in depth.sell_orders.values())
            self.volumes[p].append(volume)
            
            # 计算回报率
            if len(self.prices[p]) > 1:
                ret = (mid / self.prices[p][-2]) - 1
                self.returns[p].append(ret)
            
            # 保持历史记录在合理范围内
            max_history = 100
            if len(self.prices[p]) > max_history:
                self.prices[p].pop(0)
            if len(self.returns[p]) > max_history:
                self.returns[p].pop(0)
            if len(self.volumes[p]) > max_history:
                self.volumes[p].pop(0)
            
            # 计算各种技术指标
            self.calculate_trend(p)
            self.calculate_rsi(p)
            self.calculate_bollinger_bands(p)
            self.calculate_macd(p)
            
            # 计算综合信号强度
            self.calculate_signal_strength(p)
            
            # 如果是期权，计算Delta值
            if p.startswith("VOLCANIC_ROCK_VOUCHER"):
                self.calculate_option_delta(p)
    
    def calculate_trend(self, product: str):
        """计算趋势"""
        prices = self.prices[product]
        
        # 如果数据不足，返回中性
        if len(prices) < PARAM["trend_window_long"]:
            self.trends[product] = 0
            return
            
        # 计算短期、中期和长期趋势
        short_trend = self.calculate_trend_direction(prices, PARAM["trend_window_short"])
        medium_trend = self.calculate_trend_direction(prices, PARAM["trend_window_medium"])
        long_trend = self.calculate_trend_direction(prices, PARAM["trend_window_long"])
        
        # 综合趋势判断
        if short_trend == medium_trend == long_trend:
            # 三个时间框架趋势一致，强信号
            self.trends[product] = short_trend
        elif short_trend == medium_trend:
            # 短期和中期趋势一致，中等信号
            self.trends[product] = short_trend
        elif medium_trend == long_trend:
            # 中期和长期趋势一致，中等信号
            self.trends[product] = medium_trend
        else:
            # 趋势不一致，弱信号或中性
            self.trends[product] = short_trend if abs(short_trend) > abs(medium_trend) else medium_trend
    
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
        
        if up_ratio > PARAM["trend_threshold"]:
            return 1  # 上升趋势
        elif down_ratio > PARAM["trend_threshold"]:
            return -1  # 下降趋势
        return 0  # 中性
    
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
    
    def calculate_bollinger_bands(self, product: str):
        """计算布林带"""
        prices = self.prices[product]
        
        if len(prices) < PARAM["bollinger_period"]:
            self.bollinger_bands[product] = {"upper": None, "middle": None, "lower": None}
            return
            
        # 计算移动平均线
        recent_prices = prices[-PARAM["bollinger_period"]:]
        sma = sum(recent_prices) / len(recent_prices)
        
        # 计算标准差
        std = statistics.stdev(recent_prices)
        
        # 计算上下轨
        upper = sma + PARAM["bollinger_std"] * std
        lower = sma - PARAM["bollinger_std"] * std
        
        self.bollinger_bands[product] = {
            "upper": upper,
            "middle": sma,
            "lower": lower
        }
    
    def calculate_macd(self, product: str):
        """计算MACD指标"""
        prices = self.prices[product]
        
        if len(prices) < PARAM["macd_slow"]:
            self.macd[product] = {"macd": 0, "signal": 0, "histogram": 0}
            return
            
        # 计算EMA
        ema_fast = self.calculate_ema(prices, PARAM["macd_fast"])
        ema_slow = self.calculate_ema(prices, PARAM["macd_slow"])
        
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线
        macd_values = [self.macd[product].get("macd", 0)] * (PARAM["macd_signal"] - 1)
        macd_values.append(macd_line)
        signal_line = self.calculate_ema(macd_values, PARAM["macd_signal"])
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        self.macd[product] = {
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
    
    def calculate_signal_strength(self, product: str):
        """计算综合信号强度"""
        # 初始信号强度为0
        strength = 0
        self.confirmed_signals[product] = set()
        
        # 1. 趋势信号
        trend = self.trends[product]
        trend_signal = trend * PARAM["signal_weights"]["trend"]
        strength += trend_signal
        if abs(trend_signal) >= 0.7:
            self.confirmed_signals[product].add("trend")
        
        # 2. RSI信号
        rsi = self.rsi_values[product]
        rsi_signal = 0
        if rsi > PARAM["rsi_overbought"]:
            rsi_signal = -1.0 * PARAM["signal_weights"]["rsi"]  # 超买，看跌信号
            self.confirmed_signals[product].add("rsi_overbought")
        elif rsi < PARAM["rsi_oversold"]:
            rsi_signal = 1.0 * PARAM["signal_weights"]["rsi"]  # 超卖，看涨信号
            self.confirmed_signals[product].add("rsi_oversold")
        strength += rsi_signal
        
        # 3. 布林带信号
        bb = self.bollinger_bands[product]
        bb_signal = 0
        if bb["upper"] is not None and bb["lower"] is not None:
            current_price = self.prices[product][-1]
            if current_price > bb["upper"]:
                bb_signal = -1.0 * PARAM["signal_weights"]["bollinger"]  # 突破上轨，看跌信号
                self.confirmed_signals[product].add("bb_upper")
            elif current_price < bb["lower"]:
                bb_signal = 1.0 * PARAM["signal_weights"]["bollinger"]  # 突破下轨，看涨信号
                self.confirmed_signals[product].add("bb_lower")
        strength += bb_signal
        
        # 4. MACD信号
        macd_data = self.macd[product]
        macd_signal = 0
        if macd_data["macd"] > macd_data["signal"]:
            macd_signal = 1.0 * PARAM["signal_weights"]["macd"]  # MACD在信号线上方，看涨信号
            self.confirmed_signals[product].add("macd_bullish")
        elif macd_data["macd"] < macd_data["signal"]:
            macd_signal = -1.0 * PARAM["signal_weights"]["macd"]  # MACD在信号线下方，看跌信号
            self.confirmed_signals[product].add("macd_bearish")
        strength += macd_signal
        
        # 5. 交易量信号
        volume_signal = 0
        if len(self.volumes[product]) > 5:
            avg_volume = sum(self.volumes[product][-5:]) / 5
            current_volume = self.volumes[product][-1]
            if current_volume > avg_volume * 1.5:
                # 交易量突然增加，与价格方向一致
                volume_signal = trend * PARAM["signal_weights"]["volume"]
                if abs(volume_signal) >= 0.3:
                    self.confirmed_signals[product].add("volume")
        strength += volume_signal
        
        self.signal_strength[product] = strength
    
    def check_risk_controls(self, product: str, current_price: float) -> bool:
        """检查风险控制，返回是否允许交易"""
        # 如果在冷却期，减少冷却计数器
        if self.cooldown_counter[product] > 0:
            self.cooldown_counter[product] -= 1
            return False  # 冷却期内不交易
        
        # 如果没有入场价格，设置当前价格为入场价格
        if self.entry_prices[product] == 0:
            self.entry_prices[product] = current_price
            return True
        
        # 计算价格变动百分比
        price_change_pct = (current_price - self.entry_prices[product]) / self.entry_prices[product]
        
        # 检查止损
        if abs(price_change_pct) > PARAM["stop_loss_pct"] and price_change_pct < 0:
            if not self.stop_loss_active[product]:
                self.stop_loss_active[product] = True
                self.cooldown_counter[product] = PARAM["cooldown_period"]
                # 重置入场价格
                self.entry_prices[product] = 0
                return False  # 触发止损，不交易
        
        # 检查止盈
        if price_change_pct > PARAM["take_profit_pct"]:
            if not self.take_profit_active[product]:
                self.take_profit_active[product] = True
                self.cooldown_counter[product] = PARAM["cooldown_period"]
                # 重置入场价格
                self.entry_prices[product] = 0
                return False  # 触发止盈，不交易
        
        # 检查最大回撤
        drawdown = max(0, self.max_drawdown_values[product] - current_price)
        if drawdown > PARAM["max_drawdown"]:
            self.cooldown_counter[product] = PARAM["cooldown_period"]
            # 重置入场价格和最大回撤值
            self.entry_prices[product] = 0
            self.max_drawdown_values[product] = 0
            return False  # 触发最大回撤，不交易
        
        # 更新最大回撤值
        if current_price > self.max_drawdown_values[product]:
            self.max_drawdown_values[product] = current_price
        
        return True  # 通过风险控制，允许交易
    
    def find_directional_opportunities(self, state: TradingState) -> List[Dict]:
        """寻找方向性交易机会"""
        opportunities = []
        
        for p in VOLCANIC_PRODUCTS:
            if p not in state.order_depths:
                continue
                
            depth = state.order_depths[p]
            mid = self._mid(depth)
            if mid is None:
                continue
                
            # 获取信号强度
            strength = self.signal_strength.get(p, 0)
            
            # 检查风险控制
            if not self.check_risk_controls(p, mid):
                continue
                
            # 如果信号强度不够强或确认信号不足，不进行方向性交易
            if (abs(strength) < PARAM["signal_threshold"] or 
                len(self.confirmed_signals[p]) < PARAM["confirmation_count"]):
                continue
                
            # 确定交易方向
            direction = 1 if strength > 0 else -1
            
            # 根据方向选择合适的交易机会
            if direction == 1:  # 看涨
                # 如果是期权，选择合适的期权进行杠杆交易
                if p.startswith("VOLCANIC_ROCK_VOUCHER"):
                    # 确保期权在订单深度中
                    a = min(depth.sell_orders.keys()) if depth.sell_orders else None
                    
                    if a is not None:
                        # 添加买入看涨期权的机会
                        opportunities.append({
                            "type": "buy_call",
                            "product": p,
                            "price": a,
                            "signal_strength": strength,
                            "confirmed_signals": len(self.confirmed_signals[p]),
                            "max_size": min(
                                abs(depth.sell_orders[a]),
                                int(VOLCANIC_PRODUCTS[p] * PARAM["directional_size_pct"])
                            )
                        })
                elif p == "VOLCANIC_ROCK":
                    # 对基础资产，直接买入
                    a = min(depth.sell_orders.keys()) if depth.sell_orders else None
                    
                    if a is not None:
                        # 添加买入基础资产的机会
                        opportunities.append({
                            "type": "buy_underlying",
                            "product": p,
                            "price": a,
                            "signal_strength": strength,
                            "confirmed_signals": len(self.confirmed_signals[p]),
                            "max_size": min(
                                abs(depth.sell_orders[a]),
                                int(VOLCANIC_PRODUCTS[p] * PARAM["directional_size_pct"])
                            )
                        })
            
            else:  # 看跌
                # 如果是期权，选择合适的期权进行杠杆交易
                if p.startswith("VOLCANIC_ROCK_VOUCHER"):
                    # 确保期权在订单深度中
                    b = max(depth.buy_orders.keys()) if depth.buy_orders else None
                    
                    if b is not None:
                        # 添加卖出看涨期权的机会
                        opportunities.append({
                            "type": "sell_call",
                            "product": p,
                            "price": b,
                            "signal_strength": strength,
                            "confirmed_signals": len(self.confirmed_signals[p]),
                            "max_size": min(
                                depth.buy_orders[b],
                                int(VOLCANIC_PRODUCTS[p] * PARAM["directional_size_pct"])
                            )
                        })
                elif p == "VOLCANIC_ROCK":
                    # 对基础资产，直接卖出
                    b = max(depth.buy_orders.keys()) if depth.buy_orders else None
                    
                    if b is not None:
                        # 添加卖出基础资产的机会
                        opportunities.append({
                            "type": "sell_underlying",
                            "product": p,
                            "price": b,
                            "signal_strength": strength,
                            "confirmed_signals": len(self.confirmed_signals[p]),
                            "max_size": min(
                                depth.buy_orders[b],
                                int(VOLCANIC_PRODUCTS[p] * PARAM["directional_size_pct"])
                            )
                        })
        
        # 按信号强度和确认信号数量排序
        opportunities.sort(key=lambda x: (x["confirmed_signals"], abs(x["signal_strength"])), reverse=True)
        
        return opportunities
    
    def execute_directional_trading(self, state: TradingState, opportunities: List[Dict]) -> Dict[str, List[Order]]:
        """执行方向性交易策略"""
        result = {}
        
        # 限制每个时间步的交易尝试次数
        attempts = 0
        
        for opp in opportunities:
            if attempts >= PARAM["max_dir_attempts"]:
                break
                
            product = opp["product"]
            price = opp["price"]
            
            if opp["type"] == "buy_call" or opp["type"] == "buy_underlying":
                # 买入
                size = min(opp["max_size"], VOLCANIC_PRODUCTS[product] - state.position.get(product, 0))
                
                if size > 0:
                    if product not in result:
                        result[product] = []
                    result[product].append(Order(product, price, size))
                    attempts += 1
            
            elif opp["type"] == "sell_call" or opp["type"] == "sell_underlying":
                # 卖出
                size = min(opp["max_size"], VOLCANIC_PRODUCTS[product] + state.position.get(product, 0))
                
                if size > 0:
                    if product not in result:
                        result[product] = []
                    result[product].append(Order(product, price, -size))
                    attempts += 1
        
        return result
    
    def standard_market_making(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """标准做市策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 记录价格历史
        if p not in self.prices or len(self.prices[p]) == 0 or self.prices[p][-1] != mid:
            self.prices[p].append(mid)
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差
        spread = int(2 + vol)
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模
        max_position = int(VOLCANIC_PRODUCTS[p] * 0.3)  # 使用30%的仓位限制
        size = max(1, max_position // 4)  # 每次用1/4的允许仓位
        
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
        
        # 寻找方向性交易机会
        opportunities = self.find_directional_opportunities(state)
        
        # 执行方向性交易策略
        if opportunities:
            dir_orders = self.execute_directional_trading(state, opportunities)
            result.update(dir_orders)
        
        # 对未在方向性交易中交易的产品应用标准做市策略
        for p in VOLCANIC_PRODUCTS:
            # 如果已经在方向性交易中交易了这个产品，跳过
            if p in result:
                continue
                
            if p in state.order_depths:
                depth = state.order_depths[p]
                pos = state.position.get(p, 0)
                result[p] = self.standard_market_making(p, depth, pos)
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
