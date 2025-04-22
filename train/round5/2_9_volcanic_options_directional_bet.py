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

# 参数设置 - 方向性押注
PARAM = {
    "trend_window_short": 10,    # 短期趋势窗口
    "trend_window_medium": 20,   # 中期趋势窗口
    "trend_window_long": 40,     # 长期趋势窗口
    "trend_threshold": 0.7,      # 趋势确认阈值
    "max_position_pct": 0.95,    # 最大仓位百分比
    "directional_size_pct": 0.5, # 方向性交易规模百分比
    "aggressive_entry": True,    # 激进进场
    "aggressive_exit": True,     # 激进出场
    "vol_scale": 1.5,            # 波动率缩放因子
    "max_dir_attempts": 10,      # 每个时间步最大方向性交易尝试次数
    "min_profit_target": 10,     # 最小利润目标
    "momentum_lookback": 5,      # 动量回溯期
    "momentum_threshold": 0.6,   # 动量阈值
    "rsi_period": 14,            # RSI周期
    "rsi_overbought": 70,        # RSI超买阈值
    "rsi_oversold": 30,          # RSI超卖阈值
    "macd_fast": 12,             # MACD快线
    "macd_slow": 26,             # MACD慢线
    "macd_signal": 9,            # MACD信号线
    "signal_strength_threshold": 2.0  # 信号强度阈值
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.returns = defaultdict(list)  # 价格回报率
        self.fair_values = {}
        self.option_deltas = defaultdict(float)  # 期权Delta值
        self.trends = defaultdict(int)  # 1=上升, -1=下降, 0=中性
        self.signal_strength = defaultdict(float)  # 信号强度
        self.rsi_values = defaultdict(float)  # RSI值
        self.macd = defaultdict(dict)  # MACD指标
        self.last_signal = defaultdict(int)  # 上次信号
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < 15:
            return 3  # 默认较高波动率
        return statistics.stdev(h[-15:]) * PARAM["vol_scale"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def calculate_technical_indicators(self, state: TradingState):
        """计算技术指标"""
        # 首先计算VOLCANIC_ROCK的指标
        if "VOLCANIC_ROCK" in state.order_depths:
            depth = state.order_depths["VOLCANIC_ROCK"]
            mid = self._mid(depth)
            if mid is None:
                return
                
            self.fair_values["VOLCANIC_ROCK"] = mid
            
            # 记录价格
            self.prices["VOLCANIC_ROCK"].append(mid)
            
            # 计算回报率
            if len(self.prices["VOLCANIC_ROCK"]) > 1:
                ret = (mid / self.prices["VOLCANIC_ROCK"][-2]) - 1
                self.returns["VOLCANIC_ROCK"].append(ret)
            
            # 保持价格和回报率历史记录在合理范围内
            if len(self.prices["VOLCANIC_ROCK"]) > 100:
                self.prices["VOLCANIC_ROCK"].pop(0)
            if len(self.returns["VOLCANIC_ROCK"]) > 100:
                self.returns["VOLCANIC_ROCK"].pop(0)
            
            # 计算趋势
            self.trends["VOLCANIC_ROCK"] = self.detect_trend("VOLCANIC_ROCK")
            
            # 计算RSI
            self.calculate_rsi("VOLCANIC_ROCK")
            
            # 计算MACD
            self.calculate_macd("VOLCANIC_ROCK")
            
            # 计算信号强度
            self.calculate_signal_strength("VOLCANIC_ROCK")
            
            # 计算各期权的Delta值
            for voucher in [p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"]:
                # 从期权名称中提取行权价
                try:
                    strike = int(voucher.split("_")[-1])
                    
                    # 简化的Delta计算
                    if mid > strike:  # 实值期权
                        delta = 0.8
                    elif mid < strike - 200:  # 深度虚值期权
                        delta = 0.2
                    else:  # 平值附近的期权
                        delta = 0.5
                    
                    self.option_deltas[voucher] = delta
                except ValueError:
                    continue
    
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
        
        # 3. MACD信号
        macd_data = self.macd[product]
        if macd_data["macd"] > macd_data["signal"]:
            strength += 0.5  # MACD在信号线上方，看涨信号
        elif macd_data["macd"] < macd_data["signal"]:
            strength -= 0.5  # MACD在信号线下方，看跌信号
        
        # 4. 动量信号
        if len(self.returns[product]) >= PARAM["momentum_lookback"]:
            momentum = sum(self.returns[product][-PARAM["momentum_lookback"]:])
            if momentum > 0:
                strength += 0.5  # 正动量，看涨信号
            else:
                strength -= 0.5  # 负动量，看跌信号
        
        self.signal_strength[product] = strength
    
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
        
        if up_ratio > PARAM["trend_threshold"]:
            return 1  # 上升趋势
        elif down_ratio > PARAM["trend_threshold"]:
            return -1  # 下降趋势
        return 0  # 中性
    
    def find_directional_opportunities(self, state: TradingState) -> List[Dict]:
        """寻找方向性交易机会"""
        opportunities = []
        
        # 如果没有VOLCANIC_ROCK数据，无法进行方向性交易
        if "VOLCANIC_ROCK" not in self.signal_strength:
            return opportunities
            
        # 获取信号强度
        strength = self.signal_strength["VOLCANIC_ROCK"]
        
        # 如果信号强度不够强，不进行方向性交易
        if abs(strength) < PARAM["signal_strength_threshold"]:
            return opportunities
            
        # 确定交易方向
        direction = 1 if strength > 0 else -1
        
        # 根据方向选择合适的期权
        if direction == 1:  # 看涨
            # 选择虚值期权进行杠杆交易
            options = sorted([p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"], 
                             key=lambda x: int(x.split("_")[-1]))
            
            # 找到当前价格附近的虚值期权
            current_price = self.prices["VOLCANIC_ROCK"][-1]
            
            for option in options:
                try:
                    strike = int(option.split("_")[-1])
                    
                    # 寻找略微虚值的期权
                    if strike > current_price and strike < current_price * 1.1:
                        # 确保期权在订单深度中
                        if option in state.order_depths:
                            depth = state.order_depths[option]
                            a = min(depth.sell_orders.keys()) if depth.sell_orders else None
                            
                            if a is not None:
                                # 添加买入看涨期权的机会
                                opportunities.append({
                                    "type": "buy_call",
                                    "option": option,
                                    "price": a,
                                    "strike": strike,
                                    "signal_strength": strength,
                                    "max_size": min(
                                        abs(depth.sell_orders[a]),
                                        int(VOLCANIC_PRODUCTS[option] * PARAM["directional_size_pct"])
                                    )
                                })
                                break  # 只选择一个最合适的期权
                except ValueError:
                    continue
            
            # 同时考虑直接买入基础资产
            if "VOLCANIC_ROCK" in state.order_depths:
                depth = state.order_depths["VOLCANIC_ROCK"]
                a = min(depth.sell_orders.keys()) if depth.sell_orders else None
                
                if a is not None:
                    # 添加买入基础资产的机会
                    opportunities.append({
                        "type": "buy_underlying",
                        "price": a,
                        "signal_strength": strength,
                        "max_size": min(
                            abs(depth.sell_orders[a]),
                            int(VOLCANIC_PRODUCTS["VOLCANIC_ROCK"] * PARAM["directional_size_pct"])
                        )
                    })
        
        else:  # 看跌
            # 选择实值期权进行杠杆交易
            options = sorted([p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"], 
                             key=lambda x: int(x.split("_")[-1]), reverse=True)
            
            # 找到当前价格附近的实值期权
            current_price = self.prices["VOLCANIC_ROCK"][-1]
            
            for option in options:
                try:
                    strike = int(option.split("_")[-1])
                    
                    # 寻找略微实值的期权
                    if strike < current_price and strike > current_price * 0.9:
                        # 确保期权在订单深度中
                        if option in state.order_depths:
                            depth = state.order_depths[option]
                            b = max(depth.buy_orders.keys()) if depth.buy_orders else None
                            
                            if b is not None:
                                # 添加卖出看涨期权的机会
                                opportunities.append({
                                    "type": "sell_call",
                                    "option": option,
                                    "price": b,
                                    "strike": strike,
                                    "signal_strength": strength,
                                    "max_size": min(
                                        depth.buy_orders[b],
                                        int(VOLCANIC_PRODUCTS[option] * PARAM["directional_size_pct"])
                                    )
                                })
                                break  # 只选择一个最合适的期权
                except ValueError:
                    continue
            
            # 同时考虑直接卖出基础资产
            if "VOLCANIC_ROCK" in state.order_depths:
                depth = state.order_depths["VOLCANIC_ROCK"]
                b = max(depth.buy_orders.keys()) if depth.buy_orders else None
                
                if b is not None:
                    # 添加卖出基础资产的机会
                    opportunities.append({
                        "type": "sell_underlying",
                        "price": b,
                        "signal_strength": strength,
                        "max_size": min(
                            depth.buy_orders[b],
                            int(VOLCANIC_PRODUCTS["VOLCANIC_ROCK"] * PARAM["directional_size_pct"])
                        )
                    })
        
        return opportunities
    
    def execute_directional_trading(self, state: TradingState, opportunities: List[Dict]) -> Dict[str, List[Order]]:
        """执行方向性交易策略"""
        result = {}
        
        # 限制每个时间步的交易尝试次数
        attempts = 0
        
        for opp in opportunities:
            if attempts >= PARAM["max_dir_attempts"]:
                break
                
            if opp["type"] == "buy_call":
                # 买入看涨期权
                option = opp["option"]
                price = opp["price"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS[option] - state.position.get(option, 0))
                
                if size > 0:
                    if option not in result:
                        result[option] = []
                    result[option].append(Order(option, price, size))
                    attempts += 1
            
            elif opp["type"] == "sell_call":
                # 卖出看涨期权
                option = opp["option"]
                price = opp["price"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS[option] + state.position.get(option, 0))
                
                if size > 0:
                    if option not in result:
                        result[option] = []
                    result[option].append(Order(option, price, -size))
                    attempts += 1
            
            elif opp["type"] == "buy_underlying":
                # 买入基础资产
                price = opp["price"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS["VOLCANIC_ROCK"] - state.position.get("VOLCANIC_ROCK", 0))
                
                if size > 0:
                    if "VOLCANIC_ROCK" not in result:
                        result["VOLCANIC_ROCK"] = []
                    result["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", price, size))
                    attempts += 1
            
            elif opp["type"] == "sell_underlying":
                # 卖出基础资产
                price = opp["price"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS["VOLCANIC_ROCK"] + state.position.get("VOLCANIC_ROCK", 0))
                
                if size > 0:
                    if "VOLCANIC_ROCK" not in result:
                        result["VOLCANIC_ROCK"] = []
                    result["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", price, -size))
                    attempts += 1
        
        return result
    
    def aggressive_directional_trading(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """激进的方向性交易策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 如果不是VOLCANIC_ROCK，记录价格历史
        if p != "VOLCANIC_ROCK":
            self.prices[p].append(mid)
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差 - 使用更窄的价差
        spread = max(1, int(vol * 0.3))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模 - 使用更大的规模
        max_position = int(VOLCANIC_PRODUCTS[p] * PARAM["max_position_pct"])
        size = max(1, max_position // 3)  # 每次用1/3的允许仓位
        
        # 根据VOLCANIC_ROCK的信号强度调整策略
        if "VOLCANIC_ROCK" in self.signal_strength:
            strength = self.signal_strength["VOLCANIC_ROCK"]
            
            if abs(strength) >= PARAM["signal_strength_threshold"]:
                direction = 1 if strength > 0 else -1
                
                if direction == 1:  # 看涨信号
                    # 更积极地买入
                    buy_px = int(mid - spread * 0.5)  # 提高买入价格
                    sell_px = int(mid + spread * 1.5)  # 提高卖出价格
                    size_buy = int(size * 1.5)  # 增加买入规模
                    size_sell = int(size * 0.5)  # 减少卖出规模
                else:  # 看跌信号
                    # 更积极地卖出
                    buy_px = int(mid - spread * 1.5)  # 降低买入价格
                    sell_px = int(mid + spread * 0.5)  # 降低卖出价格
                    size_buy = int(size * 0.5)  # 减少买入规模
                    size_sell = int(size * 1.5)  # 增加卖出规模
            else:
                # 中性信号
                size_buy = size
                size_sell = size
        else:
            # 没有信号强度数据
            size_buy = size
            size_sell = size
        
        # 主动吃单
        if PARAM["aggressive_entry"]:
            b, a = best_bid_ask(depth)
            if a is not None and b is not None:
                # 如果卖单价格低于我们的买入价，主动买入
                if a < buy_px and pos < max_position:
                    qty = min(size_buy, max_position - pos, abs(depth.sell_orders[a]))
                    if qty > 0: orders.append(Order(p, a, qty))
                
                # 如果买单价格高于我们的卖出价，主动卖出
                if b > sell_px and pos > -max_position:
                    qty = min(size_sell, max_position + pos, depth.buy_orders[b])
                    if qty > 0: orders.append(Order(p, b, -qty))
        
        # 常规做市订单
        if pos < max_position:
            orders.append(Order(p, buy_px, min(size_buy, max_position - pos)))
        if pos > -max_position:
            orders.append(Order(p, sell_px, -min(size_sell, max_position + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 计算技术指标
        self.calculate_technical_indicators(state)
        
        # 寻找方向性交易机会
        opportunities = self.find_directional_opportunities(state)
        
        # 执行方向性交易策略
        if opportunities:
            dir_orders = self.execute_directional_trading(state, opportunities)
            result.update(dir_orders)
        
        # 对未在方向性交易中交易的产品应用激进方向性交易策略
        for p in VOLCANIC_PRODUCTS:
            # 如果已经在方向性交易中交易了这个产品，跳过
            if p in result:
                continue
                
            if p in state.order_depths:
                depth = state.order_depths[p]
                pos = state.position.get(p, 0)
                result[p] = self.aggressive_directional_trading(p, depth, pos)
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
