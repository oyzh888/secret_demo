from typing import Dict, List, Tuple, Optional
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np

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
    # 高波动性产品 - 积极追踪趋势
    "VOLCANIC_ROCK": {"breakout_periods": [5, 20], "volatility_mult": 1.2, "lookback": 30, "max_size": 0.4},
    "VOLCANIC_ROCK_VOUCHER_9500": {"breakout_periods": [7, 25], "volatility_mult": 1.1, "lookback": 40, "max_size": 0.35},
    "VOLCANIC_ROCK_VOUCHER_9750": {"breakout_periods": [7, 25], "volatility_mult": 1.1, "lookback": 40, "max_size": 0.35},
    "VOLCANIC_ROCK_VOUCHER_10000": {"breakout_periods": [7, 25], "volatility_mult": 1.1, "lookback": 40, "max_size": 0.35},
    "VOLCANIC_ROCK_VOUCHER_10250": {"breakout_periods": [7, 25], "volatility_mult": 1.1, "lookback": 40, "max_size": 0.35},
    "VOLCANIC_ROCK_VOUCHER_10500": {"breakout_periods": [7, 25], "volatility_mult": 1.1, "lookback": 40, "max_size": 0.35},
    
    # 中等波动性产品
    "PICNIC_BASKET1": {"breakout_periods": [10, 30], "volatility_mult": 1.0, "lookback": 50, "max_size": 0.35},
    "PICNIC_BASKET2": {"breakout_periods": [10, 30], "volatility_mult": 1.0, "lookback": 50, "max_size": 0.35},
    
    # 低波动性产品 - 更谨慎的趋势跟踪
    "DEFAULT": {"breakout_periods": [10, 40], "volatility_mult": 0.9, "lookback": 60, "max_size": 0.3}
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
        # 交易量历史
        self.volumes = defaultdict(list)
        # 最高价历史
        self.highs = defaultdict(list)
        # 最低价历史
        self.lows = defaultdict(list)
        # 移动平均线
        self.sma = defaultdict(dict)
        # 移动平均线差值
        self.sma_diff = defaultdict(dict)
        # 成交量加权平均价格 (VWAP)
        self.vwap = defaultdict(float)
        # 波动率历史
        self.volatility = defaultdict(float)
        # 上一次交易的时间戳
        self.last_trade_timestamp = defaultdict(int)
        # 关键价格水平 (支撑/阻力)
        self.key_levels = defaultdict(list)
        # 动态阈值
        self.breakout_thresholds = defaultdict(float)
        # 突破确认计数
        self.breakout_confirmations = defaultdict(int)
        # 突破方向 (1=上行突破, -1=下行突破, 0=无突破)
        self.breakout_direction = defaultdict(int)
        # 趋势跟踪状态 (1=做多, -1=做空, 0=中性)
        self.trend_state = defaultdict(int)
        # 市场整体趋势
        self.market_trend = 0
        # 订单标记
        self.order_tags = {}
        # 止损价格
        self.stop_losses = defaultdict(float)
        # 获利价格
        self.take_profits = defaultdict(float)
        # 整体市场波动性
        self.market_volatility = 0.0
        
    def update_price_history(self, product: str, depth: OrderDepth):
        """更新价格历史数据"""
        # 获取中间价
        price = mid_price(depth)
        if price is None:
            return False
            
        # 获取最新最高/最低价
        bid, _, ask, _ = best_bid_ask(depth)
        
        # 更新价格历史
        self.prices[product].append(price)
        
        # 估计交易量
        volume = 0
        if depth.buy_orders:
            volume += sum(abs(q) for q in depth.buy_orders.values())
        if depth.sell_orders:
            volume += sum(abs(q) for q in depth.sell_orders.values())
        self.volumes[product].append(volume)
        
        # 更新最高最低价
        if bid is not None and ask is not None:
            self.highs[product].append(ask)
            self.lows[product].append(bid)
        
        return True
        
    def calculate_indicators(self, product: str, config: dict):
        """计算各种技术指标"""
        prices = self.prices[product]
        if len(prices) < 5:  # 至少需要5个数据点
            return
            
        # 计算简单移动平均线
        for period in config["breakout_periods"]:
            if len(prices) >= period:
                self.sma[product][period] = np.mean(prices[-period:])
        
        # 计算移动平均线差值
        if len(config["breakout_periods"]) >= 2:
            fast = config["breakout_periods"][0]
            slow = config["breakout_periods"][1]
            if fast in self.sma[product] and slow in self.sma[product]:
                self.sma_diff[product]["fast_slow"] = self.sma[product][fast] - self.sma[product][slow]
        
        # 计算成交量加权平均价格 (VWAP)
        if len(prices) == len(self.volumes[product]) and len(prices) > 0:
            prices_array = np.array(prices[-min(len(prices), 30):])
            volumes_array = np.array(self.volumes[product][-min(len(self.volumes[product]), 30):])
            if np.sum(volumes_array) > 0:
                self.vwap[product] = np.sum(prices_array * volumes_array) / np.sum(volumes_array)
            else:
                self.vwap[product] = np.mean(prices_array) if len(prices_array) > 0 else 0
        
        # 计算波动率 (标准差/均价的百分比)
        if len(prices) >= 20:
            recent_prices = prices[-20:]
            self.volatility[product] = statistics.stdev(recent_prices) / statistics.mean(recent_prices)
        
        # 识别关键价格水平 (支撑/阻力)
        self.identify_key_levels(product, config)
        
        # 计算动态突破阈值
        self.calculate_breakout_threshold(product, config)
        
    def identify_key_levels(self, product: str, config: dict):
        """识别关键价格水平"""
        # 使用价格历史识别支撑和阻力位
        highs = self.highs[product]
        lows = self.lows[product]
        lookback = min(config["lookback"], len(highs), len(lows))
        
        if lookback < 10:  # 不够的数据点
            return
            
        # 用最近的高点和低点识别关键水平
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # 简化版聚类查找价格水平
        # 查找价格密集区
        hist_high, bin_edges_high = np.histogram(recent_highs, bins=10)
        hist_low, bin_edges_low = np.histogram(recent_lows, bins=10)
        
        # 找出频率最高的高点和低点区域
        high_clusters = [(hist_high[i], (bin_edges_high[i], bin_edges_high[i+1])) for i in range(len(hist_high))]
        low_clusters = [(hist_low[i], (bin_edges_low[i], bin_edges_low[i+1])) for i in range(len(hist_low))]
        
        # 按频率排序
        high_clusters.sort(reverse=True)
        low_clusters.sort(reverse=True)
        
        # 选择前3个最强的阻力位和支撑位
        resistance_levels = [np.mean(hc[1]) for hc in high_clusters[:3] if hc[0] > lookback / 10]
        support_levels = [np.mean(lc[1]) for lc in low_clusters[:3] if lc[0] > lookback / 10]
        
        # 更新关键水平
        self.key_levels[product] = support_levels + resistance_levels
        
    def calculate_breakout_threshold(self, product: str, config: dict):
        """计算动态突破阈值"""
        # 基于波动率的突破阈值
        if self.volatility[product] > 0:
            # 高波动率需要更大的阈值
            self.breakout_thresholds[product] = max(
                0.005,  # 最小阈值
                self.volatility[product] * config["volatility_mult"]
            )
        else:
            # 默认阈值
            self.breakout_thresholds[product] = 0.01
            
    def detect_breakout(self, product: str, price: float, config: dict):
        """检测价格突破"""
        # 获取突破阈值
        threshold = self.breakout_thresholds[product]
        
        # 根据关键水平检测突破
        for level in self.key_levels[product]:
            # 忽略远离当前价格的水平
            if abs(level - price) / price > 0.1:  # 超过10%差距
                continue
                
            # 检测上行突破
            if 0 < (price - level) / level < threshold * 2:
                # 刚刚突破支撑/阻力位
                return 1
                
            # 检测下行突破
            elif 0 < (level - price) / level < threshold * 2:
                # 刚刚跌破支撑/阻力位
                return -1
        
        # 使用移动平均线交叉检测趋势变化
        if "fast_slow" in self.sma_diff[product]:
            sma_diff = self.sma_diff[product]["fast_slow"]
            # 判断快线是否穿过慢线
            if abs(sma_diff) < price * 0.001:  # 非常接近交叉
                if sma_diff > 0:  # 快线在慢线之上
                    return 1
                else:  # 快线在慢线之下
                    return -1
                    
        # 价格突破检测
        # 检查是否突破最近N个周期的范围
        lookback = min(config["breakout_periods"][1], len(self.prices[product])-1)
        if lookback > 5:
            recent_high = max(self.prices[product][-lookback:-1])
            recent_low = min(self.prices[product][-lookback:-1])
            
            # 价格突破上方区间
            if price > recent_high and (price - recent_high) / recent_high > threshold:
                return 1
            
            # 价格突破下方区间
            if price < recent_low and (recent_low - price) / recent_low > threshold:
                return -1
        
        # 没有检测到突破
        return 0
        
    def confirm_breakout(self, product: str, direction: int, price: float):
        """确认价格突破的有效性"""
        # 如果方向改变，重置确认计数
        if direction != self.breakout_direction[product]:
            self.breakout_confirmations[product] = 0
            self.breakout_direction[product] = direction
            return False
        
        # 增加确认计数
        if direction != 0:
            self.breakout_confirmations[product] += 1
        
        # 需要连续几个周期确认突破
        required_confirmations = 2
        
        # 检查VWAP确认
        vwap_confirms = False
        if self.vwap[product] > 0:
            if direction == 1 and price > self.vwap[product]:  # 上行突破，价格高于VWAP
                vwap_confirms = True
            elif direction == -1 and price < self.vwap[product]:  # 下行突破，价格低于VWAP
                vwap_confirms = True
        
        # 判断是否确认
        return (self.breakout_confirmations[product] >= required_confirmations and vwap_confirms)
    
    def update_trend_state(self, product: str, confirmed_breakout: bool, direction: int):
        """更新趋势状态"""
        if confirmed_breakout:
            # 确认突破后更新趋势状态
            self.trend_state[product] = direction
            
        # 如果有足够数据，计算短期趋势
        prices = self.prices[product]
        if len(prices) >= 5:
            # 使用最后5个价格点计算短期趋势
            short_trend = 1 if prices[-1] > prices[-5] else -1 if prices[-1] < prices[-5] else 0
            
            # 如果短期趋势与当前趋势状态相反，可能是趋势减弱的信号
            if short_trend != 0 and short_trend != self.trend_state[product]:
                # 将趋势状态减弱（但不立即反转）
                if abs(self.trend_state[product]) == 2:  # 强趋势
                    self.trend_state[product] = self.trend_state[product] // 2  # 减弱到普通趋势
            
            # 如果短期和主要趋势一致，可能是趋势增强的信号
            elif short_trend != 0 and short_trend == self.trend_state[product] and abs(self.trend_state[product]) == 1:
                # 将趋势状态增强
                self.trend_state[product] = self.trend_state[product] * 2  # 增强到强趋势
    
    def calculate_position_size(self, product: str, direction: int, position: int, config: dict):
        """计算头寸规模"""
        limit = POSITION_LIMITS[product]
        max_size_pct = config["max_size"]
        
        # 基于趋势强度的规模
        trend_strength = abs(self.trend_state[product]) / 2  # 范围 0.5-1.0
        
        # 基于波动率的风险调整
        vol_adj = 1.0
        if self.volatility[product] > 0.03:  # 高波动
            vol_adj = 0.7  # 减少规模
        elif self.volatility[product] < 0.01:  # 低波动
            vol_adj = 1.2  # 增加规模
            
        # 基于市场整体趋势的调整
        market_adj = 1.0
        if self.market_trend != 0 and self.market_trend == direction:
            market_adj = 1.2  # 与市场趋势一致，增加规模
        elif self.market_trend != 0 and self.market_trend != direction:
            market_adj = 0.8  # 与市场趋势相反，减少规模
            
        # 综合计算规模
        size_pct = max_size_pct * trend_strength * vol_adj * market_adj
        
        # 确保规模在合理范围内
        size_pct = min(0.5, max(0.1, size_pct))
        
        # 计算实际订单数量
        base_size = max(1, int(limit * size_pct))
        
        # 考虑当前持仓
        if direction == 1:  # 做多
            # 已有多头仓位时减少买入量，已有空头仓位时增加买入量
            if position > 0:
                # 减少规模以避免过度集中
                size = min(base_size, limit - position)
            else:
                # 增加规模以平仓并开新仓
                size = min(base_size + abs(position), limit)
        else:  # 做空
            # 已有空头仓位时减少卖出量，已有多头仓位时增加卖出量
            if position < 0:
                # 减少规模以避免过度集中
                size = min(base_size, limit + position)
            else:
                # 增加规模以平仓并开新仓
                size = min(base_size + position, limit)
        
        return size
            
    def set_stop_loss_take_profit(self, product: str, price: float, direction: int):
        """设置止损和止盈价格"""
        volatility = max(0.01, self.volatility[product])
        
        if direction == 1:  # 做多
            # 止损：低于进场价一定百分比
            self.stop_losses[product] = price * (1 - volatility * 2)
            # 止盈：高于进场价一定百分比
            self.take_profits[product] = price * (1 + volatility * 3)
        else:  # 做空
            # 止损：高于进场价一定百分比
            self.stop_losses[product] = price * (1 + volatility * 2)
            # 止盈：低于进场价一定百分比
            self.take_profits[product] = price * (1 - volatility * 3)
    
    def check_exit_conditions(self, product: str, price: float, position: int):
        """检查是否应该退出当前交易"""
        # 没有持仓
        if position == 0:
            return False, 0
            
        # 检查止损
        if position > 0 and price <= self.stop_losses[product]:  # 多头止损
            return True, -position
        elif position < 0 and price >= self.stop_losses[product]:  # 空头止损
            return True, -position
            
        # 检查止盈
        if position > 0 and price >= self.take_profits[product]:  # 多头止盈
            return True, -position
        elif position < 0 and price <= self.take_profits[product]:  # 空头止盈
            return True, -position
            
        # 检查趋势反转
        if position > 0 and self.trend_state[product] < 0:  # 多头，但趋势转为空头
            return True, -position
        elif position < 0 and self.trend_state[product] > 0:  # 空头，但趋势转为多头
            return True, -position
            
        return False, 0
        
    def update_market_trend(self):
        """更新整体市场趋势"""
        # 统计各个产品的趋势状态
        up_trends = 0
        down_trends = 0
        
        for product, state in self.trend_state.items():
            if state > 0:
                up_trends += 1
            elif state < 0:
                down_trends += 1
                
        # 确定市场整体趋势
        if up_trends > down_trends * 2:  # 多数产品上涨
            self.market_trend = 1
        elif down_trends > up_trends * 2:  # 多数产品下跌
            self.market_trend = -1
        else:  # 趋势不明确
            self.market_trend = 0
            
    def trend_breakout_strategy(self, product: str, timestamp: int, depth: OrderDepth, position: int):
        """趋势突破策略"""
        # 获取产品配置
        config = get_product_config(product)
        
        # 更新价格历史
        if not self.update_price_history(product, depth):
            return []
            
        # 获取当前价格
        price = mid_price(depth)
        if price is None:
            return []
            
        # 计算技术指标
        self.calculate_indicators(product, config)
        
        # 检测突破
        breakout_direction = self.detect_breakout(product, price, config)
        
        # 确认突破
        confirmed_breakout = self.confirm_breakout(product, breakout_direction, price)
        
        # 更新趋势状态
        self.update_trend_state(product, confirmed_breakout, breakout_direction)
        
        # 获取最新价格
        bid, bid_q, ask, ask_q = best_bid_ask(depth)
        if bid is None or ask is None:
            return []
            
        # 检查是否应该退出当前交易
        should_exit, exit_size = self.check_exit_conditions(product, price, position)
        
        orders = []
        
        # 执行退出逻辑
        if should_exit and exit_size != 0:
            if exit_size > 0:  # 买入平仓
                orders.append(Order(product, ask, exit_size))
                self.order_tags[(product, timestamp)] = "EXIT_LONG"
            else:  # 卖出平仓
                orders.append(Order(product, bid, exit_size))
                self.order_tags[(product, timestamp)] = "EXIT_SHORT"
            
            # 重置止损和止盈
            self.stop_losses[product] = 0
            self.take_profits[product] = 0
            
            return orders
            
        # 检查冷却期
        cool_down = 30  # 冷却周期
        if timestamp - self.last_trade_timestamp.get(product, 0) < cool_down:
            return []
            
        # 交易逻辑
        if self.trend_state[product] != 0:  # 有明确趋势
            direction = self.trend_state[product]
            position_size = self.calculate_position_size(product, direction, position, config)
            
            # 排除过小的交易
            if position_size < max(3, POSITION_LIMITS[product] * 0.05):
                return []
                
            if direction > 0:  # 上升趋势
                # 如果当前有空头头寸，先平仓
                if position < 0:
                    orders.append(Order(product, ask, abs(position)))
                    self.order_tags[(product, timestamp)] = "CLOSE_SHORT"
                
                # 然后开多头
                remaining_size = min(position_size, POSITION_LIMITS[product] - position)
                if remaining_size > 0:
                    orders.append(Order(product, ask, remaining_size))
                    self.order_tags[(product, timestamp)] = "OPEN_LONG"
                    
                    # 设置止损和止盈
                    self.set_stop_loss_take_profit(product, ask, 1)
                    
            elif direction < 0:  # 下降趋势
                # 如果当前有多头头寸，先平仓
                if position > 0:
                    orders.append(Order(product, bid, -position))
                    self.order_tags[(product, timestamp)] = "CLOSE_LONG"
                
                # 然后开空头
                remaining_size = min(position_size, POSITION_LIMITS[product] + position)
                if remaining_size > 0:
                    orders.append(Order(product, bid, -remaining_size))
                    self.order_tags[(product, timestamp)] = "OPEN_SHORT"
                    
                    # 设置止损和止盈
                    self.set_stop_loss_take_profit(product, bid, -1)
        
        # 更新交易时间戳
        if orders:
            self.last_trade_timestamp[product] = timestamp
            
        return orders
            
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result = {}
        
        # 更新整体市场趋势
        self.update_market_trend()
        
        # 处理每个产品
        for product, depth in state.order_depths.items():
            # 跳过没有限制的产品
            if product not in POSITION_LIMITS:
                continue
                
            # 获取当前仓位
            position = state.position.get(product, 0)
            
            # 应用趋势突破策略
            orders = self.trend_breakout_strategy(
                product, state.timestamp, depth, position
            )
            
            if orders:
                result[product] = orders
        
        return result, 0, state.traderData 