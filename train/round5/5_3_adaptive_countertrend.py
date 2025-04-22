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

# 产品分类 - 初始分类，会动态更新
INITIAL_PRODUCT_TIERS = {
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
    # 自适应参数
    "vol_window": 30,           # 波动率计算窗口
    "vol_update_freq": 10,      # 波动率更新频率
    "vol_high_threshold": 3.0,  # 高波动率阈值
    "vol_low_threshold": 1.0,   # 低波动率阈值
    "performance_window": 20,   # 表现评估窗口
    "success_threshold": 0.6,   # 成功率阈值
    
    # 趋势和反转参数
    "trend_window": 30,         # 趋势检测窗口
    "trend_threshold": 0.7,     # 趋势检测阈值
    "reversal_window": 5,       # 反转检测窗口
    "reversal_threshold": 0.8,  # 反转检测阈值
    
    # 交易参数
    "position_limit_pct": {     # 不同波动率环境的仓位限制
        "high_vol": 0.5,        # 高波动率环境
        "normal_vol": 0.7,      # 正常波动率环境
        "low_vol": 0.9          # 低波动率环境
    },
    "mm_size_frac": {           # 不同波动率环境的做市规模
        "high_vol": 0.1,
        "normal_vol": 0.15,
        "low_vol": 0.2
    },
    "counter_size_frac": {      # 不同波动率环境的反趋势交易规模
        "high_vol": 0.15,
        "normal_vol": 0.25,
        "low_vol": 0.35
    },
    "reversal_cooldown": 8,     # 反转交易冷却期
    
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
    }
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.returns = defaultdict(list)  # 价格回报率
        self.volatilities = defaultdict(list)  # 历史波动率
        self.position_history = defaultdict(list)  # 仓位历史
        self.rsi_values = defaultdict(float)  # RSI值
        self.last_counter_trade = defaultdict(int)  # 上次反向交易的时间戳
        self.trade_count = defaultdict(int)  # 交易计数
        self.trade_success = defaultdict(list)  # 交易成功记录
        self.entry_prices = defaultdict(float)  # 入场价格
        self.max_position_values = defaultdict(float)  # 最大仓位价值
        self.product_tiers = {p: tier for tier, products in INITIAL_PRODUCT_TIERS.items() for p in products}
        self.vol_regime = defaultdict(str)  # 波动率环境：high_vol, normal_vol, low_vol
        self.update_counter = 0  # 更新计数器
        self.strategy_performance = defaultdict(dict)  # 策略表现
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < PARAM["vol_window"]: return 1
        return statistics.stdev(h[-PARAM["vol_window"]:]) * PARAM["vol_scale"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def update_volatility_regime(self):
        """更新波动率环境"""
        # 增加计数器
        self.update_counter += 1
        
        # 只在特定频率更新
        if self.update_counter % PARAM["vol_update_freq"] != 0:
            return
            
        # 计算每个产品的波动率
        for p, prices in self.prices.items():
            if len(prices) < PARAM["vol_window"]:
                continue
                
            vol = self._vol(p)
            self.volatilities[p].append(vol)
            
            # 保持波动率历史记录在合理范围内
            if len(self.volatilities[p]) > 100:
                self.volatilities[p].pop(0)
            
            # 确定波动率环境
            if vol > PARAM["vol_high_threshold"]:
                self.vol_regime[p] = "high_vol"
            elif vol < PARAM["vol_low_threshold"]:
                self.vol_regime[p] = "low_vol"
            else:
                self.vol_regime[p] = "normal_vol"
    
    def update_strategy_performance(self):
        """更新策略表现"""
        # 只在特定频率更新
        if self.update_counter % PARAM["vol_update_freq"] != 0:
            return
            
        # 计算每个产品的策略表现
        for p, successes in self.trade_success.items():
            if len(successes) < PARAM["performance_window"]:
                continue
                
            # 只使用最近的交易记录
            recent_successes = successes[-PARAM["performance_window"]:]
            
            # 计算成功率
            success_rate = sum(recent_successes) / len(recent_successes)
            
            # 更新策略表现
            self.strategy_performance[p] = {
                "success_rate": success_rate,
                "sample_size": len(recent_successes)
            }
            
            # 根据表现动态调整产品分类
            if success_rate > PARAM["success_threshold"]:
                # 如果反趋势策略表现良好，将产品移至高波动性类别
                self.product_tiers[p] = "high_vol"
            elif success_rate < 0.4:
                # 如果反趋势策略表现不佳，将产品移至低波动性类别
                self.product_tiers[p] = "low_vol"
            else:
                # 否则，保持在中等波动性类别
                self.product_tiers[p] = "medium_vol"
    
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
                # 记录交易失败
                self.trade_success[p].append(0)
                
                # 重置入场价格
                self.entry_prices[p] = 0
                
                return False  # 触发止损，不允许继续同方向交易
            
            # 检查最大回撤
            drawdown = self.max_position_values[p] - current_value
            if drawdown > PARAM["max_drawdown"]:
                # 记录交易失败
                self.trade_success[p].append(0)
                
                # 重置入场价格和最大回撤值
                self.entry_prices[p] = 0
                self.max_position_values[p] = 0
                
                return False  # 触发最大回撤，不允许继续同方向交易
        
        return True  # 通过风险控制，允许交易
    
    def record_trade_result(self, p: str, pos: int, current_price: float):
        """记录交易结果"""
        # 如果没有入场价格，无法记录结果
        if self.entry_prices[p] == 0 or pos == 0:
            return
            
        # 计算价格变动百分比
        price_change_pct = (current_price - self.entry_prices[p]) / self.entry_prices[p]
        
        # 判断交易是否成功
        if (pos > 0 and price_change_pct > 0) or (pos < 0 and price_change_pct < 0):
            # 记录交易成功
            self.trade_success[p].append(1)
        else:
            # 记录交易失败
            self.trade_success[p].append(0)
        
        # 保持交易记录在合理范围内
        if len(self.trade_success[p]) > 100:
            self.trade_success[p].pop(0)
    
    def adaptive_countertrend_strategy(self, p: str, depth: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        """自适应反趋势交易策略"""
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
        
        # 获取产品分类
        tier = self.product_tiers.get(p, "medium_vol")
        
        # 反趋势交易逻辑
        if tier in ["high_vol", "medium_vol"]:  # 只对高波动性和中等波动性产品应用反趋势策略
            # 检查是否在冷却期
            cooldown_active = timestamp - self.last_counter_trade.get(p, 0) < PARAM["reversal_cooldown"]
            
            if not cooldown_active and risk_ok:
                # 获取RSI超买超卖阈值
                rsi_overbought = PARAM["rsi_overbought"][vol_env]
                rsi_oversold = PARAM["rsi_oversold"][vol_env]
                
                # 获取价格激进程度
                price_aggression = PARAM["price_aggression"][vol_env]
                
                # 检查是否有强烈趋势 + RSI超买超卖
                if trend == 1 and self.rsi_values[p] > rsi_overbought:
                    # 强烈上升趋势 + RSI超买，反向做空
                    sell_px = int(mid - price_aggression)  # 降低卖出价格以确保成交
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    self.entry_prices[p] = sell_px
                    self.trade_count[p] += 1
                    return orders  # 只做反向交易，不做常规做市
                    
                elif trend == -1 and self.rsi_values[p] < rsi_oversold:
                    # 强烈下降趋势 + RSI超卖，反向做多
                    buy_px = int(mid + price_aggression)  # 提高买入价格以确保成交
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    self.entry_prices[p] = buy_px
                    self.trade_count[p] += 1
                    return orders  # 只做反向交易，不做常规做市
                    
                # 检查是否有明确的反转信号
                elif reversal != 0:
                    if reversal == 1:  # 下降趋势反转为上升
                        # 积极买入
                        buy_px = int(mid + price_aggression)
                        orders.append(Order(p, buy_px, counter_size))
                        self.last_counter_trade[p] = timestamp
                        self.entry_prices[p] = buy_px
                        self.trade_count[p] += 1
                        return orders
                        
                    elif reversal == -1:  # 上升趋势反转为下降
                        # 积极卖出
                        sell_px = int(mid - price_aggression)
                        orders.append(Order(p, sell_px, -counter_size))
                        self.last_counter_trade[p] = timestamp
                        self.entry_prices[p] = sell_px
                        self.trade_count[p] += 1
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
        
        # 记录交易结果
        self.record_trade_result(p, pos, mid)
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 更新波动率环境
        self.update_volatility_regime()
        
        # 更新策略表现
        self.update_strategy_performance()
        
        # 对每个产品应用自适应反趋势策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.adaptive_countertrend_strategy(
                    p, 
                    depth, 
                    state.position.get(p, 0),
                    state.timestamp
                )
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
