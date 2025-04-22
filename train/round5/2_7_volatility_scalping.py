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

# 产品特性配置 - 波动性套利
PRODUCT_CONFIG = {
    # 高波动性产品 - 更激进的波动性套利
    "VOLCANIC_ROCK": {"bb_period": 15, "bb_std": 2.0, "min_vol": 0.015, "scalp_pct": 0.3, "mm_pct": 0.15},
    "VOLCANIC_ROCK_VOUCHER_9500": {"bb_period": 20, "bb_std": 1.8, "min_vol": 0.012, "scalp_pct": 0.25, "mm_pct": 0.12},
    "VOLCANIC_ROCK_VOUCHER_9750": {"bb_period": 20, "bb_std": 1.8, "min_vol": 0.012, "scalp_pct": 0.25, "mm_pct": 0.12},
    "VOLCANIC_ROCK_VOUCHER_10000": {"bb_period": 20, "bb_std": 1.8, "min_vol": 0.012, "scalp_pct": 0.25, "mm_pct": 0.12},
    "VOLCANIC_ROCK_VOUCHER_10250": {"bb_period": 20, "bb_std": 1.8, "min_vol": 0.012, "scalp_pct": 0.25, "mm_pct": 0.12},
    "VOLCANIC_ROCK_VOUCHER_10500": {"bb_period": 20, "bb_std": 1.8, "min_vol": 0.012, "scalp_pct": 0.25, "mm_pct": 0.12},
    
    # 中波动性产品
    "PICNIC_BASKET1": {"bb_period": 25, "bb_std": 1.5, "min_vol": 0.008, "scalp_pct": 0.2, "mm_pct": 0.1},
    "PICNIC_BASKET2": {"bb_period": 25, "bb_std": 1.5, "min_vol": 0.008, "scalp_pct": 0.2, "mm_pct": 0.1},
    "MAGNIFICENT_MACARONS": {"bb_period": 30, "bb_std": 1.5, "min_vol": 0.008, "scalp_pct": 0.2, "mm_pct": 0.1},
    
    # 低波动性产品/默认
    "DEFAULT": {"bb_period": 30, "bb_std": 1.2, "min_vol": 0.005, "scalp_pct": 0.15, "mm_pct": 0.08}
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
        # 价格变动历史
        self.price_changes = defaultdict(list)
        # 波动率历史
        self.volatility = defaultdict(float)
        # 布林带
        self.bollinger_bands = defaultdict(dict)
        # 订单历史
        self.orders_history = defaultdict(list)
        # 最近执行的交易
        self.last_trades = defaultdict(dict)
        # 上一次交易的时间戳
        self.last_trade_timestamp = defaultdict(int)
        # 止损价格
        self.stop_losses = defaultdict(float)
        # 获利价格
        self.take_profits = defaultdict(float)
        # 均值回归状态 (1=做多, -1=做空, 0=中性)
        self.mean_reversion_state = defaultdict(int)
        # 当前的市场强度 (0-1)
        self.market_momentum = defaultdict(float)
        # 市场整体波动率
        self.overall_volatility = 0.0
        # 适合套利的市场环境 (True/False)
        self.is_scalping_environment = False
        # 做市商模式活跃 (True/False)
        self.market_making_active = defaultdict(bool)
        # z-score历史
        self.z_scores = defaultdict(list)
        # 交易日志
        self.trade_log = []
        
    def update_price_history(self, product: str, depth: OrderDepth):
        """更新价格历史和计算基本指标"""
        # 获取中间价
        price = mid_price(depth)
        if price is None:
            return False
            
        # 更新价格历史
        if self.prices[product]:
            # 计算价格变化
            prev_price = self.prices[product][-1]
            price_change = (price - prev_price) / prev_price if prev_price != 0 else 0
            self.price_changes[product].append(price_change)
            
        self.prices[product].append(price)
        return True
    
    def calculate_bollinger_bands(self, product: str, config: dict):
        """计算布林带指标"""
        prices = self.prices[product]
        period = config["bb_period"]
        
        if len(prices) < period:
            return
            
        # 截取最近period个价格
        recent_prices = prices[-period:]
        
        # 计算移动平均线和标准差
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        # 计算布林带上下轨
        upper_band = sma + config["bb_std"] * std
        lower_band = sma - config["bb_std"] * std
        
        # 保存结果
        self.bollinger_bands[product] = {
            "sma": sma,
            "upper": upper_band,
            "lower": lower_band,
            "width": (upper_band - lower_band) / sma if sma != 0 else 0
        }
    
    def calculate_volatility(self, product: str, window: int = 20):
        """计算价格波动率"""
        changes = self.price_changes[product]
        if len(changes) < window:
            return 0.0
            
        # 计算波动率 (标准差)
        recent_changes = changes[-window:]
        vol = np.std(recent_changes) * math.sqrt(window)  # 年化调整
        
        self.volatility[product] = vol
        return vol
        
    def calculate_z_score(self, product: str):
        """计算价格偏离均值的Z-Score"""
        if product not in self.bollinger_bands or product not in self.prices or not self.prices[product]:
            return 0
            
        # 获取最新价格
        current_price = self.prices[product][-1]
        # 获取均线
        sma = self.bollinger_bands[product]["sma"]
        # 获取标准差
        std_dev = (self.bollinger_bands[product]["upper"] - sma) / PRODUCT_CONFIG[product].get("bb_std", 2.0)
        
        # 计算z-score
        if std_dev == 0:
            z_score = 0
        else:
            z_score = (current_price - sma) / std_dev
            
        # 保存z-score历史
        self.z_scores[product].append(z_score)
        
        return z_score
        
    def estimate_mean_reversion_probability(self, product: str, z_score: float):
        """估计均值回归的概率"""
        # 高z-score意味着价格远离均值，回归可能性更高
        if abs(z_score) > 2.5:
            return 0.8  # 80%概率
        elif abs(z_score) > 2.0:
            return 0.7  # 70%概率
        elif abs(z_score) > 1.5:
            return 0.6  # 60%概率
        elif abs(z_score) > 1.0:
            return 0.5  # 50%概率
        else:
            return 0.3  # 30%概率
            
    def update_market_environment(self):
        """更新整体市场环境"""
        # 计算整体波动率
        volatilities = [vol for vol in self.volatility.values() if vol > 0]
        
        if volatilities:
            # 使用波动率的加权平均
            self.overall_volatility = sum(volatilities) / len(volatilities)
            
            # 判断是否适合波动性套利
            self.is_scalping_environment = self.overall_volatility > 0.01  # 1%以上波动率
        else:
            self.overall_volatility = 0.0
            self.is_scalping_environment = False
            
    def set_stop_loss_take_profit(self, product: str, entry_price: float, direction: int, z_score: float):
        """设置止损和获利价位"""
        # 基于波动性和Z-score的动态设置
        volatility = max(0.005, self.volatility[product])  # 至少0.5%
        z_factor = min(1.5, abs(z_score) / 2)  # Z-score影响因子
        
        if direction > 0:  # 做多
            # 止损：低于进场价格一定百分比
            self.stop_losses[product] = entry_price * (1 - volatility * 3 * z_factor)
            # 获利：高于进场价格一定百分比
            self.take_profits[product] = entry_price * (1 + volatility * 4 * z_factor)
        else:  # 做空
            # 止损：高于进场价格一定百分比
            self.stop_losses[product] = entry_price * (1 + volatility * 3 * z_factor)
            # 获利：低于进场价格一定百分比
            self.take_profits[product] = entry_price * (1 - volatility * 4 * z_factor)
    
    def check_exit_conditions(self, product: str, current_price: float, position: int, z_score: float):
        """检查是否应该退出当前交易"""
        # 无持仓
        if position == 0:
            return False, 0
            
        # 检查止损
        if position > 0 and current_price <= self.stop_losses[product]:  # 多头止损
            return True, -position
        elif position < 0 and current_price >= self.stop_losses[product]:  # 空头止损
            return True, -position
            
        # 检查止盈
        if position > 0 and current_price >= self.take_profits[product]:  # 多头止盈
            return True, -position
        elif position < 0 and current_price <= self.take_profits[product]:  # 空头止盈
            return True, -position
            
        # 检查均值回归 - 当价格回归到均值附近时退出
        if abs(z_score) < 0.5:  # 接近均值
            if (position > 0 and z_score > 0) or (position < 0 and z_score < 0):
                return True, -position  # 完全平仓
                
        # 检查反向信号
        if (position > 0 and z_score < -1.0) or (position < 0 and z_score > 1.0):
            return True, -position  # 反向信号出现，平仓
            
        return False, 0
        
    def should_scalp(self, product: str, z_score: float, config: dict):
        """判断是否应该进行波动性套利"""
        # 检查波动性是否足够
        if self.volatility[product] < config["min_vol"]:
            return False, 0
            
        # 检查Z-score是否足够极端
        if abs(z_score) < 1.0:
            return False, 0
            
        # 计算回归概率
        reversion_prob = self.estimate_mean_reversion_probability(product, z_score)
        
        # 只有高概率机会才交易
        if reversion_prob < 0.6:
            return False, 0
            
        # 确定交易方向
        if z_score > 0:  # 价格高于均值
            direction = -1  # 做空，预期回落
        else:  # 价格低于均值
            direction = 1  # 做多，预期上涨
            
        return True, direction
        
    def calculate_position_size(self, product: str, direction: int, position: int, config: dict, z_score: float):
        """计算头寸规模"""
        limit = POSITION_LIMITS[product]
        
        # 基于Z-score的规模
        z_factor = min(1.0, abs(z_score) / 3)  # Z-score影响因子，最大1.0
        
        # 基于波动率的规模调整
        vol_factor = 1.0
        if self.volatility[product] > 0.02:  # 高波动
            vol_factor = 0.8  # 减少规模
        elif self.volatility[product] < 0.01:  # 低波动
            vol_factor = 1.2  # 增加规模
            
        # 基于市场环境的调整
        env_factor = 1.2 if self.is_scalping_environment else 0.8
        
        # 综合计算规模
        size_pct = config["scalp_pct"] * z_factor * vol_factor * env_factor
        
        # 确保规模在合理范围内
        size_pct = min(0.4, max(0.1, size_pct))
        
        # 计算实际订单数量
        base_size = max(1, int(limit * size_pct))
        
        # 考虑当前持仓
        if direction > 0:  # 做多
            # 如果已经有同向头寸，减少规模
            if position > 0:
                size = min(base_size, limit - position)
            else:
                size = base_size
        else:  # 做空
            # 如果已经有同向头寸，减少规模
            if position < 0:
                size = min(base_size, limit + position)
            else:
                size = base_size
                
        return size
        
    def execute_market_making(self, product: str, depth: OrderDepth, position: int, config: dict):
        """执行做市商策略，在波动性低时提供流动性"""
        # 获取买卖价
        bid, bid_q, ask, ask_q = best_bid_ask(depth)
        if bid is None or ask is None:
            return []
            
        # 计算动态点差
        spread = max(1, int((ask - bid) * 0.3))  # 使用当前点差的30%
        
        # 做市商规模
        mm_size = max(1, int(POSITION_LIMITS[product] * config["mm_pct"]))
        
        # 考虑当前持仓
        buy_size = mm_size
        sell_size = mm_size
        
        # 如果已经有大量头寸，减少相应方向的做市规模
        if position > POSITION_LIMITS[product] * 0.5:  # 多头超过限制的50%
            buy_size = max(1, mm_size // 2)  # 减少买入规模
        elif position < -POSITION_LIMITS[product] * 0.5:  # 空头超过限制的50%
            sell_size = max(1, mm_size // 2)  # 减少卖出规模
            
        # 设置做市价格
        buy_price = bid - spread
        sell_price = ask + spread
        
        orders = []
        
        # 添加买卖订单
        if abs(position) < POSITION_LIMITS[product] * 0.8:  # 仓位少于限制的80%
            # 买单
            if position + buy_size <= POSITION_LIMITS[product]:
                orders.append(Order(product, buy_price, buy_size))
                
            # 卖单
            if position - sell_size >= -POSITION_LIMITS[product]:
                orders.append(Order(product, sell_price, -sell_size))
        
        return orders
    
    def volatility_scalping_strategy(self, product: str, timestamp: int, depth: OrderDepth, position: int):
        """波动性套利策略主逻辑"""
        # 获取产品配置
        config = get_product_config(product)
        
        # 更新价格历史
        if not self.update_price_history(product, depth):
            return []
        
        # 计算技术指标
        self.calculate_bollinger_bands(product, config)
        vol = self.calculate_volatility(product)
        z_score = self.calculate_z_score(product)
        
        # 获取当前价格
        price = mid_price(depth)
        bid, _, ask, _ = best_bid_ask(depth)
        if price is None or bid is None or ask is None:
            return []
        
        # 检查是否应该退出现有交易
        should_exit, exit_size = self.check_exit_conditions(product, price, position, z_score)
        
        orders = []
        
        # 如果需要退出，执行退出订单
        if should_exit and exit_size != 0:
            if exit_size > 0:  # 买入平仓
                orders.append(Order(product, ask, exit_size))
            else:  # 卖出平仓
                orders.append(Order(product, bid, exit_size))
            
            # 重置止损和止盈
            self.stop_losses[product] = 0
            self.take_profits[product] = 0
            
            # 记录交易
            self.last_trade_timestamp[product] = timestamp
            
            return orders
        
        # 波动性套利逻辑
        should_scalp, direction = self.should_scalp(product, z_score, config)
        
        # 检查冷却时间
        cool_down = 20  # 冷却周期
        if timestamp - self.last_trade_timestamp.get(product, 0) < cool_down:
            should_scalp = False
        
        if should_scalp:
            # 计算交易规模
            size = self.calculate_position_size(product, direction, position, config, z_score)
            
            # 过滤太小的交易
            if size < 3:
                should_scalp = False
        
        # 执行套利交易
        if should_scalp:
            if direction > 0:  # 做多
                # 确保不超过限制
                buy_size = min(size, POSITION_LIMITS[product] - position)
                if buy_size > 0:
                    orders.append(Order(product, ask, buy_size))
                    # 设置止损和止盈
                    self.set_stop_loss_take_profit(product, ask, direction, z_score)
            else:  # 做空
                # 确保不超过限制
                sell_size = min(size, POSITION_LIMITS[product] + position)
                if sell_size > 0:
                    orders.append(Order(product, bid, -sell_size))
                    # 设置止损和止盈
                    self.set_stop_loss_take_profit(product, bid, direction, z_score)
            
            # 记录交易
            if orders:
                self.last_trade_timestamp[product] = timestamp
                self.market_making_active[product] = False  # 禁用做市商模式
                return orders
        
        # 如果没有套利机会，考虑做市商策略
        if not should_scalp and abs(z_score) < 1.0 and vol < config["min_vol"] * 1.5:
            self.market_making_active[product] = True
            return self.execute_market_making(product, depth, position, config)
            
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result = {}
        
        # 更新市场环境
        self.update_market_environment()
        
        # 处理每个产品
        for product, depth in state.order_depths.items():
            # 跳过没有限制的产品
            if product not in POSITION_LIMITS:
                continue
                
            # 获取当前仓位
            position = state.position.get(product, 0)
            
            # 应用波动性套利策略
            orders = self.volatility_scalping_strategy(
                product, state.timestamp, depth, position
            )
            
            if orders:
                result[product] = orders
        
        return result, 0, state.traderData 