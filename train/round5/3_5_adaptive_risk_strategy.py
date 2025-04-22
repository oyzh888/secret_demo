from typing import Dict, List, Tuple, Optional, Set
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict

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
    # 高风险产品
    "high_risk": {
        "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
        "PICNIC_BASKET1", "PICNIC_BASKET2"
    },
    # 中等风险产品
    "medium_risk": {
        "MAGNIFICENT_MACARONS", "VOLCANIC_ROCK_VOUCHER_10000", 
        "SQUID_INK", "DJEMBES", "JAMS", "VOLCANIC_ROCK_VOUCHER_10250"
    },
    # 低风险产品
    "low_risk": {
        "CROISSANTS", "KELP", "VOLCANIC_ROCK_VOUCHER_10500", "RAINFOREST_RESIN"
    }
}

# 参数设置
PARAM = {
    # 基础参数
    "base_spread": 2,
    "vol_factor": 1.0,
    "position_limit_pct": 0.7,  # 使用70%的仓位限制
    
    # 风险环境参数
    "vol_window": 20,           # 波动率窗口
    "vol_ratio_threshold": 1.5, # 波动率比率阈值
    "vol_percentile_high": 80,  # 高波动率百分位
    "vol_percentile_low": 20,   # 低波动率百分位
    
    # 风险层级参数
    "risk_tiers": {
        "high_risk": {
            "low_vol": {
                "spread_multiplier": 1.2,
                "size_limit": 0.4,
                "aggr_take": True
            },
            "normal_vol": {
                "spread_multiplier": 1.5,
                "size_limit": 0.3,
                "aggr_take": False
            },
            "high_vol": {
                "spread_multiplier": 2.0,
                "size_limit": 0.2,
                "aggr_take": False
            }
        },
        "medium_risk": {
            "low_vol": {
                "spread_multiplier": 1.0,
                "size_limit": 0.6,
                "aggr_take": True
            },
            "normal_vol": {
                "spread_multiplier": 1.2,
                "size_limit": 0.5,
                "aggr_take": True
            },
            "high_vol": {
                "spread_multiplier": 1.5,
                "size_limit": 0.4,
                "aggr_take": False
            }
        },
        "low_risk": {
            "low_vol": {
                "spread_multiplier": 0.8,
                "size_limit": 0.8,
                "aggr_take": True
            },
            "normal_vol": {
                "spread_multiplier": 1.0,
                "size_limit": 0.7,
                "aggr_take": True
            },
            "high_vol": {
                "spread_multiplier": 1.2,
                "size_limit": 0.6,
                "aggr_take": True
            }
        }
    },
    
    # 止损参数
    "stop_loss_pct": {
        "high_risk": 0.03,    # 高风险产品止损阈值 (3%)
        "medium_risk": 0.05,  # 中等风险产品止损阈值 (5%)
        "low_risk": 0.08      # 低风险产品止损阈值 (8%)
    },
    "max_drawdown": 500,      # 最大回撤
    "cooldown_period": 10,    # 冷却期
    
    # 趋势检测参数
    "trend_window": 15,
    "trend_threshold": 0.65
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.volatilities = defaultdict(list)  # 存储历史波动率
        self.position_history = defaultdict(list)  # 仓位历史
        self.vol_percentile = defaultdict(float)  # 波动率百分位
        self.vol_regime = defaultdict(str)  # 波动率环境：high, normal, low
        self.trends = defaultdict(int)  # 1=上升, -1=下降, 0=中性
        self.entry_prices = defaultdict(float)  # 入场价格
        self.stop_loss_active = defaultdict(bool)  # 止损激活状态
        self.cooldown_counter = defaultdict(int)  # 冷却计数器
        self.max_drawdown_values = defaultdict(float)  # 最大回撤值
        self.market_regime = "normal"  # 整体市场环境：high_vol, normal, low_vol
        
    def _vol(self, p: str, window: int = None) -> float:
        """计算产品波动率"""
        if window is None:
            window = PARAM["vol_window"]
            
        h = self.prices[p]
        if len(h) < window:
            return 2  # 默认中等波动率
        return statistics.stdev(h[-window:]) * PARAM["vol_factor"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def get_product_risk_tier(self, product: str) -> str:
        """获取产品风险层级"""
        for tier, products in PRODUCT_CATEGORIES.items():
            if product in products:
                return tier
        return "medium_risk"  # 默认为中等风险
    
    def update_volatility_metrics(self, state: TradingState):
        """更新波动率相关指标"""
        # 计算每个产品的波动率
        for p, depth in state.order_depths.items():
            mid = self._mid(depth)
            if mid is None:
                continue
                
            # 记录价格
            self.prices[p].append(mid)
            
            # 记录仓位
            self.position_history[p].append(state.position.get(p, 0))
            
            # 计算波动率
            vol = self._vol(p)
            self.volatilities[p].append(vol)
            
            # 保持波动率历史记录在合理范围内
            if len(self.volatilities[p]) > 100:
                self.volatilities[p].pop(0)
            
            # 计算波动率百分位
            if len(self.volatilities[p]) > 10:
                sorted_vols = sorted(self.volatilities[p])
                current_rank = sorted_vols.index(vol) / len(sorted_vols)
                self.vol_percentile[p] = current_rank * 100
                
                # 确定波动率环境
                if self.vol_percentile[p] > PARAM["vol_percentile_high"]:
                    self.vol_regime[p] = "high_vol"
                elif self.vol_percentile[p] < PARAM["vol_percentile_low"]:
                    self.vol_regime[p] = "low_vol"
                else:
                    self.vol_regime[p] = "normal_vol"
        
        # 确定整体市场环境
        high_vol_count = sum(1 for regime in self.vol_regime.values() if regime == "high_vol")
        low_vol_count = sum(1 for regime in self.vol_regime.values() if regime == "low_vol")
        
        if high_vol_count > len(self.vol_regime) * 0.4:  # 如果40%以上的产品处于高波动率环境
            self.market_regime = "high_vol"
        elif low_vol_count > len(self.vol_regime) * 0.4:  # 如果40%以上的产品处于低波动率环境
            self.market_regime = "low_vol"
        else:
            self.market_regime = "normal_vol"
    
    def detect_trend(self, product: str) -> int:
        """检测市场趋势"""
        prices = self.prices[product]
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
        
        # 获取产品风险层级
        risk_tier = self.get_product_risk_tier(product)
        
        # 计算价格变动百分比
        price_change_pct = (current_price - self.entry_prices[product]) / self.entry_prices[product]
        
        # 检查止损
        stop_loss_threshold = PARAM["stop_loss_pct"][risk_tier]
        if abs(price_change_pct) > stop_loss_threshold and price_change_pct < 0:
            if not self.stop_loss_active[product]:
                self.stop_loss_active[product] = True
                self.cooldown_counter[product] = PARAM["cooldown_period"]
                # 重置入场价格
                self.entry_prices[product] = 0
                return False  # 触发止损，不交易
        
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
    
    def adaptive_market_making(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """自适应做市策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 检查风险控制
        if not self.check_risk_controls(p, mid):
            return orders
        
        # 获取产品风险层级
        risk_tier = self.get_product_risk_tier(p)
        
        # 获取产品波动率环境
        vol_env = self.vol_regime.get(p, "normal_vol")
        
        # 获取相应的参数
        tier_params = PARAM["risk_tiers"][risk_tier][vol_env]
        
        # 检测趋势
        trend = self.detect_trend(p)
        self.trends[p] = trend
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差
        spread = int(PARAM["base_spread"] + vol * tier_params["spread_multiplier"])
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 根据趋势调整价格
        if trend == 1:  # 上升趋势
            buy_px += 1  # 更积极地买入
            sell_px += 1  # 更保守地卖出
        elif trend == -1:  # 下降趋势
            buy_px -= 1  # 更保守地买入
            sell_px -= 1  # 更积极地卖出
        
        # 根据整体市场环境调整交易规模
        market_adjustment = 1.0
        if self.market_regime == "high_vol":
            market_adjustment = 0.8  # 高波动率环境，减少交易规模
        elif self.market_regime == "low_vol":
            market_adjustment = 1.2  # 低波动率环境，增加交易规模
        
        # 计算交易规模
        max_position = int(LIMIT[p] * PARAM["position_limit_pct"] * tier_params["size_limit"] * market_adjustment)
        size = max(1, max_position // 4)  # 每次用1/4的允许仓位
        
        # 主动吃单
        if tier_params["aggr_take"]:
            b, a = best_bid_ask(depth)
            if a is not None and b is not None:
                # 如果卖单价格低于我们的买入价，主动买入
                if a < buy_px and pos < max_position:
                    qty = min(size, max_position - pos, abs(depth.sell_orders[a]))
                    if qty > 0: orders.append(Order(p, a, qty))
                
                # 如果买单价格高于我们的卖出价，主动卖出
                if b > sell_px and pos > -max_position:
                    qty = min(size, max_position + pos, depth.buy_orders[b])
                    if qty > 0: orders.append(Order(p, b, -qty))
        
        # 常规做市订单
        if pos < max_position:
            orders.append(Order(p, buy_px, min(size, max_position - pos)))
        if pos > -max_position:
            orders.append(Order(p, sell_px, -min(size, max_position + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 更新波动率相关指标
        self.update_volatility_metrics(state)
        
        # 对每个产品应用自适应做市策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.adaptive_market_making(p, depth, state.position.get(p, 0))
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
