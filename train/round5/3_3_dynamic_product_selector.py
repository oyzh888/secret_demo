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

# 初始产品分类 - 会动态更新
INITIAL_PRODUCT_TIERS = {
    # 高表现产品 - 积极交易
    "high": {
        "RAINFOREST_RESIN", "CROISSANTS", "MAGNIFICENT_MACARONS"
    },
    # 中等表现产品 - 适度交易
    "medium": {
        "KELP", "PICNIC_BASKET2", "JAMS", "VOLCANIC_ROCK_VOUCHER_10500"
    },
    # 低表现产品 - 保守交易
    "low": {
        "SQUID_INK", "PICNIC_BASKET1", "DJEMBES"
    },
    # 避开产品 - 不交易
    "avoid": {
        "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
        "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250"
    }
}

# 参数设置
PARAM = {
    "tight_spread": 2,        # 基础价差
    "k_vol": 1.0,             # 波动率因子
    "performance_window": 20, # 表现评估窗口
    "update_frequency": 5,    # 产品分类更新频率
    "high_threshold": 50,     # 高表现阈值
    "low_threshold": -50,     # 低表现阈值
    "avoid_threshold": -200,  # 避开阈值
    "vol_window": 15,         # 波动率窗口
    
    # 产品层级参数
    "tiers": {
        "high": {
            "spread_multiplier": 0.8,  # 更窄的价差
            "size_limit": 0.8,         # 更大的规模
            "aggr_take": True          # 主动吃单
        },
        "medium": {
            "spread_multiplier": 1.2,
            "size_limit": 0.5,
            "aggr_take": True
        },
        "low": {
            "spread_multiplier": 1.5,  # 更宽的价差
            "size_limit": 0.3,         # 更小的规模
            "aggr_take": False         # 不主动吃单
        },
        "avoid": {
            "spread_multiplier": 2.0,  # 非常宽的价差
            "size_limit": 0.1,         # 极小的规模
            "aggr_take": False         # 不主动吃单
        }
    }
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.position_history = defaultdict(list)
        self.product_performance = defaultdict(list)  # 产品表现历史
        self.product_tiers = {p: tier for tier, products in INITIAL_PRODUCT_TIERS.items() for p in products}
        self.update_counter = 0  # 更新计数器
        self.trends = defaultdict(int)  # 1=上升, -1=下降, 0=中性
        self.last_mid_price = {}
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < PARAM["vol_window"]: return 1
        return statistics.stdev(h[-PARAM["vol_window"]:]) * PARAM["k_vol"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def update_product_performance(self, state: TradingState):
        """更新产品表现评估"""
        for p, pos in state.position.items():
            if p not in self.prices or len(self.prices[p]) < 2:
                continue
                
            # 计算价格变动
            current_price = self.prices[p][-1]
            previous_price = self.prices[p][-2]
            price_change = (current_price - previous_price) / previous_price
            
            # 计算仓位价值变化
            position_value_change = pos * price_change * 100  # 缩放以使数值更明显
            
            # 记录表现
            self.product_performance[p].append(position_value_change)
            
            # 保持表现历史记录在合理范围内
            if len(self.product_performance[p]) > PARAM["performance_window"]:
                self.product_performance[p].pop(0)
    
    def update_product_tiers(self):
        """更新产品分类"""
        # 增加计数器
        self.update_counter += 1
        
        # 只在特定频率更新
        if self.update_counter % PARAM["update_frequency"] != 0:
            return
            
        # 计算每个产品的表现得分
        product_scores = {}
        for p, performance in self.product_performance.items():
            if len(performance) < PARAM["performance_window"] // 2:
                continue  # 数据不足
                
            # 计算表现得分 (最近的表现权重更高)
            weighted_performance = [perf * (i + 1) for i, perf in enumerate(performance)]
            score = sum(weighted_performance) / sum(range(1, len(performance) + 1))
            product_scores[p] = score
        
        # 根据得分更新产品分类
        for p, score in product_scores.items():
            if score > PARAM["high_threshold"]:
                self.product_tiers[p] = "high"
            elif score < PARAM["avoid_threshold"]:
                self.product_tiers[p] = "avoid"
            elif score < PARAM["low_threshold"]:
                self.product_tiers[p] = "low"
            else:
                self.product_tiers[p] = "medium"
    
    def detect_trend(self, product: str) -> int:
        """检测市场趋势"""
        prices = self.prices[product]
        if len(prices) < 10:
            return 0  # 数据不足
            
        recent_prices = prices[-10:]  
        up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
        
        up_ratio = up_moves / (len(recent_prices) - 1)
        down_ratio = down_moves / (len(recent_prices) - 1)
        
        if up_ratio > 0.7:
            return 1  # 上升趋势
        elif down_ratio > 0.7:
            return -1  # 下降趋势
        return 0  # 中性
    
    def adaptive_market_making(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """自适应做市策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 记录价格历史
        self.prices[p].append(mid)
        
        # 记录仓位历史
        self.position_history[p].append(pos)
        
        # 检测趋势
        trend = self.detect_trend(p)
        self.trends[p] = trend
        
        # 获取产品层级
        tier = self.product_tiers.get(p, "medium")  # 默认为中等
        tier_params = PARAM["tiers"][tier]
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差
        spread = int(PARAM["tight_spread"] + vol * tier_params["spread_multiplier"])
        
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
        
        # 计算交易规模
        max_position = int(LIMIT[p] * tier_params["size_limit"])
        size = max(1, max_position // 4)  # 每次用1/4的允许仓位
        
        # 如果是"avoid"层级，可能直接返回空订单
        if tier == "avoid" and abs(pos) < 5:
            return []
        
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
        
        # 更新产品表现评估
        self.update_product_performance(state)
        
        # 更新产品分类
        self.update_product_tiers()
        
        # 对每个产品应用自适应做市策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.adaptive_market_making(p, depth, state.position.get(p, 0))
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
