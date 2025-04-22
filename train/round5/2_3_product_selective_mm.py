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

# 只交易表现良好的产品
GOOD_PRODUCTS = {
    "RAINFOREST_RESIN": 50,
    "CROISSANTS": 250,
    "MAGNIFICENT_MACARONS": 75,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "KELP": 50
}

# 避开表现不佳的产品
BAD_PRODUCTS = {
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100
}

# 参数
PARAM = {
    "tight_spread": 2,        # 基础价差
    "k_vol": 1.0,             # 波动率因子
    "mm_size_frac": 0.2,      # 订单规模
    "aggr_take": True,        # 主动吃单
    "trend_window": 20,       # 趋势检测窗口
    "trend_threshold": 0.6,   # 趋势检测阈值
    "good_product_boost": 0.3, # 好产品的规模提升
    "bad_product_limit": 0.05  # 坏产品的规模限制
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.product_performance = defaultdict(list)  # 记录每个产品的表现
        self.dynamic_good_products = set(GOOD_PRODUCTS.keys())  # 动态更新的好产品集合
        self.dynamic_bad_products = set(BAD_PRODUCTS.keys())   # 动态更新的坏产品集合
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
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
    
    def update_product_performance(self, state: TradingState):
        """更新产品表现评估"""
        # 根据当前仓位和价格变动评估产品表现
        for p, pos in state.position.items():
            if p not in self.prices or not self.prices[p]:
                continue
                
            current_price = self.prices[p][-1]
            
            # 计算仓位价值变化
            if len(self.prices[p]) > 1:
                price_change = current_price - self.prices[p][-2]
                position_value_change = pos * price_change
                
                # 记录表现
                self.product_performance[p].append(position_value_change)
                
                # 保持最近的表现记录
                if len(self.product_performance[p]) > 50:
                    self.product_performance[p].pop(0)
    
    def update_dynamic_product_lists(self):
        """动态更新好产品和坏产品列表"""
        for p, performance in self.product_performance.items():
            if len(performance) < 10:
                continue
                
            # 计算最近的表现
            recent_performance = sum(performance[-10:])
            
            # 更新产品分类
            if recent_performance > 100:  # 表现良好
                self.dynamic_good_products.add(p)
                if p in self.dynamic_bad_products:
                    self.dynamic_bad_products.remove(p)
            elif recent_performance < -100:  # 表现不佳
                self.dynamic_bad_products.add(p)
                if p in self.dynamic_good_products:
                    self.dynamic_good_products.remove(p)
    
    def selective_mm_product(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """选择性做市策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 检测趋势
        trend = self.detect_trend(p)
        
        # 计算价差
        spread = int(PARAM["tight_spread"] + PARAM["k_vol"] * self._vol(p))
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 根据趋势调整价格
        if trend == 1:  # 上升趋势
            buy_px += 1  # 更积极地买入
            sell_px += 1  # 更保守地卖出
        elif trend == -1:  # 下降趋势
            buy_px -= 1  # 更保守地买入
            sell_px -= 1  # 更积极地卖出
        
        # 根据产品类型调整交易规模
        if p in self.dynamic_good_products:
            # 好产品使用更大的规模
            size = max(1, int(LIMIT[p] * (PARAM["mm_size_frac"] + PARAM["good_product_boost"])))
        elif p in self.dynamic_bad_products:
            # 坏产品使用极小的规模或不交易
            size = max(1, int(LIMIT[p] * PARAM["bad_product_limit"]))
            # 对于非常差的产品，可以选择不交易
            if p in BAD_PRODUCTS:
                return []
        else:
            # 中性产品使用正常规模
            size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"]))
        
        # 主动吃单
        b, a = best_bid_ask(depth)
        if PARAM["aggr_take"] and b is not None and a is not None:
            if a < mid - spread and pos < LIMIT[p]:
                qty = min(size, LIMIT[p] - pos, abs(depth.sell_orders[a]))
                if qty > 0: orders.append(Order(p, a, qty))
            if b > mid + spread and pos > -LIMIT[p]:
                qty = min(size, LIMIT[p] + pos, depth.buy_orders[b])
                if qty > 0: orders.append(Order(p, b, -qty))
        
        # 常规做市订单
        if pos < LIMIT[p]:
            orders.append(Order(p, buy_px, min(size, LIMIT[p] - pos)))
        if pos > -LIMIT[p]:
            orders.append(Order(p, sell_px, -min(size, LIMIT[p] + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 更新产品表现
        self.update_product_performance(state)
        
        # 更新动态产品列表
        self.update_dynamic_product_lists()
        
        # 对每个产品应用选择性做市策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.selective_mm_product(p, depth, state.position.get(p, 0))
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
