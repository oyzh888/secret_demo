from typing import Dict, List, Tuple, Optional
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

# 专注产品
FOCUS_PRODUCTS = {
    "RAINFOREST_RESIN": 50,
    "CROISSANTS": 250,
    "MAGNIFICENT_MACARONS": 75
}

# 参数设置 - 雨林树脂专家
PARAM = {
    "tight_spread": 1,        # 非常窄的基础价差
    "k_vol": 0.5,             # 降低波动率影响
    "mm_size_frac": 0.9,      # 非常大的交易规模
    "aggr_take": True,        # 主动吃单
    "max_position_pct": 0.95, # 使用95%的仓位限制
    "order_levels": 3,        # 挂单层数
    "level_spacing": 1,       # 层间距
    "level_size_decay": 0.7,  # 层规模衰减
    "trend_window": 15,       # 趋势检测窗口
    "trend_threshold": 0.6,   # 趋势检测阈值
    "min_profit_target": 2,   # 最小利润目标
    "max_spread": 5,          # 最大价差
    "imbalance_threshold": 0.3 # 订单簿不平衡阈值
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.position_history = defaultdict(list)
        self.order_book_imbalance = defaultdict(float)
        self.trends = defaultdict(int)  # 1=上升, -1=下降, 0=中性
        self.last_mid_price = {}
        self.trade_count = defaultdict(int)  # 交易计数
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) * PARAM["k_vol"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
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
    
    def specialist_strategy(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """雨林树脂专家策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 记录价格历史
        self.prices[p].append(mid)
        
        # 记录仓位历史
        self.position_history[p].append(pos)
        
        # 计算订单簿不平衡度
        imbalance = self.calculate_order_book_imbalance(depth)
        self.order_book_imbalance[p] = imbalance
        
        # 检测趋势
        trend = self.detect_trend(p)
        self.trends[p] = trend
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差 - 使用更窄的价差
        spread = min(PARAM["max_spread"], max(PARAM["tight_spread"], int(vol)))
        
        # 根据订单簿不平衡调整价差
        if abs(imbalance) > PARAM["imbalance_threshold"]:
            # 如果买方压力大，增加卖出价格；如果卖方压力大，降低买入价格
            spread_adjustment = int(imbalance * 2)  # 将不平衡度转换为价格调整
            spread += abs(spread_adjustment)
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 根据趋势和不平衡度调整价格
        if trend == 1 or imbalance > PARAM["imbalance_threshold"]:  # 上升趋势或买方压力大
            buy_px += 1  # 更积极地买入
            sell_px += 1  # 更保守地卖出
        elif trend == -1 or imbalance < -PARAM["imbalance_threshold"]:  # 下降趋势或卖方压力大
            buy_px -= 1  # 更保守地买入
            sell_px -= 1  # 更积极地卖出
        
        # 计算交易规模 - 使用更大的规模
        max_position = int(LIMIT[p] * PARAM["max_position_pct"])
        base_size = max(1, max_position // 3)  # 基础规模
        
        # 主动吃单
        if PARAM["aggr_take"]:
            b, a = best_bid_ask(depth)
            if a is not None and b is not None:
                # 如果卖单价格低于我们的买入价，主动买入
                if a < buy_px and pos < max_position:
                    qty = min(base_size, max_position - pos, abs(depth.sell_orders[a]))
                    if qty > 0: 
                        orders.append(Order(p, a, qty))
                        self.trade_count[p] += 1
                
                # 如果买单价格高于我们的卖出价，主动卖出
                if b > sell_px and pos > -max_position:
                    qty = min(base_size, max_position + pos, depth.buy_orders[b])
                    if qty > 0: 
                        orders.append(Order(p, b, -qty))
                        self.trade_count[p] += 1
        
        # 多层次挂单
        for level in range(PARAM["order_levels"]):
            # 计算当前层的价格
            level_buy_px = buy_px - level * PARAM["level_spacing"]
            level_sell_px = sell_px + level * PARAM["level_spacing"]
            
            # 计算当前层的规模
            level_size = int(base_size * (PARAM["level_size_decay"] ** level))
            
            # 确保最小规模为1
            level_size = max(1, level_size)
            
            # 挂买单
            if pos < max_position:
                orders.append(Order(p, level_buy_px, min(level_size, max_position - pos)))
            
            # 挂卖单
            if pos > -max_position:
                orders.append(Order(p, level_sell_px, -min(level_size, max_position + pos)))
        
        return orders
    
    def passive_strategy(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """被动策略 - 用于非专注产品"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差 - 使用更宽的价差
        spread = max(PARAM["tight_spread"] * 2, int(vol * 1.5))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模 - 使用更小的规模
        max_position = int(LIMIT[p] * PARAM["max_position_pct"] * 0.3)  # 只使用30%的仓位限制
        size = max(1, max_position // 2)
        
        # 常规做市订单
        if pos < max_position:
            orders.append(Order(p, buy_px, min(size, max_position - pos)))
        if pos > -max_position:
            orders.append(Order(p, sell_px, -min(size, max_position + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 对每个产品应用相应的策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                pos = state.position.get(p, 0)
                
                if p in FOCUS_PRODUCTS:
                    # 对专注产品使用专家策略
                    result[p] = self.specialist_strategy(p, depth, pos)
                else:
                    # 对其他产品使用被动策略
                    result[p] = self.passive_strategy(p, depth, pos)
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
