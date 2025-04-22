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

# 参数 - 完全反转原来的费用感知策略
PARAM = {
    "tight_spread": 3,        # 更宽的基础价差
    "k_vol": 0.5,             # 降低波动率影响
    "mm_size_frac": 0.1,      # 减小订单规模
    "aggr_take": False,       # 不主动吃单
    "fee_vol_tiers": [50, 100, 200], # 降低交易量阈值
    "fee_edge_add": [0, -1, -2, -3]  # 反向调整边际 (负值表示更紧的价差)
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.daily_volume = defaultdict(int)  # 跟踪每个产品的日交易量
        self.current_day = -1  # 跟踪当前日期以重置交易量
        self.last_mid_prices = {}  # 记录上一个中间价
        self.price_trends = defaultdict(list)  # 记录价格趋势
        
    def _update_daily_volume(self, state: TradingState):
        """更新日交易量"""
        # 在新的一天开始时重置交易量
        day = state.timestamp // 1_000_000  # 假设时间戳反映天数
        if day != self.current_day:
            self.daily_volume.clear()
            self.current_day = day
        
        # 添加本时间戳的自有交易量
        for product, trades in state.own_trades.items():
            for trade in trades:
                # 处理不同的交易格式
                if hasattr(trade, 'quantity'):
                    self.daily_volume[product] += abs(trade.quantity)
                # 在回测器中，交易可能是元组
                elif isinstance(trade, tuple) and len(trade) >= 4:
                    # 假设格式为 (symbol, buyer/seller, price, quantity, ...)
                    self.daily_volume[product] += abs(trade[3])
    
    def _get_fee_tier_edge(self, p: str) -> int:
        """获取费用层级的边际调整"""
        volume = self.daily_volume[p]
        tier_index = 0
        for i, tier_vol in enumerate(PARAM["fee_vol_tiers"]):
            if volume >= tier_vol:
                tier_index = i + 1
            else:
                break
        return PARAM["fee_edge_add"][tier_index]
    
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) or 1
    
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def _detect_price_trend(self, p: str) -> int:
        """检测价格趋势: 1=上升, -1=下降, 0=中性"""
        if p not in self.last_mid_prices or p not in self.prices:
            return 0
            
        # 记录最近的价格变动
        current_price = self.prices[p][-1] if self.prices[p] else None
        last_price = self.last_mid_prices[p]
        
        if current_price is None or last_price is None:
            return 0
            
        # 记录趋势
        if len(self.price_trends[p]) > 10:
            self.price_trends[p].pop(0)
            
        if current_price > last_price:
            self.price_trends[p].append(1)
        elif current_price < last_price:
            self.price_trends[p].append(-1)
        else:
            self.price_trends[p].append(0)
            
        # 分析趋势
        if len(self.price_trends[p]) < 5:
            return 0
            
        avg_trend = sum(self.price_trends[p][-5:]) / 5
        if avg_trend > 0.6:
            return 1  # 上升趋势
        elif avg_trend < -0.6:
            return -1  # 下降趋势
        return 0  # 中性
    
    def reverse_mm_product(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """反向做市策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 检测趋势
        trend = self._detect_price_trend(p)
        
        # 更新上一个中间价
        self.last_mid_prices[p] = mid
        
        # 计算价差，考虑基础、波动率和费用层级
        fee_edge = self._get_fee_tier_edge(p)
        vol_spread = PARAM["k_vol"] * self._vol(p)
        
        # 反向策略：减小价差而不是增加
        required_edge = PARAM["tight_spread"] + fee_edge
        spread = int(max(1, required_edge - vol_spread))  # 确保最小价差为1
        
        # 根据趋势调整价格
        if trend == 1:  # 上升趋势
            # 反向策略：在上升趋势中更积极地卖出，更保守地买入
            buy_px = int(mid - spread - 1)
            sell_px = int(mid + spread - 1)
        elif trend == -1:  # 下降趋势
            # 反向策略：在下降趋势中更积极地买入，更保守地卖出
            buy_px = int(mid - spread + 1)
            sell_px = int(mid + spread + 1)
        else:  # 中性趋势
            buy_px = int(mid - spread)
            sell_px = int(mid + spread)
        
        # 计算订单规模 - 使用更小的规模
        size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"]))
        
        # 根据仓位调整策略
        position_ratio = abs(pos) / LIMIT[p] if LIMIT[p] > 0 else 0
        
        # 如果仓位接近限制，更积极地平仓
        if position_ratio > 0.7:
            if pos > 0:  # 多仓，需要卖出
                sell_px = int(mid - 2)  # 更积极地卖出
                size = max(size, pos // 2)  # 更大的卖出规模
                orders.append(Order(p, sell_px, -min(size, pos)))
            else:  # 空仓，需要买入
                buy_px = int(mid + 2)  # 更积极地买入
                size = max(size, abs(pos) // 2)  # 更大的买入规模
                orders.append(Order(p, buy_px, min(size, -pos)))
        else:
            # 正常做市
            if pos < LIMIT[p]:
                orders.append(Order(p, buy_px, min(size, LIMIT[p] - pos)))
            if pos > -LIMIT[p]:
                orders.append(Order(p, sell_px, -min(size, LIMIT[p] + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        self._update_daily_volume(state)  # 更新交易量
        
        # 对每个产品应用反向做市策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.reverse_mm_product(p, depth, state.position.get(p, 0))
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
