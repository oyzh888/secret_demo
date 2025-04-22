from typing import Dict, List, Tuple, Optional, Set
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np

# 只关注火山岩及其凭证
VOLCANIC_PRODUCTS = {
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200, 
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200, 
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200
}

# 其他产品的限制（仅用于参考，不会主动交易）
OTHER_PRODUCTS = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "MAGNIFICENT_MACARONS": 75
}

# 合并所有产品限制
LIMIT = {**VOLCANIC_PRODUCTS, **OTHER_PRODUCTS}

# 参数设置 - 非常保守
PARAM = {
    "spread_multiplier": 2.5,     # 更宽的价差
    "min_spread": 3,              # 最小价差
    "position_limit_pct": 0.3,    # 只使用30%的仓位限制
    "vol_window": 30,             # 更长的波动率窗口
    "vol_scale": 0.8,             # 降低波动率影响
    "arb_threshold": 0.005,       # 套利阈值 (0.5%)
    "arb_size_limit": 0.05,       # 套利交易规模限制
    "panic_threshold": 0.5,       # 更低的恐慌阈值
    "panic_spread_add": 8,        # 恐慌模式下更宽的价差
    "max_order_count": 2,         # 每个产品最多挂2个订单
    "trend_window": 50,           # 更长的趋势检测窗口
    "trend_threshold": 0.7,       # 更高的趋势确认阈值
    "fair_value_adjust": 0.2      # 公平价值调整因子
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.position_history = defaultdict(list)
        self.fair_values = {}
        self.trends = defaultdict(int)  # 1=上升, -1=下降, 0=中性
        self.last_trade_prices = defaultdict(list)
        self.voucher_fair_values = {}
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < PARAM["vol_window"]:
            return 3  # 默认较高波动率
        return statistics.stdev(h[-PARAM["vol_window"]:]) * PARAM["vol_scale"] or 1
        
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
    
    def calculate_fair_values(self, state: TradingState):
        """计算所有火山岩产品的公平价值"""
        # 首先计算VOLCANIC_ROCK的公平价值
        if "VOLCANIC_ROCK" in state.order_depths:
            depth = state.order_depths["VOLCANIC_ROCK"]
            mid = self._mid(depth)
            if mid is not None:
                self.fair_values["VOLCANIC_ROCK"] = mid
                
                # 然后计算各凭证的公平价值
                for voucher in [p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"]:
                    # 从凭证名称中提取价值
                    try:
                        voucher_value = int(voucher.split("_")[-1])
                        # 计算理论公平价值比率
                        fair_ratio = voucher_value / 10000  # 标准化到10000基准
                        # 计算凭证的公平价值
                        self.voucher_fair_values[voucher] = fair_ratio * mid
                    except ValueError:
                        continue
    
    def check_arbitrage(self, state: TradingState) -> Dict[str, List[Order]]:
        """检查套利机会"""
        result = {}
        
        # 如果没有VOLCANIC_ROCK数据，无法套利
        if "VOLCANIC_ROCK" not in state.order_depths or "VOLCANIC_ROCK" not in self.fair_values:
            return result
            
        rock_depth = state.order_depths["VOLCANIC_ROCK"]
        rock_fair_value = self.fair_values["VOLCANIC_ROCK"]
        
        # 检查每个凭证
        for voucher in [p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"]:
            if voucher not in state.order_depths or voucher not in self.voucher_fair_values:
                continue
                
            voucher_depth = state.order_depths[voucher]
            voucher_fair_value = self.voucher_fair_values[voucher]
            
            # 提取凭证价值
            try:
                voucher_value = int(voucher.split("_")[-1])
            except ValueError:
                continue
                
            # 检查买入凭证、卖出VOLCANIC_ROCK的套利机会
            if voucher_depth.sell_orders and rock_depth.buy_orders:
                best_ask_voucher = min(voucher_depth.sell_orders.keys())
                best_bid_rock = max(rock_depth.buy_orders.keys())
                
                # 计算理论套利利润
                rock_qty_per_voucher = voucher_value / 10000
                voucher_cost = best_ask_voucher
                rock_revenue = best_bid_rock * rock_qty_per_voucher
                
                profit_pct = (rock_revenue - voucher_cost) / voucher_cost
                
                # 如果利润超过阈值，执行套利
                if profit_pct > PARAM["arb_threshold"]:
                    # 计算交易规模
                    rock_pos = state.position.get("VOLCANIC_ROCK", 0)
                    voucher_pos = state.position.get(voucher, 0)
                    
                    # 限制套利规模
                    arb_size_limit = int(VOLCANIC_PRODUCTS[voucher] * PARAM["arb_size_limit"])
                    
                    max_buy_voucher = min(
                        abs(voucher_depth.sell_orders[best_ask_voucher]),
                        VOLCANIC_PRODUCTS[voucher] - voucher_pos,
                        arb_size_limit
                    )
                    
                    if max_buy_voucher > 0:
                        # 买入凭证
                        if voucher not in result:
                            result[voucher] = []
                        result[voucher].append(Order(voucher, best_ask_voucher, max_buy_voucher))
                        
                        # 卖出等价的VOLCANIC_ROCK
                        rock_qty = int(max_buy_voucher * rock_qty_per_voucher)
                        max_sell_rock = min(
                            rock_depth.buy_orders[best_bid_rock],
                            rock_pos + VOLCANIC_PRODUCTS["VOLCANIC_ROCK"]
                        )
                        
                        if rock_qty > 0 and rock_qty <= max_sell_rock:
                            if "VOLCANIC_ROCK" not in result:
                                result["VOLCANIC_ROCK"] = []
                            result["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", best_bid_rock, -rock_qty))
            
            # 检查卖出凭证、买入VOLCANIC_ROCK的套利机会
            if voucher_depth.buy_orders and rock_depth.sell_orders:
                best_bid_voucher = max(voucher_depth.buy_orders.keys())
                best_ask_rock = min(rock_depth.sell_orders.keys())
                
                # 计算理论套利利润
                rock_qty_per_voucher = voucher_value / 10000
                voucher_revenue = best_bid_voucher
                rock_cost = best_ask_rock * rock_qty_per_voucher
                
                profit_pct = (voucher_revenue - rock_cost) / rock_cost
                
                # 如果利润超过阈值，执行套利
                if profit_pct > PARAM["arb_threshold"]:
                    # 计算交易规模
                    rock_pos = state.position.get("VOLCANIC_ROCK", 0)
                    voucher_pos = state.position.get(voucher, 0)
                    
                    # 限制套利规模
                    arb_size_limit = int(VOLCANIC_PRODUCTS[voucher] * PARAM["arb_size_limit"])
                    
                    max_sell_voucher = min(
                        voucher_depth.buy_orders[best_bid_voucher],
                        voucher_pos + VOLCANIC_PRODUCTS[voucher],
                        arb_size_limit
                    )
                    
                    if max_sell_voucher > 0:
                        # 卖出凭证
                        if voucher not in result:
                            result[voucher] = []
                        result[voucher].append(Order(voucher, best_bid_voucher, -max_sell_voucher))
                        
                        # 买入等价的VOLCANIC_ROCK
                        rock_qty = int(max_sell_voucher * rock_qty_per_voucher)
                        max_buy_rock = min(
                            abs(rock_depth.sell_orders[best_ask_rock]),
                            VOLCANIC_PRODUCTS["VOLCANIC_ROCK"] - rock_pos
                        )
                        
                        if rock_qty > 0 and rock_qty <= max_buy_rock:
                            if "VOLCANIC_ROCK" not in result:
                                result["VOLCANIC_ROCK"] = []
                            result["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", best_ask_rock, rock_qty))
        
        return result
    
    def conservative_mm(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """保守的做市策略"""
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
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差 - 使用更宽的价差
        spread = max(PARAM["min_spread"], int(vol * PARAM["spread_multiplier"]))
        
        # 根据趋势调整价格
        if trend == 1:  # 上升趋势
            buy_px = int(mid - spread * 0.8)  # 更积极地买入
            sell_px = int(mid + spread * 1.2)  # 更保守地卖出
        elif trend == -1:  # 下降趋势
            buy_px = int(mid - spread * 1.2)  # 更保守地买入
            sell_px = int(mid + spread * 0.8)  # 更积极地卖出
        else:  # 中性趋势
            buy_px = int(mid - spread)
            sell_px = int(mid + spread)
        
        # 计算交易规模 - 使用更小的规模
        max_position = int(VOLCANIC_PRODUCTS[p] * PARAM["position_limit_pct"])
        size = max(1, max_position // 5)  # 每次只用20%的允许仓位
        
        # 恐慌模式 - 如果仓位接近限制，更积极地平仓
        if abs(pos) >= VOLCANIC_PRODUCTS[p] * PARAM["panic_threshold"]:
            if pos > 0:  # 多仓，需要卖出
                sell_px = int(mid - PARAM["panic_spread_add"])  # 更积极地卖出
                size = max(size, abs(pos) // 2)  # 更大的卖出规模
                # 只卖出，不买入
                orders.append(Order(p, sell_px, -min(size, pos)))
                return orders
            else:  # 空仓，需要买入
                buy_px = int(mid + PARAM["panic_spread_add"])  # 更积极地买入
                size = max(size, abs(pos) // 2)  # 更大的买入规模
                # 只买入，不卖出
                orders.append(Order(p, buy_px, min(size, -pos)))
                return orders
        
        # 正常做市 - 限制订单数量
        if len(orders) < PARAM["max_order_count"]:
            # 只在仓位允许的情况下挂单
            if pos < max_position:
                orders.append(Order(p, buy_px, min(size, max_position - pos)))
            
            if pos > -max_position:
                orders.append(Order(p, sell_px, -min(size, max_position + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 计算所有火山岩产品的公平价值
        self.calculate_fair_values(state)
        
        # 首先检查套利机会
        arb_orders = self.check_arbitrage(state)
        result.update(arb_orders)
        
        # 然后对每个火山岩产品应用保守的做市策略
        for p in VOLCANIC_PRODUCTS:
            # 如果已经在套利中交易了这个产品，跳过
            if p in result:
                continue
                
            if p in state.order_depths:
                depth = state.order_depths[p]
                pos = state.position.get(p, 0)
                result[p] = self.conservative_mm(p, depth, pos)
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
