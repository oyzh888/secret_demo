from typing import Dict, List, Tuple, Optional, Set
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

# 产品分类
PRODUCT_TIERS = {
    # 高风险产品 - 限制交易规模，更宽的价差
    "high_risk": {
        "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
        "PICNIC_BASKET1", "PICNIC_BASKET2"
    },
    # 中等风险产品 - 适中的交易规模和价差
    "medium_risk": {
        "MAGNIFICENT_MACARONS", "VOLCANIC_ROCK_VOUCHER_10000", 
        "SQUID_INK", "DJEMBES", "JAMS", "VOLCANIC_ROCK_VOUCHER_10250"
    },
    # 低风险产品 - 更大的交易规模，更紧的价差
    "low_risk": {
        "CROISSANTS", "KELP", "VOLCANIC_ROCK_VOUCHER_10500", "RAINFOREST_RESIN"
    }
}

# 关键交易对手
KEY_COUNTERPARTIES = {"Caesar", "Camilla", "Charlie"}

# 参数
PARAM = {
    # 基础参数
    "base_spread": 2,
    "vol_factor": 1.0,
    "position_limit_pct": 0.6,  # 只使用60%的仓位限制
    "stop_loss_pct": 0.02,      # 止损阈值 (2%)
    
    # 风险层级参数
    "risk_tiers": {
        "high_risk": {
            "spread_multiplier": 2.0,
            "size_limit": 0.15,
            "aggr_take": False
        },
        "medium_risk": {
            "spread_multiplier": 1.5,
            "size_limit": 0.25,
            "aggr_take": True
        },
        "low_risk": {
            "spread_multiplier": 1.0,
            "size_limit": 0.35,
            "aggr_take": True
        }
    },
    
    # 交易对手参数
    "cp_aggr_add": 1,        # 对激进交易对手增加价差
    "cp_passive_sub": 0,     # 对被动交易对手减少价差
    
    # 趋势检测参数
    "trend_window": 30,
    "trend_threshold": 0.65,
    
    # 止损参数
    "max_loss_per_product": 1000,  # 每个产品的最大损失
    "cooldown_period": 10          # 止损后的冷却期
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.position_values = defaultdict(list)  # 记录仓位价值
        self.last_counterparty = defaultdict(lambda: None)  # 上次交易对手
        self.product_pnl = defaultdict(float)  # 每个产品的盈亏
        self.stop_loss_active = defaultdict(bool)  # 止损激活状态
        self.stop_loss_cooldown = defaultdict(int)  # 止损冷却计数器
        self.fair_values = {}  # 产品公平价值
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) * PARAM["vol_factor"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def update_last_counterparty(self, own_trades: Dict[str, List]):
        """更新上次交易对手"""
        for symbol, trades in own_trades.items():
            if trades:
                # 使用最近交易的交易对手
                trade = trades[-1]
                if hasattr(trade, 'counter_party'):
                    self.last_counterparty[symbol] = trade.counter_party
                # 尝试获取买方/卖方
                elif hasattr(trade, 'buyer') and hasattr(trade, 'seller'):
                    # 如果我们是买方，交易对手是卖方
                    if trade.buyer == 'SUBMISSION':
                        self.last_counterparty[symbol] = trade.seller
                    # 如果我们是卖方，交易对手是买方
                    elif trade.seller == 'SUBMISSION':
                        self.last_counterparty[symbol] = trade.buyer
    
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
    
    def adjust_price_for_counterparty(self, base_price: int, is_buy: bool, counterparty: Optional[str]) -> int:
        """根据交易对手调整价格"""
        # 如果没有交易对手信息，使用基础价格
        if not counterparty or counterparty not in KEY_COUNTERPARTIES:
            return base_price
            
        if counterparty == "Caesar":
            # Caesar买入激进，卖出更高
            if not is_buy:
                return base_price + PARAM["cp_aggr_add"]
        elif counterparty == "Camilla":
            # Camilla买入有偏向，卖出更高（趋势跟随）
            if not is_buy:
                return base_price + PARAM["cp_aggr_add"]
        elif counterparty == "Charlie":
            # Charlie中性，可能略微收紧价差
            pass  # 暂时保持基础价格
            
        return base_price
    
    def update_product_pnl(self, state: TradingState):
        """更新产品盈亏"""
        for p, pos in state.position.items():
            if p not in self.prices or not self.prices[p]:
                continue
                
            current_price = self.prices[p][-1]
            current_value = pos * current_price
            
            # 记录仓位价值
            self.position_values[p].append(current_value)
            
            # 计算盈亏变化
            if len(self.position_values[p]) > 1:
                value_change = self.position_values[p][-1] - self.position_values[p][-2]
                self.product_pnl[p] += value_change
    
    def check_stop_loss(self, p: str) -> bool:
        """检查是否触发止损"""
        # 如果已经在冷却期，减少冷却计数器
        if self.stop_loss_cooldown[p] > 0:
            self.stop_loss_cooldown[p] -= 1
            return True  # 仍在冷却期，维持止损状态
            
        # 如果止损已激活但冷却期结束，重置止损状态
        if self.stop_loss_active[p] and self.stop_loss_cooldown[p] == 0:
            self.stop_loss_active[p] = False
            self.product_pnl[p] = 0  # 重置盈亏计数
            return False
            
        # 检查是否需要激活止损
        if self.product_pnl[p] < -PARAM["max_loss_per_product"]:
            self.stop_loss_active[p] = True
            self.stop_loss_cooldown[p] = PARAM["cooldown_period"]
            return True
            
        return False
    
    def get_risk_tier(self, p: str) -> str:
        """获取产品的风险层级"""
        if p in PRODUCT_TIERS["high_risk"]:
            return "high_risk"
        elif p in PRODUCT_TIERS["medium_risk"]:
            return "medium_risk"
        else:
            return "low_risk"
    
    def hybrid_strategy(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """混合风险控制策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        self.fair_values[p] = mid
        
        # 检查是否触发止损
        if self.check_stop_loss(p):
            # 如果触发止损，只平仓，不开新仓
            if pos > 0:
                # 有多仓，卖出平仓
                sell_px = int(mid - 2)  # 更积极地卖出
                orders.append(Order(p, sell_px, -pos))
            elif pos < 0:
                # 有空仓，买入平仓
                buy_px = int(mid + 2)  # 更积极地买入
                orders.append(Order(p, buy_px, -pos))
            return orders
        
        # 获取产品风险层级
        risk_tier = self.get_risk_tier(p)
        tier_params = PARAM["risk_tiers"][risk_tier]
        
        # 检测趋势
        trend = self.detect_trend(p)
        
        # 获取上次交易对手
        last_cp = self.last_counterparty[p]
        
        # 计算价差
        vol = self._vol(p)
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
        
        # 根据交易对手调整价格
        buy_px = self.adjust_price_for_counterparty(buy_px, True, last_cp)
        sell_px = self.adjust_price_for_counterparty(sell_px, False, last_cp)
        
        # 计算交易规模
        max_position = int(LIMIT[p] * PARAM["position_limit_pct"] * tier_params["size_limit"])
        size = max(1, max_position // 4)  # 每次只用25%的允许仓位
        
        # 主动吃单
        if tier_params["aggr_take"]:
            b, a = best_bid_ask(depth)
            if a is not None and b is not None:
                if a < mid - spread and pos < max_position:
                    qty = min(size, max_position - pos, abs(depth.sell_orders[a]))
                    if qty > 0: orders.append(Order(p, a, qty))
                if b > mid + spread and pos > -max_position:
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
        
        # 更新交易对手信息
        self.update_last_counterparty(state.own_trades)
        
        # 更新产品盈亏
        self.update_product_pnl(state)
        
        # 对每个产品应用混合策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                result[p] = self.hybrid_strategy(p, depth, state.position.get(p, 0))
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
