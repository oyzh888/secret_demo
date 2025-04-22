from typing import Dict, List, Tuple, Optional, Set
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np

# 火山岩及其期权
VOLCANIC_PRODUCTS = {
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,  # 行权价9500的看涨期权
    "VOLCANIC_ROCK_VOUCHER_9750": 200,  # 行权价9750的看涨期权
    "VOLCANIC_ROCK_VOUCHER_10000": 200, # 行权价10000的看涨期权
    "VOLCANIC_ROCK_VOUCHER_10250": 200, # 行权价10250的看涨期权
    "VOLCANIC_ROCK_VOUCHER_10500": 200  # 行权价10500的看涨期权
}

# 其他产品（不会主动交易）
OTHER_PRODUCTS = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "MAGNIFICENT_MACARONS": 75
}

# 合并所有产品限制
LIMIT = {**VOLCANIC_PRODUCTS, **OTHER_PRODUCTS}

# 参数设置 - 激进套利
PARAM = {
    "arb_threshold": 0.002,       # 套利阈值 (0.2%)，更低的阈值意味着更激进的套利
    "arb_size_limit": 0.5,        # 套利交易规模限制，使用50%的仓位限制
    "max_position_pct": 0.9,      # 最大仓位百分比，使用90%的仓位限制
    "min_edge": 5,                # 最小价差边际
    "aggressive_entry": True,     # 激进进场
    "aggressive_exit": True,      # 激进出场
    "multi_leg_arb": True,        # 多腿套利
    "vol_window": 20,             # 波动率窗口
    "vol_scale": 1.5,             # 波动率缩放因子
    "max_legs": 3,                # 最大套利腿数
    "max_arb_attempts": 5,        # 每个时间步最大套利尝试次数
    "min_profit_target": 10,      # 最小利润目标
    "max_loss_per_trade": 50      # 每笔交易最大损失
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.fair_values = {}
        self.option_values = {}
        self.position_values = defaultdict(float)
        self.trade_history = defaultdict(list)
        self.arb_opportunities = []
        self.last_arb_timestamp = 0
        
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
    
    def calculate_option_values(self, state: TradingState):
        """计算所有火山岩期权的理论价值"""
        # 首先计算VOLCANIC_ROCK的公平价值
        if "VOLCANIC_ROCK" in state.order_depths:
            depth = state.order_depths["VOLCANIC_ROCK"]
            mid = self._mid(depth)
            if mid is not None:
                self.fair_values["VOLCANIC_ROCK"] = mid
                
                # 计算波动率
                vol = self._vol("VOLCANIC_ROCK")
                
                # 然后计算各期权的理论价值
                for voucher in [p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"]:
                    # 从期权名称中提取行权价
                    try:
                        strike = int(voucher.split("_")[-1])
                        
                        # 简化的期权定价模型（内在价值 + 时间价值）
                        intrinsic_value = max(0, mid - strike)
                        time_value = vol * 10  # 简化的时间价值计算
                        
                        # 期权理论价值
                        option_value = intrinsic_value + time_value
                        self.option_values[voucher] = option_value
                    except ValueError:
                        continue
    
    def find_arbitrage_opportunities(self, state: TradingState) -> List[Dict]:
        """寻找套利机会"""
        opportunities = []
        
        # 如果没有VOLCANIC_ROCK数据，无法套利
        if "VOLCANIC_ROCK" not in state.order_depths or "VOLCANIC_ROCK" not in self.fair_values:
            return opportunities
            
        rock_depth = state.order_depths["VOLCANIC_ROCK"]
        rock_fair_value = self.fair_values["VOLCANIC_ROCK"]
        
        # 1. 单腿套利：期权与基础资产之间
        for voucher in [p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"]:
            if voucher not in state.order_depths or voucher not in self.option_values:
                continue
                
            voucher_depth = state.order_depths[voucher]
            voucher_fair_value = self.option_values[voucher]
            
            # 提取行权价
            try:
                strike = int(voucher.split("_")[-1])
            except ValueError:
                continue
                
            # 检查买入期权、卖出VOLCANIC_ROCK的套利机会
            if voucher_depth.sell_orders and rock_depth.buy_orders:
                best_ask_voucher = min(voucher_depth.sell_orders.keys())
                best_bid_rock = max(rock_depth.buy_orders.keys())
                
                # 计算理论套利利润
                rock_qty_per_voucher = 1  # 每个期权对应1个基础资产
                voucher_cost = best_ask_voucher
                rock_revenue = best_bid_rock - strike  # 减去行权价
                
                profit = rock_revenue - voucher_cost
                
                # 如果利润超过阈值，记录套利机会
                if profit > PARAM["min_profit_target"]:
                    opportunities.append({
                        "type": "buy_option_sell_rock",
                        "option": voucher,
                        "option_price": best_ask_voucher,
                        "rock_price": best_bid_rock,
                        "strike": strike,
                        "profit": profit,
                        "max_size": min(
                            abs(voucher_depth.sell_orders[best_ask_voucher]),
                            rock_depth.buy_orders[best_bid_rock],
                            int(VOLCANIC_PRODUCTS[voucher] * PARAM["arb_size_limit"])
                        )
                    })
            
            # 检查卖出期权、买入VOLCANIC_ROCK的套利机会
            if voucher_depth.buy_orders and rock_depth.sell_orders:
                best_bid_voucher = max(voucher_depth.buy_orders.keys())
                best_ask_rock = min(rock_depth.sell_orders.keys())
                
                # 计算理论套利利润
                rock_qty_per_voucher = 1  # 每个期权对应1个基础资产
                voucher_revenue = best_bid_voucher
                rock_cost = best_ask_rock - strike  # 减去行权价
                
                profit = voucher_revenue - rock_cost
                
                # 如果利润超过阈值，记录套利机会
                if profit > PARAM["min_profit_target"]:
                    opportunities.append({
                        "type": "sell_option_buy_rock",
                        "option": voucher,
                        "option_price": best_bid_voucher,
                        "rock_price": best_ask_rock,
                        "strike": strike,
                        "profit": profit,
                        "max_size": min(
                            voucher_depth.buy_orders[best_bid_voucher],
                            abs(rock_depth.sell_orders[best_ask_rock]),
                            int(VOLCANIC_PRODUCTS[voucher] * PARAM["arb_size_limit"])
                        )
                    })
        
        # 2. 多腿套利：期权与期权之间
        if PARAM["multi_leg_arb"]:
            options = [p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"]
            
            # 检查所有期权对之间的套利机会
            for i, option1 in enumerate(options):
                if option1 not in state.order_depths:
                    continue
                    
                depth1 = state.order_depths[option1]
                try:
                    strike1 = int(option1.split("_")[-1])
                except ValueError:
                    continue
                
                for option2 in options[i+1:]:
                    if option2 not in state.order_depths:
                        continue
                        
                    depth2 = state.order_depths[option2]
                    try:
                        strike2 = int(option2.split("_")[-1])
                    except ValueError:
                        continue
                    
                    # 检查买入低行权价期权、卖出高行权价期权的套利机会（牛市价差策略）
                    if strike1 < strike2 and depth1.sell_orders and depth2.buy_orders:
                        best_ask_option1 = min(depth1.sell_orders.keys())
                        best_bid_option2 = max(depth2.buy_orders.keys())
                        
                        # 计算理论套利利润
                        cost = best_ask_option1
                        revenue = best_bid_option2
                        
                        # 考虑行权价差异
                        theoretical_diff = (strike2 - strike1)
                        actual_diff = revenue - cost
                        
                        # 如果实际差价大于理论差价，存在套利机会
                        if actual_diff > theoretical_diff + PARAM["min_profit_target"]:
                            opportunities.append({
                                "type": "bull_spread",
                                "low_strike_option": option1,
                                "high_strike_option": option2,
                                "low_price": best_ask_option1,
                                "high_price": best_bid_option2,
                                "profit": actual_diff - theoretical_diff,
                                "max_size": min(
                                    abs(depth1.sell_orders[best_ask_option1]),
                                    depth2.buy_orders[best_bid_option2],
                                    int(min(VOLCANIC_PRODUCTS[option1], VOLCANIC_PRODUCTS[option2]) * PARAM["arb_size_limit"])
                                )
                            })
                    
                    # 检查卖出低行权价期权、买入高行权价期权的套利机会（熊市价差策略）
                    if strike1 < strike2 and depth1.buy_orders and depth2.sell_orders:
                        best_bid_option1 = max(depth1.buy_orders.keys())
                        best_ask_option2 = min(depth2.sell_orders.keys())
                        
                        # 计算理论套利利润
                        revenue = best_bid_option1
                        cost = best_ask_option2
                        
                        # 考虑行权价差异
                        theoretical_diff = (strike2 - strike1)
                        actual_diff = revenue - cost
                        
                        # 如果实际差价小于理论差价，存在套利机会
                        if theoretical_diff - actual_diff > PARAM["min_profit_target"]:
                            opportunities.append({
                                "type": "bear_spread",
                                "low_strike_option": option1,
                                "high_strike_option": option2,
                                "low_price": best_bid_option1,
                                "high_price": best_ask_option2,
                                "profit": theoretical_diff - actual_diff,
                                "max_size": min(
                                    depth1.buy_orders[best_bid_option1],
                                    abs(depth2.sell_orders[best_ask_option2]),
                                    int(min(VOLCANIC_PRODUCTS[option1], VOLCANIC_PRODUCTS[option2]) * PARAM["arb_size_limit"])
                                )
                            })
        
        # 按利润排序
        opportunities.sort(key=lambda x: x["profit"], reverse=True)
        
        return opportunities
    
    def execute_arbitrage(self, state: TradingState, opportunities: List[Dict]) -> Dict[str, List[Order]]:
        """执行套利策略"""
        result = {}
        
        # 限制每个时间步的套利尝试次数
        attempts = 0
        
        for opp in opportunities:
            if attempts >= PARAM["max_arb_attempts"]:
                break
                
            if opp["type"] == "buy_option_sell_rock":
                # 买入期权，卖出VOLCANIC_ROCK
                option = opp["option"]
                option_price = opp["option_price"]
                rock_price = opp["rock_price"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS[option] - state.position.get(option, 0))
                
                if size > 0:
                    # 买入期权
                    if option not in result:
                        result[option] = []
                    result[option].append(Order(option, option_price, size))
                    
                    # 卖出VOLCANIC_ROCK
                    if "VOLCANIC_ROCK" not in result:
                        result["VOLCANIC_ROCK"] = []
                    result["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", rock_price, -size))
                    
                    attempts += 1
            
            elif opp["type"] == "sell_option_buy_rock":
                # 卖出期权，买入VOLCANIC_ROCK
                option = opp["option"]
                option_price = opp["option_price"]
                rock_price = opp["rock_price"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS[option] + state.position.get(option, 0))
                
                if size > 0:
                    # 卖出期权
                    if option not in result:
                        result[option] = []
                    result[option].append(Order(option, option_price, -size))
                    
                    # 买入VOLCANIC_ROCK
                    if "VOLCANIC_ROCK" not in result:
                        result["VOLCANIC_ROCK"] = []
                    result["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", rock_price, size))
                    
                    attempts += 1
            
            elif opp["type"] == "bull_spread":
                # 牛市价差策略：买入低行权价期权，卖出高行权价期权
                low_option = opp["low_strike_option"]
                high_option = opp["high_strike_option"]
                low_price = opp["low_price"]
                high_price = opp["high_price"]
                size = min(
                    opp["max_size"],
                    VOLCANIC_PRODUCTS[low_option] - state.position.get(low_option, 0),
                    VOLCANIC_PRODUCTS[high_option] + state.position.get(high_option, 0)
                )
                
                if size > 0:
                    # 买入低行权价期权
                    if low_option not in result:
                        result[low_option] = []
                    result[low_option].append(Order(low_option, low_price, size))
                    
                    # 卖出高行权价期权
                    if high_option not in result:
                        result[high_option] = []
                    result[high_option].append(Order(high_option, high_price, -size))
                    
                    attempts += 1
            
            elif opp["type"] == "bear_spread":
                # 熊市价差策略：卖出低行权价期权，买入高行权价期权
                low_option = opp["low_strike_option"]
                high_option = opp["high_strike_option"]
                low_price = opp["low_price"]
                high_price = opp["high_price"]
                size = min(
                    opp["max_size"],
                    VOLCANIC_PRODUCTS[low_option] + state.position.get(low_option, 0),
                    VOLCANIC_PRODUCTS[high_option] - state.position.get(high_option, 0)
                )
                
                if size > 0:
                    # 卖出低行权价期权
                    if low_option not in result:
                        result[low_option] = []
                    result[low_option].append(Order(low_option, low_price, -size))
                    
                    # 买入高行权价期权
                    if high_option not in result:
                        result[high_option] = []
                    result[high_option].append(Order(high_option, high_price, size))
                    
                    attempts += 1
        
        return result
    
    def aggressive_market_making(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """激进的做市策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 记录价格历史
        self.prices[p].append(mid)
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差 - 使用更窄的价差
        spread = max(1, int(vol * 0.5))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模 - 使用更大的规模
        max_position = int(VOLCANIC_PRODUCTS[p] * PARAM["max_position_pct"])
        size = max(1, max_position // 3)  # 每次用1/3的允许仓位
        
        # 主动吃单
        if PARAM["aggressive_entry"]:
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
        
        # 计算所有火山岩期权的理论价值
        self.calculate_option_values(state)
        
        # 寻找套利机会
        opportunities = self.find_arbitrage_opportunities(state)
        
        # 执行套利策略
        if opportunities:
            arb_orders = self.execute_arbitrage(state, opportunities)
            result.update(arb_orders)
        
        # 对未在套利中交易的产品应用激进做市策略
        for p in VOLCANIC_PRODUCTS:
            # 如果已经在套利中交易了这个产品，跳过
            if p in result:
                continue
                
            if p in state.order_depths:
                depth = state.order_depths[p]
                pos = state.position.get(p, 0)
                result[p] = self.aggressive_market_making(p, depth, pos)
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
