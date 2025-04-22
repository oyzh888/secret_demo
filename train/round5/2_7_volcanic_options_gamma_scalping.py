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

# 参数设置 - 伽马刷单
PARAM = {
    "gamma_threshold": 0.5,      # 伽马值阈值
    "delta_hedge_ratio": 1.0,    # Delta对冲比率
    "rebalance_threshold": 0.1,  # 重新平衡阈值
    "max_position_pct": 0.95,    # 最大仓位百分比
    "scalp_size_pct": 0.3,       # 刷单规模百分比
    "min_price_move": 3,         # 最小价格变动
    "aggressive_entry": True,    # 激进进场
    "aggressive_exit": True,     # 激进出场
    "vol_window": 15,            # 波动率窗口
    "vol_scale": 1.8,            # 波动率缩放因子
    "atm_range": 200,            # 平值期权范围
    "max_scalp_attempts": 10,    # 每个时间步最大刷单尝试次数
    "min_profit_target": 5       # 最小利润目标
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.fair_values = {}
        self.option_greeks = defaultdict(dict)  # 存储期权的希腊字母
        self.position_values = defaultdict(float)
        self.last_rebalance = 0
        self.price_moves = defaultdict(int)
        self.atm_option = None  # 当前平值期权
        
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
    
    def calculate_option_greeks(self, state: TradingState):
        """计算所有火山岩期权的希腊字母"""
        # 首先计算VOLCANIC_ROCK的公平价值
        if "VOLCANIC_ROCK" in state.order_depths:
            depth = state.order_depths["VOLCANIC_ROCK"]
            mid = self._mid(depth)
            if mid is None:
                return
                
            self.fair_values["VOLCANIC_ROCK"] = mid
            
            # 计算波动率
            vol = self._vol("VOLCANIC_ROCK")
            
            # 找到当前平值期权
            closest_strike = float('inf')
            self.atm_option = None
            
            # 然后计算各期权的希腊字母
            for voucher in [p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"]:
                # 从期权名称中提取行权价
                try:
                    strike = int(voucher.split("_")[-1])
                    
                    # 找到最接近平值的期权
                    if abs(strike - mid) < abs(closest_strike - mid):
                        closest_strike = strike
                        self.atm_option = voucher
                    
                    # 计算Delta（简化模型）
                    if mid > strike:  # 实值期权
                        delta = 0.8
                    elif mid < strike - PARAM["atm_range"]:  # 虚值期权
                        delta = 0.2
                    else:  # 平值附近的期权
                        delta = 0.5
                    
                    # 计算Gamma（简化模型）
                    # Gamma在平值附近最大
                    gamma = 1.0 - min(1.0, abs(mid - strike) / PARAM["atm_range"])
                    
                    # 存储希腊字母
                    self.option_greeks[voucher] = {
                        "delta": delta,
                        "gamma": gamma,
                        "strike": strike
                    }
                except ValueError:
                    continue
    
    def detect_price_move(self, state: TradingState) -> bool:
        """检测价格是否有显著变动"""
        if "VOLCANIC_ROCK" not in self.prices or len(self.prices["VOLCANIC_ROCK"]) < 2:
            return False
            
        current_price = self.prices["VOLCANIC_ROCK"][-1]
        previous_price = self.prices["VOLCANIC_ROCK"][-2]
        
        price_change = abs(current_price - previous_price)
        
        return price_change >= PARAM["min_price_move"]
    
    def find_gamma_scalping_opportunities(self, state: TradingState) -> List[Dict]:
        """寻找伽马刷单机会"""
        opportunities = []
        
        # 如果没有检测到价格变动，或者没有平值期权，则不进行刷单
        if not self.detect_price_move(state) or not self.atm_option:
            return opportunities
            
        # 获取平值期权
        atm_option = self.atm_option
        
        # 如果平值期权不在订单深度中，则不进行刷单
        if atm_option not in state.order_depths:
            return opportunities
            
        # 获取平值期权的订单深度
        atm_depth = state.order_depths[atm_option]
        
        # 获取平值期权的希腊字母
        if atm_option not in self.option_greeks:
            return opportunities
            
        gamma = self.option_greeks[atm_option]["gamma"]
        
        # 如果Gamma值不够高，则不进行刷单
        if gamma < PARAM["gamma_threshold"]:
            return opportunities
            
        # 获取平值期权的最优买卖价
        b, a = best_bid_ask(atm_depth)
        if b is None or a is None:
            return opportunities
            
        # 计算期权的中间价
        mid = (b + a) / 2
        
        # 获取基础资产的价格变动方向
        if len(self.prices["VOLCANIC_ROCK"]) < 2:
            return opportunities
            
        current_price = self.prices["VOLCANIC_ROCK"][-1]
        previous_price = self.prices["VOLCANIC_ROCK"][-2]
        
        price_direction = 1 if current_price > previous_price else -1
        
        # 根据价格变动方向和Gamma值确定刷单策略
        if price_direction > 0:
            # 价格上涨，买入期权
            opportunities.append({
                "type": "buy_option",
                "option": atm_option,
                "price": a,  # 买入价格
                "gamma": gamma,
                "max_size": min(
                    abs(atm_depth.sell_orders[a]),
                    int(VOLCANIC_PRODUCTS[atm_option] * PARAM["scalp_size_pct"])
                )
            })
        else:
            # 价格下跌，卖出期权
            opportunities.append({
                "type": "sell_option",
                "option": atm_option,
                "price": b,  # 卖出价格
                "gamma": gamma,
                "max_size": min(
                    atm_depth.buy_orders[b],
                    int(VOLCANIC_PRODUCTS[atm_option] * PARAM["scalp_size_pct"])
                )
            })
        
        # 如果VOLCANIC_ROCK在订单深度中，添加Delta对冲机会
        if "VOLCANIC_ROCK" in state.order_depths:
            rock_depth = state.order_depths["VOLCANIC_ROCK"]
            rock_b, rock_a = best_bid_ask(rock_depth)
            
            if rock_b is not None and rock_a is not None:
                delta = self.option_greeks[atm_option]["delta"]
                
                if price_direction > 0:
                    # 价格上涨，卖出基础资产对冲
                    opportunities.append({
                        "type": "hedge_sell_rock",
                        "price": rock_b,  # 卖出价格
                        "delta": delta,
                        "max_size": min(
                            rock_depth.buy_orders[rock_b],
                            int(VOLCANIC_PRODUCTS["VOLCANIC_ROCK"] * PARAM["scalp_size_pct"])
                        )
                    })
                else:
                    # 价格下跌，买入基础资产对冲
                    opportunities.append({
                        "type": "hedge_buy_rock",
                        "price": rock_a,  # 买入价格
                        "delta": delta,
                        "max_size": min(
                            abs(rock_depth.sell_orders[rock_a]),
                            int(VOLCANIC_PRODUCTS["VOLCANIC_ROCK"] * PARAM["scalp_size_pct"])
                        )
                    })
        
        return opportunities
    
    def execute_gamma_scalping(self, state: TradingState, opportunities: List[Dict]) -> Dict[str, List[Order]]:
        """执行伽马刷单策略"""
        result = {}
        
        # 限制每个时间步的刷单尝试次数
        attempts = 0
        
        for opp in opportunities:
            if attempts >= PARAM["max_scalp_attempts"]:
                break
                
            if opp["type"] == "buy_option":
                # 买入期权
                option = opp["option"]
                price = opp["price"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS[option] - state.position.get(option, 0))
                
                if size > 0:
                    if option not in result:
                        result[option] = []
                    result[option].append(Order(option, price, size))
                    attempts += 1
            
            elif opp["type"] == "sell_option":
                # 卖出期权
                option = opp["option"]
                price = opp["price"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS[option] + state.position.get(option, 0))
                
                if size > 0:
                    if option not in result:
                        result[option] = []
                    result[option].append(Order(option, price, -size))
                    attempts += 1
            
            elif opp["type"] == "hedge_buy_rock":
                # 买入基础资产对冲
                price = opp["price"]
                delta = opp["delta"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS["VOLCANIC_ROCK"] - state.position.get("VOLCANIC_ROCK", 0))
                
                # 根据Delta调整对冲规模
                hedge_size = int(size * delta * PARAM["delta_hedge_ratio"])
                
                if hedge_size > 0:
                    if "VOLCANIC_ROCK" not in result:
                        result["VOLCANIC_ROCK"] = []
                    result["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", price, hedge_size))
                    attempts += 1
            
            elif opp["type"] == "hedge_sell_rock":
                # 卖出基础资产对冲
                price = opp["price"]
                delta = opp["delta"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS["VOLCANIC_ROCK"] + state.position.get("VOLCANIC_ROCK", 0))
                
                # 根据Delta调整对冲规模
                hedge_size = int(size * delta * PARAM["delta_hedge_ratio"])
                
                if hedge_size > 0:
                    if "VOLCANIC_ROCK" not in result:
                        result["VOLCANIC_ROCK"] = []
                    result["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", price, -hedge_size))
                    attempts += 1
        
        return result
    
    def aggressive_scalping(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """激进的刷单策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 记录价格历史
        self.prices[p].append(mid)
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差 - 使用更窄的价差
        spread = max(1, int(vol * 0.3))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模 - 使用更大的规模
        max_position = int(VOLCANIC_PRODUCTS[p] * PARAM["max_position_pct"])
        size = max(1, max_position // 4)  # 每次用1/4的允许仓位
        
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
        
        # 计算所有火山岩期权的希腊字母
        self.calculate_option_greeks(state)
        
        # 寻找伽马刷单机会
        opportunities = self.find_gamma_scalping_opportunities(state)
        
        # 执行伽马刷单策略
        if opportunities:
            scalp_orders = self.execute_gamma_scalping(state, opportunities)
            result.update(scalp_orders)
        
        # 对未在刷单中交易的产品应用激进刷单策略
        for p in VOLCANIC_PRODUCTS:
            # 如果已经在刷单中交易了这个产品，跳过
            if p in result:
                continue
                
            if p in state.order_depths:
                depth = state.order_depths[p]
                pos = state.position.get(p, 0)
                result[p] = self.aggressive_scalping(p, depth, pos)
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
