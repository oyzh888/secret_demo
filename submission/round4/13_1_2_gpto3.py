#######################################################################
# Variant‑2  Options and Basket Arbitrage Focus
#######################################################################
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple, Any
import statistics, math, jsonpickle
from collections import defaultdict, deque
import numpy as np

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

# 期权参数
OPTION_PARAMS = {
    "VOLCANIC_ROCK_VOUCHER_9500": {"strike": 9500, "delta_limit": 0.3},
    "VOLCANIC_ROCK_VOUCHER_9750": {"strike": 9750, "delta_limit": 0.3},
    "VOLCANIC_ROCK_VOUCHER_10000": {"strike": 10000, "delta_limit": 0.3},
    "VOLCANIC_ROCK_VOUCHER_10250": {"strike": 10250, "delta_limit": 0.3},
    "VOLCANIC_ROCK_VOUCHER_10500": {"strike": 10500, "delta_limit": 0.3}
}

# 篮子组成
BASKET_COMPS = {
    "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
    "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
}

def best_bid_ask(depth: OrderDepth) -> Tuple[int|None,int|None]:
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)   # 价格历史
        self.vols = defaultdict(float)    # 波动率
        self.positions = defaultdict(int)  # 持仓
        
    def _mid(self, depth): 
        b,a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
        
    def _update_vol(self, product: str, price: float):
        self.prices[product].append(price)
        if len(self.prices[product]) > 30:
            returns = np.diff(np.log(self.prices[product][-30:]))
            self.vols[product] = np.std(returns) * np.sqrt(252)
            
    def _black_scholes(self, S, K, T, σ):
        if T <= 0 or σ <= 0:
            return max(0, S-K)
        d1 = (math.log(S/K) + 0.5*σ*σ*T)/(σ*math.sqrt(T))
        d2 = d1 - σ*math.sqrt(T)
        N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
        return S*N(d1) - K*math.exp(-0*T)*N(d2)
        
    def _delta(self, S, K, T, σ):
        if T <= 0 or σ <= 0:
            return 1.0 if S > K else 0.0
        d1 = (math.log(S/K) + 0.5*σ*σ*T)/(σ*math.sqrt(T))
        return 0.5*(1+math.erf(d1/math.sqrt(2)))
        
    def trade_options(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = defaultdict(list)
        if "VOLCANIC_ROCK" not in state.order_depths:
            return orders
            
        # 获取基础资产价格
        rock_mid = self._mid(state.order_depths["VOLCANIC_ROCK"])
        if rock_mid is None:
            return orders
            
        self._update_vol("VOLCANIC_ROCK", rock_mid)
        σ = max(self.vols["VOLCANIC_ROCK"], 0.1)  # 最小波动率设为10%
        T = 2/252  # 假设还剩2天到期
        
        for opt, params in OPTION_PARAMS.items():
            if opt not in state.order_depths:
                continue
                
            depth = state.order_depths[opt]
            bid, ask = best_bid_ask(depth)
            if bid is None or ask is None:
                continue
                
            # 计算理论价和Delta
            theo = self._black_scholes(rock_mid, params["strike"], T, σ)
            delta = self._delta(rock_mid, params["strike"], T, σ)
            
            # 价差超过2%时交易
            if ask < theo * 0.98:  # 期权便宜，买入
                pos = state.position.get(opt, 0)
                if pos < LIMIT[opt]:
                    qty = min(20, LIMIT[opt] - pos)  # 每次最多买20张
                    orders[opt].append(Order(opt, ask, qty))
                    # Delta对冲
                    hedge_qty = int(-qty * delta)
                    if abs(hedge_qty) > 0:
                        rock_pos = state.position.get("VOLCANIC_ROCK", 0)
                        if rock_pos + hedge_qty <= LIMIT["VOLCANIC_ROCK"]:
                            orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK", 
                                state.order_depths["VOLCANIC_ROCK"].sell_orders and min(state.order_depths["VOLCANIC_ROCK"].sell_orders) or int(rock_mid),
                                hedge_qty))
                            
            elif bid > theo * 1.02:  # 期权贵，卖出
                pos = state.position.get(opt, 0)
                if pos > -LIMIT[opt]:
                    qty = min(20, LIMIT[opt] + pos)
                    orders[opt].append(Order(opt, bid, -qty))
                    # Delta对冲
                    hedge_qty = int(qty * delta)
                    if abs(hedge_qty) > 0:
                        rock_pos = state.position.get("VOLCANIC_ROCK", 0)
                        if rock_pos + hedge_qty <= LIMIT["VOLCANIC_ROCK"]:
                            orders["VOLCANIC_ROCK"].append(Order("VOLCANIC_ROCK",
                                state.order_depths["VOLCANIC_ROCK"].buy_orders and max(state.order_depths["VOLCANIC_ROCK"].buy_orders) or int(rock_mid),
                                hedge_qty))
                                
        return orders
        
    def trade_baskets(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = defaultdict(list)
        
        for basket, comps in BASKET_COMPS.items():
            if basket not in state.order_depths:
                continue
                
            # 检查所有组件是否都有市场
            if not all(c in state.order_depths for c in comps):
                continue
                
            # 计算理论价值
            theo = 0
            for comp, ratio in comps.items():
                comp_mid = self._mid(state.order_depths[comp])
                if comp_mid is None:
                    break
                theo += comp_mid * ratio
            else:  # 所有组件都有价格
                basket_bid, basket_ask = best_bid_ask(state.order_depths[basket])
                if basket_bid is None or basket_ask is None:
                    continue
                    
                # 如果篮子比组件贵3%以上，卖篮子买组件
                if basket_bid > theo * 1.03:
                    basket_pos = state.position.get(basket, 0)
                    if basket_pos > -LIMIT[basket]:
                        # 确定可交易数量
                        max_qty = LIMIT[basket] + basket_pos
                        for comp, ratio in comps.items():
                            comp_pos = state.position.get(comp, 0)
                            comp_qty = (LIMIT[comp] - comp_pos) // ratio
                            max_qty = min(max_qty, comp_qty)
                            
                        if max_qty >= 1:
                            # 执行套利
                            orders[basket].append(Order(basket, basket_bid, -max_qty))
                            for comp, ratio in comps.items():
                                comp_qty = max_qty * ratio
                                orders[comp].append(Order(comp, 
                                    state.order_depths[comp].buy_orders and max(state.order_depths[comp].buy_orders) or int(self._mid(state.order_depths[comp])),
                                    comp_qty))
                                    
                # 如果篮子比组件便宜3%以上，买篮子卖组件
                elif basket_ask < theo * 0.97:
                    basket_pos = state.position.get(basket, 0)
                    if basket_pos < LIMIT[basket]:
                        # 确定可交易数量
                        max_qty = LIMIT[basket] - basket_pos
                        for comp, ratio in comps.items():
                            comp_pos = state.position.get(comp, 0)
                            comp_qty = (LIMIT[comp] + comp_pos) // ratio
                            max_qty = min(max_qty, comp_qty)
                            
                        if max_qty >= 1:
                            # 执行套利
                            orders[basket].append(Order(basket, basket_ask, max_qty))
                            for comp, ratio in comps.items():
                                comp_qty = max_qty * ratio
                                orders[comp].append(Order(comp,
                                    state.order_depths[comp].sell_orders and min(state.order_depths[comp].sell_orders) or int(self._mid(state.order_depths[comp])),
                                    -comp_qty))
                                    
        return orders
        
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, Any]:
        # 更新持仓
        self.positions = state.position
        
        # 交易期权
        result = self.trade_options(state)
        
        # 交易篮子
        basket_orders = self.trade_baskets(state)
        for product, orders in basket_orders.items():
            result[product].extend(orders)
            
        return result, 0, state.traderData 