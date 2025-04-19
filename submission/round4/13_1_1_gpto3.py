#######################################################################
# Variant‑1  Enhanced Market‑Making with Product‑Specific Parameters
#######################################################################
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
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

# 产品特定参数
PRODUCT_PARAMS = {
    # 盈利稳定的产品 - 更激进的做市
    "RAINFOREST_RESIN": {
        "tight_spread": 1,
        "k_vol": 0.8,           # 降低波动率影响
        "mm_size_frac": 0.4,    # 增加做市规模
        "panic_ratio": 0.9,     # 提高恐慌阈值
        "aggr_take": True
    },
    # 波动大的产品 - 更保守的做市
    "CROISSANTS": {
        "tight_spread": 2,      # 增加基础点差
        "k_vol": 1.8,          # 提高波动率影响
        "mm_size_frac": 0.15,   # 降低做市规模
        "panic_ratio": 0.7,     # 降低恐慌阈值
        "aggr_take": False      # 关闭吃单
    },
    "JAMS": {
        "tight_spread": 2,
        "k_vol": 1.8,
        "mm_size_frac": 0.15,
        "panic_ratio": 0.7,
        "aggr_take": False
    },
    # 默认参数
    "DEFAULT": {
        "tight_spread": 1,
        "k_vol": 1.2,
        "panic_ratio": 0.8,
        "panic_add": 4,
        "mm_size_frac": 0.25,
        "aggr_take": True
    }
}

def best_bid_ask(depth: OrderDepth) -> Tuple[int|None,int|None]:
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)   # 用于估波动
        self.vols = defaultdict(list)     # 存储成交量
        
    def _vol(self, p:str) -> float:
        h=self.prices[p]
        if len(h)<15: return 1
        return statistics.stdev(h[-15:]) or 1
        
    def _mid(self, depth): 
        b,a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
        
    def _get_params(self, product: str) -> dict:
        return PRODUCT_PARAMS.get(product, PRODUCT_PARAMS["DEFAULT"])
        
    def mm_product(self, p:str, depth:OrderDepth, pos:int)->List[Order]:
        orders=[]
        mid=self._mid(depth)
        if mid is None: return orders
        
        # 获取产品特定参数
        params = self._get_params(p)
        
        self.prices[p].append(mid)
        spread=int(params["tight_spread"]+params["k_vol"]*self._vol(p))
        buy_px=int(mid-spread)
        sell_px=int(mid+spread)
        size=max(1,int(LIMIT[p]*params["mm_size_frac"]))
        
        # 计算成交量
        vol = sum(abs(q) for q in depth.buy_orders.values()) + sum(abs(q) for q in depth.sell_orders.values())
        self.vols[p].append(vol)
        
        # 根据成交量调整size
        if len(self.vols[p]) > 10:
            avg_vol = statistics.mean(self.vols[p][-10:])
            if vol > avg_vol * 1.5:  # 当前成交量明显高于平均
                size = int(size * 1.3)  # 增加做市规模
        
        # panic 强清
        if abs(pos)>=LIMIT[p]*params["panic_ratio"]:
            buy_px=int(mid-params.get("panic_add", 4)-spread)
            sell_px=int(mid+params.get("panic_add", 4)+spread)
            size=max(size,abs(pos)//2)
            
        # 吃单
        b,a = best_bid_ask(depth)
        if params["aggr_take"] and b is not None and a is not None:
            if a<mid-spread and pos< LIMIT[p]:
                qty=min(size,LIMIT[p]-pos,abs(depth.sell_orders[a]))
                if qty: orders.append(Order(p,a,qty))
            if b>mid+spread and pos>-LIMIT[p]:
                qty=min(size,LIMIT[p]+pos,depth.buy_orders[b])
                if qty: orders.append(Order(p,b,-qty))
                
        # 常规挂单
        if pos< LIMIT[p]:
            orders.append(Order(p,buy_px,min(size,LIMIT[p]-pos)))
        if pos>-LIMIT[p]:
            orders.append(Order(p,sell_px,-min(size,LIMIT[p]+pos)))
            
        return orders
        
    def run(self,state:TradingState):
        res:Dict[str,List[Order]]={}
        for p,depth in state.order_depths.items():
            res[p]=self.mm_product(p,depth,state.position.get(p,0))
        return res,0,state.traderData 