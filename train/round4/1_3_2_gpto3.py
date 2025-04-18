# trader.py
# https://chatgpt.com/c/68021d38-a838-8013-8342-ea275d9a80c9
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import numpy as np
from datamodel import Order, OrderDepth, TradingState               # 官方提供的类型

# ────────────────────────────── 1. 全局超参数 ──────────────────────────────
@dataclass
class Hyper:
    # 仓位/转换
    POS_LIM: int   = 75
    CONV_LIM: int  = 10           # 目前未用到，但给出接口方便你后面做跨市场套利

    # —— 动量策略参数 ——
    MA_SHORT: int = 6             # 短期均线
    MA_LONG: int = 26            # 长期均线
    PRESS_HIGH: float = 0.70     # 买压阈值
    PRESS_LOW: float = 0.30      # 卖压阈值
    IMB_TH: float = 0.25        # 订单簿不平衡阈值
    
    # —— 仓位管理 ——
    QTY_INIT: int = 10          # 初始下单量
    QTY_ADDON: int = 6          # 加仓下单量
    POS_THRESH: int = 40        # 仓位阈值
    QTY_CLOSE: int = 12         # 平仓数量
    
    # —— ATR止损参数 ——
    ATR_WINDOW: int = 14        # ATR窗口
    ATR_MULT: float = 0.5       # ATR乘数
    ATR_CLOSE_MULT: float = 0.5 # 即时止损ATR乘数

    # —— 微观结构参数 ——
    DEPTH: int = 3              # 只看盘口前 3 档
    FLOW_WIN: int = 5           # 流量比率窗口
    FLOW_HIGH: float = 1.7      # 买流阈值
    FLOW_LOW: float = 0.6       # 卖流阈值
    MICRO_POS_LIM: int = 25     # 微观策略仓位限制
    MICRO_QTY: int = 10         # 微观策略下单量
    TIGHT_SPREAD: float = 2.0   # 快速平仓价差阈值

H = Hyper()                     # 方便下面调用


# ────────────────────────────── 2. 工具函数 ──────────────────────────────
def _ma(arr: List[float], win: int) -> float:
    """简单移动平均；长度不足时返回末值"""
    return arr[-1] if len(arr) < win else np.mean(arr[-win:])

def _norm(val: float, hist: List[float]) -> float:
    """0‑1 归一化，历史长度不足 / 极差为 0 → 返回 0.5"""
    if len(hist) < 2: return 0.5
    lo, hi = min(hist), max(hist)
    return 0.5 if hi == lo else (val - lo) / (hi - lo)


# ────────────────────────────── 3. 交易类 ──────────────────────────────
class Trader:
    def __init__(self) -> None:
        self.mid_hist = []      # 中间价历史
        self.buyP_hist = []     # 买压历史
        self.flow = []          # 买卖流量比率历史

    def _micro_metrics(self, od: OrderDepth) -> Tuple[float, float, float, float]:
        """
        返回：
            buyP   : 买压 (0‑1)
            relSpr : 相对价差
            imb    : 订单簿不平衡 (-1~1, 正为买盘多)
            mid    : 中间价
        """
        # a) best bid / ask
        if not od.buy_orders or not od.sell_orders:
            return 0.5, 1.0, 0.0, 0.0                   # 市场挂空 → 不交易

        bid, ask = max(od.buy_orders), min(od.sell_orders)
        mid = (bid + ask) / 2
        spr = ask - bid
        relSpr = spr / bid if bid else 1.0

        # b) 按价格权重汇总前 DEPTH 档量价
        def _pv(orders: Dict[int, int], reverse=False):
            tot = 0
            for p, q in (sorted(orders.items(), reverse=reverse)[:H.DEPTH]):
                tot += p * q
            return tot

        buy_val = _pv(od.buy_orders, True)
        sell_val = _pv(od.sell_orders, False)
        
        # 更新买卖流量比率
        self.flow.append(buy_val/sell_val if sell_val else 1)
        
        tot_val = buy_val + sell_val
        buyP = buy_val / tot_val if tot_val else 0.5
        imb = (buy_val - sell_val) / tot_val if tot_val else 0.0
        return buyP, relSpr, imb, mid

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, Any]:
        """
        组合策略主入口：动量突破 + 微观结构
        """
        product = "MAGNIFICENT_MACARONS"
        orders: Dict[str, List[Order]] = {product: []}
        conversions: int = 0

        if product not in state.order_depths:
            return orders, conversions, state.traderData

        od = state.order_depths[product]
        pos = state.position.get(product, 0)

        if not od.buy_orders or not od.sell_orders:
            return orders, conversions, state.traderData

        # 获取基本价格信息
        bid, ask = max(od.buy_orders), min(od.sell_orders)
        mid = (bid + ask) / 2
        self.mid_hist.append(mid)

        # 计算技术指标
        ma_short = _ma(self.mid_hist, H.MA_SHORT)
        ma_long = _ma(self.mid_hist, H.MA_LONG)
        buyP, _, imb, _ = self._micro_metrics(od)
        self.buyP_hist.append(buyP)

        # ① 动量信号
        momentum_orders = []
        long_sig = ma_short > ma_long and buyP > H.PRESS_HIGH and imb > H.IMB_TH
        short_sig = ma_short < ma_long and buyP < H.PRESS_LOW and imb < -H.IMB_TH

        # 根据仓位确定下单量
        qty = H.QTY_INIT if abs(pos) < H.POS_THRESH else H.QTY_ADDON

        # 动量策略开仓/加仓
        if long_sig and pos < H.POS_LIM:
            momentum_orders.append(Order(product, ask, qty))
        elif short_sig and pos > -H.POS_LIM:
            momentum_orders.append(Order(product, bid, -qty))

        # ② 微观结构信号
        micro_orders = []
        if len(self.flow) % H.FLOW_WIN == 0:  # 每5个tick检查一次
            ratio = np.mean(self.flow[-H.FLOW_WIN:])
            if ratio > H.FLOW_HIGH and pos > -H.MICRO_POS_LIM:  # 买流扎堆 → 做空
                micro_orders.append(Order(product, bid, -min(H.MICRO_QTY, H.MICRO_POS_LIM + pos)))
            elif ratio < H.FLOW_LOW and pos < H.MICRO_POS_LIM:  # 卖流扎堆 → 做多
                micro_orders.append(Order(product, ask, min(H.MICRO_QTY, H.MICRO_POS_LIM - pos)))

        # ③ 快速平仓：价差收窄且仓位≠0
        if abs(pos) > 0 and ask - bid < H.TIGHT_SPREAD:
            if pos > 0:
                orders[product].append(Order(product, bid, -pos))
            else:
                orders[product].append(Order(product, ask, -pos))
            return orders, conversions, state.traderData

        # ④ 动态止损
        if len(self.mid_hist) >= H.ATR_WINDOW + 1:
            atr = (max(self.mid_hist[-H.ATR_WINDOW:]) - min(self.mid_hist[-H.ATR_WINDOW:])) * H.ATR_MULT
            
            if pos > 0 and (
                mid < ma_long - atr or 
                mid - self.mid_hist[-2] < -atr * H.ATR_CLOSE_MULT
            ):
                orders[product].append(Order(product, bid, -min(pos, H.QTY_CLOSE)))
                return orders, conversions, state.traderData
            
            elif pos < 0 and (
                mid > ma_long + atr or 
                mid - self.mid_hist[-2] > atr * H.ATR_CLOSE_MULT
            ):
                orders[product].append(Order(product, ask, min(-pos, H.QTY_CLOSE)))
                return orders, conversions, state.traderData

        # ⑤ 执行交易信号
        # 优先执行动量策略
        if momentum_orders:
            orders[product].extend(momentum_orders)
        # 如果没有动量信号，执行微观策略
        elif micro_orders:
            orders[product].extend(micro_orders)

        return orders, conversions, state.traderData
