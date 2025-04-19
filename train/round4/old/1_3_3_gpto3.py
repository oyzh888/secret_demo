# trader_aggressive.py  –  Round‑4 “High‑Gear” strategy
#
# 复制整段保存为 trader.py 即可提交；无需其他依赖（仅用 numpy）
# -----------------------------------------------------------
from __future__ import annotations
from typing  import Dict, List, Tuple, Any
import numpy as np
from datamodel import Order, OrderDepth, TradingState

# ——————————————————— 超参数 ———————————————————
POS_LIM          = 75     # 官方仓位限制
CONV_LIM         = 10
QTY_MAX_ACTIVE   = 20     # 主动吃单量
QTY_MM           = 10     # 做市挂单量
MA_FAST, MA_SLOW = 6, 24  # 动量均线
IMB_TH           = 0.25   # 盘口不平衡阈值
PRESS_TH         = 0.70   # 买压阈值
REL_SPR_TH       = 0.0012 # 相对价差阈值（0.12%）
MM_SPREAD        = 2      # 做市价差（整数 price tick）
FEE_BUFFER       = 0      # 手续费缓冲（若有，填正数）

PRODUCT = "MAGNIFICENT_MACARONS"

# ——————————————————— 工具函数 ———————————————————
def ma(arr: List[float], win: int) -> float:
    return arr[-1] if len(arr) < win else float(np.mean(arr[-win:]))

def imbalance(od: OrderDepth, depth: int = 4) -> Tuple[float, float]:
    buy_val = sum(p*q for p,q in sorted(od.buy_orders.items(),reverse=True)[:depth])
    sell_val= sum(p*q for p,q in sorted(od.sell_orders.items())[:depth])
    tot = buy_val + sell_val
    if tot == 0: return 0.5, 0.0
    buyP = buy_val / tot
    imb  = (buy_val - sell_val) / tot
    return buyP, imb

# ——————————————————— 主类 ———————————————————
class Trader:
    def __init__(self):
        self.mid_hist: List[float] = []
        self.buyP_hist: List[float] = []

    # ---------------------------------- run ----------------------------------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, Any]:
        orders: Dict[str, List[Order]] = {PRODUCT: []}
        conversions = 0                                                    # 未用

        if PRODUCT not in state.order_depths:
            return orders, conversions, state.traderData

        od  = state.order_depths[PRODUCT]
        pos = state.position.get(PRODUCT, 0)

        if not od.buy_orders or not od.sell_orders:
            return orders, conversions, state.traderData

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2
        rel_spr = (best_ask - best_bid) / best_bid
        self.mid_hist.append(mid)

        buyP, imb = imbalance(od)
        self.buyP_hist.append(buyP)

        ma_f = ma(self.mid_hist, MA_FAST)
        ma_s = ma(self.mid_hist, MA_SLOW)

        # ========== 1. 趋势进攻 ==========
        long_sig  = (ma_f > ma_s) and (buyP > PRESS_TH) and (imb >  IMB_TH)
        short_sig = (ma_f < ma_s) and (buyP < 1-PRESS_TH) and (imb < -IMB_TH)

        # 允许在趋势信号存在且价差合理时直接吃单
        if long_sig and pos < POS_LIM:
            qty = min(QTY_MAX_ACTIVE, POS_LIM - pos)
            orders[PRODUCT].append(Order(PRODUCT, best_ask + FEE_BUFFER,  qty))

        if short_sig and pos > -POS_LIM:
            qty = min(QTY_MAX_ACTIVE, POS_LIM + pos)
            orders[PRODUCT].append(Order(PRODUCT, best_bid - FEE_BUFFER, -qty))

        # ========== 2. 做市捕差 ==========
        # 价差足够宽且当前仓位较轻 → 同时挂双边
        if rel_spr > REL_SPR_TH and abs(pos) <= 15:
            bid_mk = best_bid + MM_SPREAD
            ask_mk = best_ask - MM_SPREAD
            orders[PRODUCT] += [
                Order(PRODUCT, bid_mk,  QTY_MM),
                Order(PRODUCT, ask_mk, -QTY_MM),
            ]

        # ========== 3. 快速止盈/止损 ==========
        # 均线反转 or 盘口压强反向 → 立即减仓
        rev_long = (ma_f < ma_s) or (buyP < 0.45)
        rev_short= (ma_f > ma_s) or (buyP > 0.55)

        if pos > 0 and rev_long:
            qty = min(QTY_MAX_ACTIVE, pos)
            orders[PRODUCT].append(Order(PRODUCT, best_bid - FEE_BUFFER, -qty))

        if pos < 0 and rev_short:
            qty = min(QTY_MAX_ACTIVE, -pos)
            orders[PRODUCT].append(Order(PRODUCT, best_ask + FEE_BUFFER,  qty))

        return orders, conversions, state.traderData
