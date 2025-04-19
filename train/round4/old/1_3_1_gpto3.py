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

    # —— 基本面因子权重（∑=1）——
    W_SUGAR:    float = 0.30      # 糖价（越低越利多）
    W_SUN:      float = 0.25      # 日照指数（越高产量越大 → 利空）
    W_SHIP:     float = 0.15      # 船运成本（越高利空）
    W_TARIFF:   float = 0.15      # 关税（越高利空）
    W_STORAGE:  float = 0.15      # 储存费（恒定 0.1，但若官方未来变动可捕捉）

    # —— 技术面参数 ——
    MA_FAST: int = 8
    MA_SLOW: int = 40

    # —— 微观结构参数 ——
    DEPTH: int    = 4             # 只看盘口前 4 档
    IMB_TH: float = 0.28          # 订单簿不平衡阈值
    SPR_REL_TH: float = 0.12      # 相对价差阈值（%）
    PRESS_TH: float  = 0.68       # 买卖压强阈值
    FLOW_MULT: float = 1.4        # 主动买卖量比率

    # —— 组合与风控 ——
    STR_WIN: int = 10             # 策略评估回看窗口
    SIG_TH:  float = 0.55         # 两策略权重必须 > 0.55 才执行
    QTY_MAX: int   = 12           # 单笔最大下单量
    FLAT_SPREAD_MULT: float = 1.6 # 价差过宽时择机平仓
    FLAT_PSC_NEU: float = 0.50    # 价格得分中性位
    FLAT_PSC_TH: float = 0.08

    # —— 均值回归参数 ——
    COST_WIN: int = 60           # 成本均值窗口
    PRICE_WIN: int = 60          # 价格均值窗口
    MIN_HIST: int = 20          # 最小历史数据要求
    Z_THRESH: float = 1.0       # Z分数阈值
    MM_POS_LIM: int = 15        # 做市商仓位限制
    MM_QTY: int = 5            # 做市商下单量
    MM_SPREAD_MIN: float = 0.0015  # 最小价差要求

H = Hyper()                       # 方便下面调用


# ────────────────────────────── 2. 工具函数 ──────────────────────────────
def _ma(arr: List[float], win: int) -> float:
    """简单移动平均；长度不足时返回末值"""
    return arr[-1] if len(arr) < 1 else np.mean(arr[-win:])

def _norm(val: float, hist: List[float]) -> float:
    """0‑1 归一化，历史长度不足 / 极差为 0 → 返回 0.5"""
    if len(hist) < 2: return 0.5
    lo, hi = min(hist), max(hist)
    return 0.5 if hi == lo else (val - lo) / (hi - lo)


# ────────────────────────────── 3. 交易类 ──────────────────────────────
class Trader:
    def __init__(self) -> None:
        # —— 历史序列 ——  ⤵︎ 这些列表只存必要数据，长度自控，防爆内存
        self.mid_hist, self.sugar_hist, self.sun_hist = [], [], []
        self.ship_hist, self.tariff_hist, self.store_hist = [], [], []
        self.buyP_hist, self.relSpr_hist, self.imb_hist = [], [], []
        self.cost_hist = []  # 成本历史

    def _fair(self, obs) -> float:
        """计算公平价格（基于成本）"""
        sugar = getattr(obs, "sugarPrice", 0.0)
        store = getattr(obs, "storageCost", 0.1)
        self.cost_hist.append(sugar + store)          # 简化：加法近似
        return np.mean(self.cost_hist[-H.COST_WIN:])  # 60‑tick 移动均成本

    # ——————————————————— 3‑A.  基本面 + 技术面（策略 1） ———————————————————
    def _price_score(self, obs) -> float:
        """
        输出 ∈[-1,1] 的综合得分；正值 → 看多
        """
        # a) 读取官方 observation（字段名参考官方 dashboard）
        sugar  = getattr(obs, "sugarPrice", 0.0)
        sun    = getattr(obs, "sunlight", 0.0)
        ship   = getattr(obs, "shippingCosts", 0.0)
        tariff = getattr(obs, "importTariff", 0.0) + getattr(obs, "exportTariff", 0.0)
        store  = getattr(obs, "storageCost", 0.10)   # 官方文档：0.1/个

        # b) 记录历史
        self.sugar_hist.append(sugar)
        self.sun_hist.append(sun)
        self.ship_hist.append(ship)
        self.tariff_hist.append(tariff)
        self.store_hist.append(store)

        # c) 单因子归一化后线性组合
        s_sugar  = _norm(sugar,  self.sugar_hist)      # 越低越好 → 反向
        s_sun    = _norm(sun,    self.sun_hist)        # 越高供给越大 → 反向
        s_ship   = _norm(ship,   self.ship_hist)       # 越高利空
        s_tariff = _norm(tariff, self.tariff_hist)
        s_store  = _norm(store,  self.store_hist)

        long_side = (1 - s_sugar) * H.W_SUGAR          # 低糖价 → 利多
        short_side = (
            s_sun    * H.W_SUN +
            s_ship   * H.W_SHIP +
            s_tariff * H.W_TARIFF +
            s_store  * H.W_STORAGE
        )
        score = long_side - short_side                 # ∈[-1,1] 大致对称
        return max(min(score, 1), -1)

    # ——————————————————— 3‑B.  微观结构（策略 2） ———————————————————
    def _micro_metrics(self, od: OrderDepth) -> Tuple[float, float, float]:
        """
        返回：
            buyP   : 买压 (0‑1)
            relSpr : 相对价差
            imb    : 订单簿不平衡 (-1~1, 正为买盘多)
        """
        # a) best bid / ask
        if not od.buy_orders or not od.sell_orders:
            return 0.5, 1.0, 0.0                        # 市场挂空 → 不交易

        bid, ask = max(od.buy_orders), min(od.sell_orders)
        mid      = (bid + ask) / 2
        spr      = ask - bid
        relSpr   = spr / bid if bid else 1.0

        # b) 按价格权重汇总前 DEPTH 档量价
        def _pv(orders: Dict[int, int], reverse=False):
            tot = 0
            for p, q in (sorted(orders.items(), reverse=reverse)[:H.DEPTH]):
                tot += p * q
            return tot

        buy_val  = _pv(od.buy_orders,  True)
        sell_val = _pv(od.sell_orders, False)
        tot_val  = buy_val + sell_val

        buyP = buy_val / tot_val if tot_val else 0.5
        imb  = (buy_val - sell_val) / tot_val if tot_val else 0.0
        return buyP, relSpr, imb, mid

    # ——————————————————— 3‑C.  策略权重自适应 ———————————————————
    def _weight_blend(self) -> Tuple[float, float]:
        """
        根据最近窗口内：
            ‑ mid 价格涨跌（策略 1 偏趋势）
            ‑ 买压变化（策略 2 偏反转）
        输出两个策略的权重 (w1, w2)，和为 1
        """
        win = H.STR_WIN
        if len(self.mid_hist) < win or len(self.buyP_hist) < win:
            return 0.5, 0.5

        r_price   = np.diff(self.mid_hist[-win:])
        r_press   = np.diff(self.buyP_hist[-win:])

        s1 = np.sign(np.mean(r_price))   * abs(np.mean(r_price))
        s2 = -np.sign(np.mean(r_press))  * abs(np.mean(r_press))   # 买压回落时→做空有利

        tot = abs(s1) + abs(s2)
        return (abs(s1) / tot, abs(s2) / tot) if tot else (0.5, 0.5)

    # ——————————————————— 3‑D.  主入口 ———————————————————
    def run(self, state: TradingState):
        """
        组合策略主入口：同时运行趋势和均值回归策略
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

        # 基础市场数据
        bid, ask = max(od.buy_orders), min(od.sell_orders)
        mid = (bid + ask) / 2
        self.mid_hist.append(mid)

        # ① 趋势策略信号
        buyP, relSpr, imb, _ = self._micro_metrics(od)
        self.buyP_hist.append(buyP)
        self.relSpr_hist.append(relSpr)
        self.imb_hist.append(imb)

        price_score = self._price_score(state.observations)
        ma_fast = _ma(self.mid_hist, H.MA_FAST)
        ma_slow = _ma(self.mid_hist, H.MA_SLOW)

        w1, w2 = self._weight_blend()

        sig1_long = price_score > 0.2 and ma_fast > ma_slow
        sig1_short = price_score < -0.2 and ma_fast < ma_slow

        sig2_long = (
            buyP < (1 - H.PRESS_TH) and
            relSpr < H.SPR_REL_TH and
            imb < -H.IMB_TH
        )
        sig2_short = (
            buyP > H.PRESS_TH and
            relSpr < H.SPR_REL_TH and
            imb > H.IMB_TH
        )

        # ② 均值回归信号
        fair = self._fair(state.observations)
        mean_rev_orders = []
        
        if len(self.mid_hist) >= H.MIN_HIST:
            mu, sigma = np.mean(self.mid_hist[-H.PRICE_WIN:]), np.std(self.mid_hist[-H.PRICE_WIN:]) or 1
            z = (mid - mu) / sigma

            if z > H.Z_THRESH and pos > -H.POS_LIM:   # 偏高→卖
                qty = min(H.QTY_MAX, H.POS_LIM + pos)
                mean_rev_orders.append(Order(product, bid, -qty))
            elif z < -H.Z_THRESH and pos < H.POS_LIM:  # 偏低→买
                qty = min(H.QTY_MAX, H.POS_LIM - pos)
                mean_rev_orders.append(Order(product, ask, qty))

        # ③ 做市商策略
        if relSpr > H.MM_SPREAD_MIN and abs(pos) < H.MM_POS_LIM:
            mean_rev_orders.extend([
                Order(product, bid + 1, H.MM_QTY),
                Order(product, ask - 1, -H.MM_QTY)
            ])

        # ④ 组合策略执行
        # 先执行趋势策略
        if ((sig1_long and w1 > H.SIG_TH) or
            (sig2_long and w2 > H.SIG_TH)):
            avail = H.POS_LIM - pos
            if avail > 0:
                orders[product].append(Order(product, ask, min(avail, H.QTY_MAX)))

        elif ((sig1_short and w1 > H.SIG_TH) or
              (sig2_short and w2 > H.SIG_TH)):
            avail = H.POS_LIM + pos
            if avail > 0:
                orders[product].append(Order(product, bid, -min(avail, H.QTY_MAX)))

        # 如果趋势策略没有信号，执行均值回归策略
        elif not orders[product]:
            orders[product].extend(mean_rev_orders)

        # 平仓逻辑保持不变
        elif (
            abs(imb) < 0.1 and (
                relSpr > H.SPR_REL_TH * H.FLAT_SPREAD_MULT or
                abs(price_score - H.FLAT_PSC_NEU) < H.FLAT_PSC_TH
            )
        ):
            if pos > 0:
                orders[product].append(Order(product, bid, -min(pos, H.QTY_MAX)))
            elif pos < 0:
                orders[product].append(Order(product, ask, min(-pos, H.QTY_MAX)))

        return orders, conversions, state.traderData

# trader_meanrev.py
class TraderMeanRev:
    def __init__(self):
        self.mid, self.cost_hist = [], []

    def _fair(self, obs):
        sugar  = getattr(obs, "sugarPrice", 0.0)
        store  = getattr(obs, "storageCost", 0.1)
        self.cost_hist.append(sugar + store)          # 简化：加法近似
        return np.mean(self.cost_hist[-60:])          # 60‑tick 移动均成本

    def run(self, state):
        p="MAGNIFICENT_MACARONS"; od=state.order_depths[p]; pos=state.position.get(p,0)
        orders=[]; c=0
        if not od.buy_orders or not od.sell_orders: return {p:orders},c,state.traderData
        bid,ask=max(od.buy_orders),min(od.sell_orders); mid=(bid+ask)/2
        self.mid.append(mid)

        fair = self._fair(state.observations)
        if len(self.mid)>=20:
            mu, sigma = np.mean(self.mid[-60:]), np.std(self.mid[-60:]) or 1
            z = (mid - mu)/sigma

            if z>1 and pos>-H.POS_LIM:   # 偏高→卖
                qty=min(H.QTY_MAX, H.POS_LIM+pos)
                orders.append(Order(p,bid,-qty))
            elif z<-1 and pos<H.POS_LIM: # 偏低→买
                qty=min(H.QTY_MAX, H.POS_LIM-pos)
                orders.append(Order(p,ask, qty))

        # 做市：价差>0.15% 且仓位轻时，挂双边赚 spread
        relSpr=(ask-bid)/bid
        if relSpr>0.0015 and abs(pos)<15:
            orders+= [
                Order(p,bid+1,  5),
                Order(p,ask-1, -5)
            ]
        return {p:orders},c,state.traderData
