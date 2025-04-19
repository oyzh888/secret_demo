# -*- coding: utf-8 -*-
"""
IMC Prosperity – Round 4
Aggressive multi‑variant Trader implementation for MAGNIFICENT_MACARONS
Variants:
  • FACTOR_EMA      – 基于因子估值快速 EMA 回归
  • HFT_MOMENTUM    – 超短周期动量＋均线交叉高频满仓
  • ARB_IMBALANCE   – 跨市场（Conversion）套利 + 盘口不平衡吃单
选择激进策略：修改 ACTIVE_STRATEGY = " ..." 行即可
"""

from collections import deque
from typing import Dict, List, Any
import math

# === 比赛框架自带 ===
from datamodel import OrderDepth, TradingState, Order, ConversionObservation

# --------------------------------------------------
ACTIVE_STRATEGY = "ARB_IMBALANCE"       # 选 "FACTOR_EMA" / "HFT_MOMENTUM" / "ARB_IMBALANCE"
PRODUCT = "MAGNIFICENT_MACARONS"
# --------------------------------------------------

# 全局参数（所有策略共享，可按需调整）
PARAMS = {
    "position_limit": 75,
    "max_trade_qty": 25,          # 单笔下单量 (激进, 靠近上限)
    "tick_size": 1,              # 最小报价间隔 (若未知可设1)
    # === FACTOR_EMA ===
    "ema_alpha": 0.35,           # EMA 快速权重
    "fv_threshold": 0.002,       # (FairValue-市场价)/价格 触发阈值 (0.2%)
    "factor_weights": {          # 估值模型权重 (自行根据经验/拟合调整)
        "sugarPrice": 1.0,
        "sunlightIndex": -0.5,   # 日照多→供应↑→价格下行, 因此负权
        "importTariff": 0.8,
        "shippingCost": 0.6,
        "storageCost": 0.4,
    },
    # === HFT_MOMENTUM ===
    "ma_short_window": 3,
    "ma_long_window": 7,
    "momentum_qty_frac": 1.0,    # 信号触发时持仓占用比 (1.0=满仓)
    "momentum_cooldown": 0,      # 连续交易冷却 tick, 激进设0
    # === ARB_IMBALANCE ===
    "arb_fee": 0.1,              # 每只 Macaron 转换固定费用 (示例)
    "imb_ratio_trigger": 0.20,   # (买-卖)/总 > 20% 触发
    "imb_eat_levels": 3,         # 吃掉对手盘前几档
}
# --------------------------------------------------


class _BaseTrader:
    """公共工具与状态"""
    def __init__(self):
        self.pos_limit = PARAMS["position_limit"]
        self.position = 0

    # ---------- 基础工具 ----------
    @staticmethod
    def mid_price(od: OrderDepth) -> float | None:
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2

    @staticmethod
    def best_bid_ask(od: OrderDepth) -> tuple[int, int]:
        bid = max(od.buy_orders) if od.buy_orders else None
        ask = min(od.sell_orders) if od.sell_orders else None
        return bid, ask

    # ---------- 下单助手 ----------
    def _build_orders(
        self,
        od: OrderDepth,
        side: str,
        qty: int,
    ) -> List[Order]:
        bid, ask = self.best_bid_ask(od)
        if side == "BUY" and ask is not None:
            return [Order(PRODUCT, ask, +qty)]
        if side == "SELL" and bid is not None:
            return [Order(PRODUCT, bid, -qty)]
        return []

    # ---------- 持仓管理 ----------
    def _available_to_buy(self) -> int:
        return max(0, self.pos_limit - self.position)

    def _available_to_sell(self) -> int:
        return max(0, self.pos_limit + self.position)


# ==================================================
# ===============  策略 1：FACTOR_EMA  ==============
# ==================================================
class FactorEMATrader(_BaseTrader):
    """快速因子‑EMA 估值回归 + 满仓交易"""
    def __init__(self):
        super().__init__()
        # EMA 状态
        self.factor_ema: Dict[str, float] = {}
        self.price_ema = None

    # --- 估值 ---
    def _update_ema(self, name: str, value: float):
        alpha = PARAMS["ema_alpha"]
        if name not in self.factor_ema or self.factor_ema[name] is None:
            self.factor_ema[name] = value
        else:
            self.factor_ema[name] = alpha * value + (1 - alpha) * self.factor_ema[name]

    def _fair_value(self) -> float | None:
        if self.price_ema is None:
            return None
        fv = 0
        for k, w in PARAMS["factor_weights"].items():
            v = self.factor_ema.get(k)
            if v is None:
                return None
            fv += w * v
        return fv

    # --- 主逻辑 ---
    def run(self, state: TradingState):
        od = state.order_depths.get(PRODUCT)
        if od is None:
            return {}, 0, state.traderData

        # 更新持仓
        self.position = state.position.get(PRODUCT, 0)

        # 获取基本因子观测
        obs = state.observations
        for attr in PARAMS["factor_weights"]:
            self._update_ema(attr, getattr(obs, attr, 0.0))

        # 更新价格 EMA
        mp = self.mid_price(od)
        if mp is not None:
            alpha = PARAMS["ema_alpha"]
            self.price_ema = mp if self.price_ema is None else alpha * mp + (1 - alpha) * self.price_ema

        fv = self._fair_value()
        if fv is None or mp is None:
            return {}, 0, state.traderData

        diff_ratio = (fv - mp) / mp
        orders: List[Order] = []

        # --- 触发买 / 卖 ---
        if diff_ratio > PARAMS["fv_threshold"]:          # 市价低估 → 买
            qty = min(self._available_to_buy(), PARAMS["max_trade_qty"])
            orders += self._build_orders(od, "BUY", qty)
        elif diff_ratio < -PARAMS["fv_threshold"]:       # 市价高估 → 卖/做空
            qty = min(self._available_to_sell(), PARAMS["max_trade_qty"])
            orders += self._build_orders(od, "SELL", qty)

        return {PRODUCT: orders}, 0, state.traderData


# ==================================================
# ==============  策略 2：HFT_MOMENTUM  =============
# ==================================================
class HFTMomentumTrader(_BaseTrader):
    def __init__(self):
        super().__init__()
        self.prices: deque[float] = deque(maxlen=PARAMS["ma_long_window"])
        self.cooldown = 0  # 连续触发冷却

    def _sma(self, window: int) -> float:
        if len(self.prices) < window:
            return self.prices[-1]
        return sum(list(self.prices)[-window:]) / window

    def run(self, state: TradingState):
        od = state.order_depths.get(PRODUCT)
        if od is None:
            return {}, 0, state.traderData
        self.position = state.position.get(PRODUCT, 0)

        mid = self.mid_price(od)
        if mid is None:
            return {}, 0, state.traderData
        self.prices.append(mid)

        short_ma = self._sma(PARAMS["ma_short_window"])
        long_ma = self._sma(PARAMS["ma_long_window"])

        orders: List[Order] = []
        if self.cooldown > 0:
            self.cooldown -= 1
        else:
            # 均线交叉
            if short_ma > long_ma:          # 看涨
                qty_target = int(self._available_to_buy() * PARAMS["momentum_qty_frac"])
                if qty_target > 0:
                    orders += self._build_orders(od, "BUY", qty_target)
                    self.cooldown = PARAMS["momentum_cooldown"]
            elif short_ma < long_ma:        # 看跌
                qty_target = int(self._available_to_sell() * PARAMS["momentum_qty_frac"])
                if qty_target > 0:
                    orders += self._build_orders(od, "SELL", qty_target)
                    self.cooldown = PARAMS["momentum_cooldown"]

        return {PRODUCT: orders}, 0, state.traderData


# ==================================================
# ============  策略 3：ARB_IMBALANCE  ==============
# ==================================================
class ArbImbalanceTrader(_BaseTrader):
    def __init__(self):
        super().__init__()
        self.conversion_left = PARAMS["position_limit"]  # 可用兑换额度（示例逻辑）

    # ---- 盘口不平衡检测 ----
    @staticmethod
    def _imbalance_ratio(od: OrderDepth, levels: int) -> float:
        buy_vol = sum(list(od.buy_orders.values())[:levels])
        sell_vol = sum(list(od.sell_orders.values())[:levels])
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        return (buy_vol - sell_vol) / total  # 正→买强，负→卖强

    def run(self, state: TradingState):
        od = state.order_depths.get(PRODUCT)
        if od is None:
            return {}, 0, state.traderData
        self.position = state.position.get(PRODUCT, 0)

        # ---------- 跨市场套利 ----------
        conversions = 0
        orders: List[Order] = []

        conv_obs: ConversionObservation | None = state.observations.conversionObservations.get(PRODUCT) \
            if hasattr(state.observations, "conversionObservations") else None

        if conv_obs:
            bid_ext = conv_obs.bidPrice - PARAMS["arb_fee"]
            ask_ext = conv_obs.askPrice + PARAMS["arb_fee"]
            bid_int, ask_int = self.best_bid_ask(od)

            # 外部买价高 → 先买内盘再外盘卖 (出口)
            if bid_int is not None and bid_ext - bid_int > 0:
                qty = min(self._available_to_buy(), PARAMS["max_trade_qty"], self.conversion_left)
                if qty > 0:
                    orders += [Order(PRODUCT, ask_int or bid_int, qty)]
                    conversions += qty
                    self.conversion_left -= qty

            # 外部卖价低 → 先外盘买再内盘卖 (进口)
            elif ask_int is not None and ask_int - ask_ext > 0:
                qty = min(self._available_to_sell(), PARAMS["max_trade_qty"], self.conversion_left)
                if qty > 0:
                    orders += [Order(PRODUCT, bid_int or ask_int, -qty)]
                    conversions += qty
                    self.conversion_left -= qty

        # ---------- 订单簿不平衡 ----------
        imb = self._imbalance_ratio(od, PARAMS["imb_eat_levels"])
        if imb > PARAMS["imb_ratio_trigger"]:      # 买盘占优 → 推高
            qty = min(self._available_to_buy(), PARAMS["max_trade_qty"])
            orders += self._build_orders(od, "BUY", qty)
        elif imb < -PARAMS["imb_ratio_trigger"]:   # 卖盘占优 → 砸盘
            qty = min(self._available_to_sell(), PARAMS["max_trade_qty"])
            orders += self._build_orders(od, "SELL", qty)

        return {PRODUCT: orders}, conversions, state.traderData


# ==================================================
# ================= 主 Trader 入口 =================
# ==================================================
class Trader:
    """
    统一对接比赛引擎的 Trader
    改动 ACTIVE_STRATEGY 即可切换
    """
    def __init__(self):
        if ACTIVE_STRATEGY == "FACTOR_EMA":
            self.trader = FactorEMATrader()
        elif ACTIVE_STRATEGY == "HFT_MOMENTUM":
            self.trader = HFTMomentumTrader()
        elif ACTIVE_STRATEGY == "ARB_IMBALANCE":
            self.trader = ArbImbalanceTrader()
        else:
            raise ValueError(f"Unknown ACTIVE_STRATEGY: {ACTIVE_STRATEGY}")

    def run(self, state: TradingState):
        return self.trader.run(state)
