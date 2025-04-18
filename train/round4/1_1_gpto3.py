# trader_mix.py  ——  IMC Round‑4 多策略融合
from datamodel import OrderDepth, TradingState, Order, Trade
from typing import Dict, List, Any
import jsonpickle, numpy as np, math, statistics

# ------------- 全局参数，可自由调试 ----------------
PARAMS = dict(
    # 通用
    position_limit       = 75,
    quote_volume         = 10,
    regression_window    = 400,
    sigma_window         = 200,
    base_spread_default  = 2.0,
    clip_sigma           = 3.0,

    # VolKiller
    VaR_limit_pct        = 0.20,   # 占账户权益 %
    drawdown_cap         = 30000,  # seashell
    stop_sigma           = 4.0,    # 累计回撤 σ

    # Stat‑Arb
    lead_lag             = 300,
    z_entry              = 1.5,
    z_exit               = 0.2,
    pair_volume          = 15,
)

PRODUCT = "MAGNIFICENT_MACARONS"
LEAD_PROD = "RAINFOREST_RESIN"     # 可替换为 KELP 等

# ---------------------------------------------------
FACTORS = ["sugarPrice", "sunlightIndex", "importTariff"]


# ========== 1. 改进 Market‑Maker ==========
class BaseMM:
    def __init__(self):
        self.hist = {k: [] for k in ["mid"] + FACTORS}
        self.beta = np.zeros(len(FACTORS) + 1)
        self.sigma = 2.0
        self.base_spread = PARAMS["base_spread_default"]
        self.edge_clip = 6.0

    # --- 工具 ---
    def _update_reg(self):
        if len(self.hist["mid"]) < PARAMS["regression_window"]: return
        X = np.array([self.hist[f][-PARAMS["regression_window"]:] for f in FACTORS]).T
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        y = np.array(self.hist["mid"][-PARAMS["regression_window"]:])
        self.beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    def _update_sigma(self):
        if len(self.hist["mid"]) < PARAMS["sigma_window"] + 1: return
        ret = np.diff(self.hist["mid"][-PARAMS["sigma_window"]:])
        self.sigma = np.std(ret) if len(ret) else 2.0
        self.edge_clip  = PARAMS["clip_sigma"] * self.sigma
        self.base_spread = max(PARAMS["base_spread_default"], 0.6 * self.sigma)

    def fair_value(self, obs):
        x = np.array([obs.get(f, 0.0) for f in FACTORS] + [1])
        return float(x @ self.beta)

    # --- 生成订单 ---
    def run(self, state: TradingState):
        if PRODUCT not in state.order_depths: return {}, 0, ""
        od: OrderDepth = state.order_depths[PRODUCT]
        if not od.buy_orders or not od.sell_orders: return {}, 0, ""

        best_bid, best_ask = max(od.buy_orders), min(od.sell_orders)
        mid = (best_bid + best_ask) / 2
        obs = state.observations.conversionObservations.get(PRODUCT, None)
        obs_dict = obs.__dict__ if obs else {}

        # 更新历史
        self.hist["mid"].append(mid)
        for f in FACTORS:
            self.hist[f].append(obs_dict.get(f, self.hist[f][-1] if self.hist[f] else 0))
        self._update_reg()
        self._update_sigma()

        fv   = self.fair_value(obs_dict)
        pos  = state.position.get(PRODUCT, 0)
        skew = 0.04 * pos

        bid_px = max(best_bid, fv - self.base_spread - skew)
        ask_px = min(best_ask, fv + self.base_spread - skew)
        bid_px = max(bid_px, fv - self.edge_clip)
        ask_px = min(ask_px, fv + self.edge_clip)

        buy_qty  = min(PARAMS["quote_volume"],  PARAMS["position_limit"] - pos)
        sell_qty = min(PARAMS["quote_volume"],  pos + PARAMS["position_limit"])

        orders = []
        if buy_qty  > 0: orders.append(Order(PRODUCT, int(round(bid_px)),  buy_qty))
        if sell_qty > 0: orders.append(Order(PRODUCT, int(round(ask_px)), -sell_qty))
        return {PRODUCT: orders}, 0, ""


# ========== 2. Vol‑Killer ==========
class VolKiller:
    def __init__(self):
        self.equity = 0.0
        self.max_equity = 0.0
        self.drawdown = 0.0
        self.hist_equity = []
        self.hist_sigma  = []

    def run(self, state: TradingState):
        # 更新权益（简单累加 own_trades）
        for trades in state.own_trades.get(PRODUCT, []):
            self.equity += -trades.price * trades.quantity
        self.max_equity = max(self.max_equity, self.equity)
        self.drawdown = self.max_equity - self.equity

        # 更新 σ 估计
        mid = None
        od = state.order_depths.get(PRODUCT, None)
        if od and od.buy_orders and od.sell_orders:
            mid = (max(od.buy_orders) + min(od.sell_orders)) / 2
        if mid is not None:
            self.hist_sigma.append(mid)
            if len(self.hist_sigma) > PARAMS["sigma_window"]:
                self.hist_sigma.pop(0)
        sigma = statistics.stdev(np.diff(self.hist_sigma)) if len(self.hist_sigma) > 2 else 2.0

        # 触发止损？
        stop = (self.drawdown > PARAMS["drawdown_cap"]) or (self.drawdown > PARAMS["stop_sigma"] * sigma)
        orders = []
        if stop:
            pos = state.position.get(PRODUCT, 0)
            if pos != 0 and od and od.buy_orders and od.sell_orders:
                mkt_price = min(od.sell_orders) if pos < 0 else max(od.buy_orders)
                orders.append(Order(PRODUCT, mkt_price, -pos))  # 市价反方向平仓
        return ({PRODUCT: orders} if orders else {}), 0, ""


# ========== 3. 统计套利 ==========
class StatArbPair:
    def __init__(self):
        self.hist_mac, self.hist_lead = [], []
        self.in_trade = False
        self.entry_side = 0   # +1 = long MAC, -1 = short MAC

    def run(self, state: TradingState):
        od_mac  = state.order_depths.get(PRODUCT, None)
        od_lead = state.order_depths.get(LEAD_PROD, None)
        if not od_mac or not od_lead: return {}, 0, ""

        # 取 mid
        mac_mid  = (max(od_mac.buy_orders)  + min(od_mac.sell_orders))  / 2
        lead_mid = (max(od_lead.buy_orders) + min(od_lead.sell_orders)) / 2
        self.hist_mac.append(mac_mid)
        self.hist_lead.append(lead_mid)
        if len(self.hist_mac) <= PARAMS["lead_lag"]: return {}, 0, ""

        # 计算 Z‑score
        lead_shift = self.hist_lead[-PARAMS["lead_lag"]]
        beta = np.cov(self.hist_mac, self.hist_lead)[0,1] / np.var(self.hist_lead)
        spread = mac_mid - beta * lead_shift
        spreads = np.array(self.hist_mac[-500:]) - beta * np.array(self.hist_lead[-500:])
        z = (spread - spreads.mean()) / (spreads.std() + 1e-6)

        orders = []
        pos    = state.position.get(PRODUCT, 0)
        qty    = PARAMS["pair_volume"]

        if not self.in_trade and abs(z) > PARAMS["z_entry"]:
            # 开仓
            if z > 0 and pos > -PARAMS["position_limit"]:
                # mac 高估 → 做空
                orders.append(Order(PRODUCT, max(od_mac.buy_orders), -qty))
                self.entry_side, self.in_trade = -1, True
            elif z < 0 and pos < PARAMS["position_limit"]:
                # mac 低估 → 做多
                orders.append(Order(PRODUCT, min(od_mac.sell_orders), qty))
                self.entry_side, self.in_trade = 1, True

        elif self.in_trade and abs(z) < PARAMS["z_exit"]:
            # 平仓
            if self.entry_side == 1 and pos > 0:
                orders.append(Order(PRODUCT, max(od_mac.buy_orders), -min(qty, pos)))
            elif self.entry_side == -1 and pos < 0:
                orders.append(Order(PRODUCT, min(od_mac.sell_orders), min(qty, -pos)))
            self.in_trade = False

        return ({PRODUCT: orders} if orders else {}), 0, ""


# ========== 主融合 Trader ==========
class Trader:
    def __init__(self):
        self.mm   = BaseMM()
        self.vk   = VolKiller()
        self.pair = StatArbPair()

    def run(self, state: TradingState):
        res_mm,  conv_mm,  _ = self.mm.run(state)
        res_vk,  conv_vk,  _ = self.vk.run(state)
        res_pair,conv_pair,_ = self.pair.run(state)

        # 合并订单
        result: Dict[str, List[Order]] = {}
        for d in (res_mm, res_vk, res_pair):
            for prod, od_list in d.items():
                result.setdefault(prod, []).extend(od_list)

        conversions = max(conv_mm, conv_vk, conv_pair)
        # traderData 此例暂不用，可存 jsonpickle.dumps(...) 自定义状态
        return result, conversions, ""
