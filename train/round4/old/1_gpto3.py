# v_alpha2.py  ——  IMC Prosperity Round‑4 “MAGNIFICENT_MACARONS” 策略
import numpy as np
from typing import Dict, List, Any
from datamodel import OrderDepth, TradingState, Order

# ---------- 全局可调参数 ----------
PARAMS = dict(
    position_limit=75,          # 总仓位上限
    clip_sigma=3.0,             # 超过 k·σ 触发 edge_clip
    base_spread=2.0,            # 公允价两侧基础报价偏移
    quote_volume=10,            # 每档下单量
    regression_window=400,      # OLS 回归窗口 (tick)
    sigma_window=200,           # 历史波动率窗口
    tariff_jump=2,              # importTariff 跳变阈
    sunlight_jump=0.5,          # sunlightIndex 跳变阈
    fast_take_edge=1.5,         # 事件触发时愿意吃到的 edge (×σ)
)

FACTORS = ["sugarPrice", "sunlightIndex", "importTariff"]

class Trader:
    def __init__(self):
        # ===== 运行时缓存 =====
        self.hist = {k: [] for k in ["mid"] + FACTORS}      # 历史序列
        self.beta = np.zeros(len(FACTORS) + 1)              # β 系数 + 截距
        self.sigma = 2.0                                    # mid_price 波动率估计
        self.base_spread = PARAMS["base_spread"]   # ← 先给个默认值
        self.edge_clip = 6.0                                # 绝对最远报价 (会自适应)
    
    # ---------- 工具 ----------
    def _update_regression(self):
        """滚动 OLS 更新 β 系数"""
        if len(self.hist["mid"]) < PARAMS["regression_window"]:
            return
        X = np.array([self.hist[f][-PARAMS["regression_window"]:] for f in FACTORS]).T
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        y = np.array(self.hist["mid"][-PARAMS["regression_window"]:])
        # 普通最小二乘
        self.beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    def _fair_value(self, obs_dict):
        """根据最新观测计算公允价"""
        x = np.array([obs_dict.get(f, 0.0) for f in FACTORS] + [1])
        return float(x @ self.beta)
    
    def _update_sigma(self):
        """滚动估计 σ 并据此调整 edge_clip / base_spread"""
        if len(self.hist["mid"]) < PARAMS["sigma_window"] + 1:
            return
        returns = np.diff(self.hist["mid"][-PARAMS["sigma_window"]:])
        self.sigma = np.std(returns)
        # 自适应
        self.edge_clip = PARAMS["clip_sigma"] * self.sigma
        self.base_spread = max(PARAMS["base_spread"], 0.6 * self.sigma)

    def _event_triggered(self):
        """检查是否出现重大跳变"""
        if len(self.hist["mid"]) < 2:
            return False
        # 简单检测最近一次观测与前一 tick 差值
        d_tariff   = abs(self.hist["importTariff"][-1] - self.hist["importTariff"][-2])
        d_sunlight = abs(self.hist["sunlightIndex"][-1] - self.hist["sunlightIndex"][-2])
        return (d_tariff   >= PARAMS["tariff_jump"]   or
                d_sunlight >= PARAMS["sunlight_jump"])
    
    # ---------- 主逻辑 ----------
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, Any]:
        res: Dict[str, List[Order]] = {}
        conversions = 0
        
        if "MAGNIFICENT_MACARONS" not in state.order_depths:
            return res, conversions, state.traderData
        
        od: OrderDepth = state.order_depths["MAGNIFICENT_MACARONS"]
        # 买一 & 卖一
        if not od.buy_orders or not od.sell_orders:
            return res, conversions, state.traderData
        
        best_bid, best_ask = max(od.buy_orders), min(od.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        
        # -------- 更新历史序列 --------
        self.hist["mid"].append(mid_price)
        obs = state.observations.__dict__ if hasattr(state, "observations") else {}
        for f in FACTORS:
            self.hist[f].append(obs.get(f, self.hist[f][-1] if self.hist[f] else 0))
        
        self._update_regression()
        self._update_sigma()
        
        fv = self._fair_value(obs)
        pos = state.position.get("MAGNIFICENT_MACARONS", 0)
        inv_skew = 0.04 * pos                               # 仓位偏移
        
        # -------- 事件单 --------
        orders: List[Order] = []
        if self._event_triggered():
            # 定向 IOC，抓跳变第一口价
            edge = self.sigma * PARAMS["fast_take_edge"]
            if fv > mid_price + edge and pos < PARAMS["position_limit"]:
                qty = min(PARAMS["quote_volume"], PARAMS["position_limit"] - pos)
                orders.append(Order("MAGNIFICENT_MACARONS", best_ask, qty))   # Buy
            elif fv < mid_price - edge and pos > -PARAMS["position_limit"]:
                qty = min(PARAMS["quote_volume"], pos + PARAMS["position_limit"])
                orders.append(Order("MAGNIFICENT_MACARONS", best_bid, -qty))  # Sell
        
        # -------- 做市双边挂单 --------
        bid_px = max(best_bid, fv - self.base_spread - inv_skew)
        ask_px = min(best_ask, fv + self.base_spread - inv_skew)
        
        # 防止挂到过深位置
        bid_px = max(bid_px, fv - self.edge_clip)
        ask_px = min(ask_px, fv + self.edge_clip)
        
        # 仓位检查
        buy_qty  = min(PARAMS["quote_volume"], PARAMS["position_limit"] - pos)
        sell_qty = min(PARAMS["quote_volume"], pos + PARAMS["position_limit"])
        if buy_qty > 0:
            orders.append(Order("MAGNIFICENT_MACARONS", int(round(bid_px)),  buy_qty))
        if sell_qty > 0:
            orders.append(Order("MAGNIFICENT_MACARONS", int(round(ask_px)), -sell_qty))
        
        res["MAGNIFICENT_MACARONS"] = orders
        return res, conversions, state.traderData
