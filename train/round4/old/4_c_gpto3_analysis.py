# ---------- Strategy C ----------
from datamodel import Order, TradingState
import numpy as np

class Trader:
    """Trade tariff & sugar shock events; 无典型 ε 回归"""
    PARAMS = {
        'pos_limit': 75,
        'event_size': 20,  # 每次冲击下注手数
        'tariff_coeff': 60,  # 经验价差≈ΔTariff*coeff
        'sugar_jump': 1.0,   # ΔSugarPrice 单位跳
    }

    def __init__(self):
        self.position = 0
        self.last_export = None
        self.last_sugar  = None

    def serialize_state(self):
        """将状态转换为可序列化的字典"""
        return {
            'position': self.position,
            'last_export': float(self.last_export) if self.last_export is not None else None,
            'last_sugar': float(self.last_sugar) if self.last_sugar is not None else None
        }

    def run(self, state: TradingState):
        obs  = state.observations
        book = state.order_depths['MAGNIFICENT_MACARONS']
        
        if not book.buy_orders or not book.sell_orders:
            return [], None, self.serialize_state()
            
        bb = max(book.buy_orders.keys())
        ba = min(book.sell_orders.keys())

        orders = []
        # ---- export tariff shock ----
        current_export = getattr(obs, "exportTariff", 0.0)
        current_sugar = getattr(obs, "sugarPrice", 0.0)
        
        if self.last_export is not None:
            delta_tariff = current_export - self.last_export
            if abs(delta_tariff) > 1e-6:
                impact = delta_tariff * self.PARAMS['tariff_coeff']
                lots   = int(np.sign(impact) * self.PARAMS['event_size'])
                # 正 ΔTariff → 做多
                price  = bb + 1 if lots > 0 else ba - 1
                orders.append(Order('MAGNIFICENT_MACARONS', price, lots))
                self.position += lots

        # ---- sugar price shock ----
        if self.last_sugar is not None:
            delta_sugar = current_sugar - self.last_sugar
            if abs(delta_sugar) > self.PARAMS['sugar_jump']:
                lots = int(np.sign(delta_sugar) * self.PARAMS['event_size'])
                # 糖价↑ → 成本↑ → 做多
                price = bb + 1 if lots > 0 else ba - 1
                orders.append(Order('MAGNIFICENT_MACARONS', price, lots))
                self.position += lots

        # ---- wind‑down: 平仓过夜 ----
        if state.timestamp > 99000 and self.position != 0:
            px = ba if self.position > 0 else bb
            orders.append(Order('MAGNIFICENT_MACARONS', px, -self.position))
            self.position = 0

        self.last_export = current_export
        self.last_sugar  = current_sugar
        return orders, None, self.serialize_state()
