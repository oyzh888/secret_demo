# ---------- Strategy A ----------
import numpy as np
from datamodel import Order, TradingState
import json

class Trader:
    """Baseline: Ridge fair‑value -> ε 回归做市"""
    PARAMS = {
        # 因子权重 (来自 RidgeCV alpha=0.316)
        'weights': dict(
            intercept=615.2,
            sugarPrice=5.18,
            sugar_lag4=2.46,
            sunlightIndex=-3.04,
            sunlight_lag2=-1.55,
            exportTariff=-59.9,
            importTariff=-49.1,
            transportFees=0.0
        ),
        'eps_sigma': 43.27,        # ε 标准差
        'spread_mean': 8.20,       # 平均盘口 spread
        'half_life': 346,          # ε 半衰期
        'pos_limit': 75,
        'lot_size': 5,             # 最小调仓粒度
        'imb_thresh': 0.6,         # 盘口失衡过滤阈值
    }

    def __init__(self):
        self.position = 0
        self.tick = 0
        self.history = {'sugarPrice': [], 'sunlightIndex': []}

    def serialize_state(self):
        """将状态转换为可序列化的字典"""
        return {
            'position': self.position,
            'tick': self.tick,
            'history': {
                'sugarPrice': list(map(float, self.history['sugarPrice'][-10:])),  # 只保留最近10个数据点
                'sunlightIndex': list(map(float, self.history['sunlightIndex'][-10:]))
            }
        }

    # --------- helper ---------
    def fair_value(self, obs):
        """加权线性 FV（含滞后特征）"""
        w = self.PARAMS['weights']
        # 使用 getattr 安全获取属性，提供默认值
        sugar_price = getattr(obs, "sugarPrice", 0.0)
        sunlight_index = getattr(obs, "sunlightIndex", 0.0)
        export_tariff = getattr(obs, "exportTariff", 0.0)
        import_tariff = getattr(obs, "importTariff", 0.0)
        transport_fees = getattr(obs, "transportFees", 0.0)
        
        sugar_lag4 = self.history['sugarPrice'][-4] if len(self.history['sugarPrice']) >= 4 else sugar_price
        sunlight_lag2 = self.history['sunlightIndex'][-2] if len(self.history['sunlightIndex']) >= 2 else sunlight_index
        
        fv = (w['intercept'] +
              w['sugarPrice']      * sugar_price +
              w['sugar_lag4']      * sugar_lag4 +
              w['sunlightIndex']   * sunlight_index +
              w['sunlight_lag2']   * sunlight_lag2 +
              w['exportTariff']    * export_tariff +
              w['importTariff']    * import_tariff +
              w['transportFees']   * transport_fees)
        return fv

    # --------- main loop ---------
    def run(self, state: TradingState):
        o   = state.observations
        bok = state.order_depths['MAGNIFICENT_MACARONS']
        
        # 获取最优买卖价
        if not bok.buy_orders or not bok.sell_orders:
            return [], None, self.serialize_state()
            
        best_bid = max(bok.buy_orders.keys())
        best_ask = min(bok.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        # ---- feature history update ----
        self.history['sugarPrice'].append(getattr(o, "sugarPrice", 0.0))
        self.history['sunlightIndex'].append(getattr(o, "sunlightIndex", 0.0))

        fv        = self.fair_value(o)
        epsilon   = mid - fv
        spread    = best_ask - best_bid
        theta     = max(0.9 * self.PARAMS['spread_mean'],
                        0.8 * self.PARAMS['eps_sigma'])

        # ---- position target ----
        target_pos = int(np.clip(-epsilon / theta,
                                 -self.PARAMS['pos_limit'],
                                  self.PARAMS['pos_limit']))
        delta = target_pos - self.position
        step  = int(np.sign(delta) *
                    min(abs(delta), self.PARAMS['lot_size']))

        orders = []
        if step != 0:
            if step > 0:
                price = best_bid + 0.5
                orders.append(Order('MAGNIFICENT_MACARONS', price,  step))
            else:
                price = best_ask - 0.5
                orders.append(Order('MAGNIFICENT_MACARONS', price,  step))
            self.position += step

        # ---- aging exit ----
        if self.tick - getattr(self, 'entry_tick', 0) > 2 * self.PARAMS['half_life']:
            if self.position != 0:
                px = best_ask if self.position > 0 else best_bid
                orders.append(Order('MAGNIFICENT_MACARONS', px, -self.position))
                self.position = 0

        self.tick += 1
        return orders, None, self.serialize_state()
