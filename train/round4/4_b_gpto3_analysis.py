# ---------- Strategy B ----------
import pickle, os
import numpy as np
from datamodel import Order, TradingState
from sklearn.ensemble import GradientBoostingRegressor

class Trader:
    """GBDT fair‑value + order‑book imbalance scalping"""
    PARAMS = {
        'gb_model_path': 'gbdt_macaron.pkl',   # 先离线训练后序列化
        'eps_sigma': 43.27,
        'spread_mean': 8.20,
        'pos_limit': 75,
        'fast_spread_markup': 0.2,  # 价差占比
        'imb_weight': 0.3
    }

    def __init__(self):
        # 线上载入离线保存好的模型
        with open(self.PARAMS['gb_model_path'], 'rb') as f:
            self.model: GradientBoostingRegressor = pickle.load(f)
        self.position = 0

    # ---------- helpers ----------
    @staticmethod
    def imbalance(book):
        buy_orders = sorted(book.buy_orders.items(), reverse=True)[:3]
        sell_orders = sorted(book.sell_orders.items())[:3]
        tot_bid = sum(b for p, b in buy_orders)
        tot_ask = sum(abs(a) for p, a in sell_orders)  # sell quantities are negative
        return (tot_bid - tot_ask) / max(tot_bid + tot_ask, 1)

    def run(self, state: TradingState):
        obs  = state.observations
        book = state.order_depths['MAGNIFICENT_MACARONS']
        
        if not book.buy_orders or not book.sell_orders:
            return [], None, {}
            
        bb = max(book.buy_orders.keys())
        ba = min(book.sell_orders.keys())
        mid = (bb + ba) / 2

        # 使用 getattr 安全获取属性
        X = np.array([[
            getattr(obs, "sugarPrice", 0.0),
            getattr(obs, "sunlightIndex", 0.0),
            getattr(obs, "exportTariff", 0.0),
            getattr(obs, "importTariff", 0.0),
            getattr(obs, "transportFees", 0.0)
        ]])
        
        fv = self.model.predict(X)[0]
        epsilon = mid - fv
        imb     = self.imbalance(book)

        theta = max(self.PARAMS['spread_mean'],
                    0.7 * self.PARAMS['eps_sigma'])
        # 加入盘口倾斜校正
        epsilon_adj = epsilon - self.PARAMS['imb_weight'] * imb * theta

        target = int(np.clip(-epsilon_adj / theta,
                             -self.PARAMS['pos_limit'],
                              self.PARAMS['pos_limit']))
        diff = target - self.position
        orders = []
        if diff != 0:
            step = int(np.sign(diff) * min(abs(diff), 5))
            price = bb + self.PARAMS['fast_spread_markup'] * (ba - bb) if step > 0 \
                    else ba - self.PARAMS['fast_spread_markup'] * (ba - bb)
            orders.append(Order('MAGNIFICENT_MACARONS', price, step))
            self.position += step

        return orders, None, {}
