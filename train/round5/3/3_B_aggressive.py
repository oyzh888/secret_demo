from typing import Dict, List, Tuple
from datamodel import OrderDepth, TradingState, Order
import jsonpickle, statistics, math
from collections import defaultdict

########## —— ① 静态配置 —— ##########
# 位置限制（官方）
LIMIT = {  # 仅列出比赛规定的 product
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75
}

# —— 运行时参数 - B-aggressive: streak size ×2、阈值降到 2
PARAM_DEFAULT = {
    # ▸ Passive MM
    "mm_spread": 1,              # 做市基础半边价差（ticks）
    "mm_size_frac": 0.20,        # 单边挂单数量 = limit × mm_size_frac
    "mm_vol_k": 1.0,             # 波动 × k → 额外外扩价差
    # ▸ Panic 出清
    "panic_ratio": 0.80,
    "panic_extra": 4,            # 触发后再外扩 ticks
    # ▸ Streak 监听
    "streak_targets": [          # 监听哪些 product 做顺势
        "CROISSANTS", "KELP", "VOLCANIC_ROCK"
    ],
    "counterparties": {          # 每个对手的默认阈值 & 下单比例
        "Camilla": {"thr": 2, "size_frac": 0.20},  # 更激进：size_frac 从 0.10 增加到 0.20
        "Caesar":  {"thr": 2, "size_frac": 0.20}   # 更激进：size_frac 从 0.10 增加到 0.20
    },
    "mean_revert_after": True    # 平均回转： streak 结束后立刻反手挂被动
}

########## —— ② 工具函数 —— ##########
def best_bid_ask(depth: OrderDepth) -> Tuple[int | None, int | None]:
    best_bid = max(depth.buy_orders) if depth.buy_orders else None
    best_ask = min(depth.sell_orders) if depth.sell_orders else None
    return best_bid, best_ask

def tick_spread(mid: float, vol: float, base: int, k: float) -> int:
    """动态价差：base + k · vol（向上取整、至少 1）"""
    return max(1, int(math.ceil(base + k * vol)))

########## —— ③ Trader 类 —— ##########
class Trader:

    def __init__(self):
        # run‑time 可变数据（会序列化到 traderData）
        self.state = {
            "last_side": defaultdict(lambda: None),   # 对手 → BUY/SELL
            "streak":    defaultdict(int),            # 当前连续次数
            "prices":    defaultdict(list)            # product → mid 历史
        }

    # —————— 3.1 更新 streak 统计 ——————
    def _update_streaks(self, market_trades, params):
        for prod, trades in market_trades.items():
            for t in trades:
                cp = t.buyer if t.buyer not in (None, "SUBMISSION") else t.seller
                if cp not in params["counterparties"]:
                    continue
                side = "BUY" if cp == t.buyer else "SELL"
                last = self.state["last_side"][cp]
                if side == last:
                    self.state["streak"][cp] += 1
                else:
                    self.state["streak"][cp] = 1
                    self.state["last_side"][cp] = side

    # —————— 3.2 做市模块 ——————
    def _passive_mm(self, p, depth, pos, params, orders):
        mid = self._mid(depth)
        if mid is None:
            return
        # 记录 mid 用于波动估计
        self.state["prices"][p].append(mid)
        hist = self.state["prices"][p]
        vol = statistics.stdev(hist[-15:]) if len(hist) >= 15 else 1.0

        spr = tick_spread(mid, vol, params["mm_spread"], params["mm_vol_k"])
        buy_px, sell_px = int(mid - spr), int(mid + spr)
        size = max(1, int(LIMIT[p] * params["mm_size_frac"]))

        # panic 强平：仓位占比过高时外扩价差、增大挂单量
        if abs(pos) >= LIMIT[p] * params["panic_ratio"]:
            buy_px -= params["panic_extra"]
            sell_px += params["panic_extra"]
            size = max(size, abs(pos) // 2)

        if pos < LIMIT[p]:
            orders.append(Order(p, buy_px, min(size, LIMIT[p] - pos)))
        if pos > -LIMIT[p]:
            orders.append(Order(p, sell_px, -min(size, LIMIT[p] + pos)))

    # —————— 3.3 streak‑Momentum 模块 ——————
    def _streak_momentum(self, p, depth, pos, params, orders):
        best_bid, best_ask = best_bid_ask(depth)
        if best_bid is None or best_ask is None:
            return
        mid = (best_bid + best_ask) / 2

        for cp, cfg in params["counterparties"].items():
            thr = cfg["thr"]
            if self.state["streak"][cp] < thr:
                continue
            side = self.state["last_side"][cp]  # 该 streak 的方向
            qty = max(1, int(LIMIT[p] * cfg["size_frac"]))
            # —— 顺势吃单 —— 
            if side == "BUY" and pos < LIMIT[p]:
                px, available = best_ask, abs(depth.sell_orders[best_ask])
                take = min(qty, available, LIMIT[p] - pos)
                if take:
                    orders.append(Order(p, px, take))
            elif side == "SELL" and pos > -LIMIT[p]:
                px, available = best_bid, depth.buy_orders[best_bid]
                take = min(qty, available, LIMIT[p] + pos)
                if take:
                    orders.append(Order(p, px, -take))

            # —— streak 结束 → 反手做市 ——
            if params["mean_revert_after"] and self.state["streak"][cp] == thr:
                # 只挂反向被动，不立即平仓，以赚回转差价
                spr = params["mm_spread"] + 1
                if side == "BUY" and pos + qty <= LIMIT[p]:
                    orders.append(Order(p, int(mid + spr), -qty))
                elif side == "SELL" and pos - qty >= -LIMIT[p]:
                    orders.append(Order(p, int(mid - spr), qty))

    # —————— 3.4 小工具 ——————
    @staticmethod
    def _mid(depth):
        b, a = best_bid_ask(depth)
        return (b + a) / 2 if b is not None and a is not None else None

    # —————— 3.5 入口 ——————
    def run(self, state: TradingState):
        # —— 反序列化参数 & 持久化数据 ——
        params = PARAM_DEFAULT.copy()
        if state.traderData:
            loaded = jsonpickle.decode(state.traderData)
            self.state.update(loaded.get("persist", {}))

        # —— 更新 streak 计数 ——
        self._update_streaks(state.market_trades, params)

        results: Dict[str, List[Order]] = {}
        for prod, depth in state.order_depths.items():
            orders: List[Order] = []
            pos = state.position.get(prod, 0)

            # ① 被动做市（全部 product）
            self._passive_mm(prod, depth, pos, params, orders)

            # ② 顺势策略（限定 target product）
            if prod in params["streak_targets"]:
                self._streak_momentum(prod, depth, pos, params, orders)

            if orders:
                results[prod] = orders

        # —— 持久化 —— 
        traderData = jsonpickle.encode({
            "persist": self.state
        })

        return results, 0, traderData
