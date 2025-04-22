###############################################################################
#  IMC Prosperity 2025 – Round‑4  统一交易器
#  直接上传此文件中的 Trader 类即可。
###############################################################################
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Any, Tuple
import math, statistics, jsonpickle, numpy as np
from collections import deque, defaultdict

DEBUG = False          # 打开可打印内部日志（注意 3 KB 限制）
WINDOW_MAX = 200       # 全局历史窗口

# ─────────────────────────── 参数区 ───────────────────────────
PARAM = {
    # 做市
    "mm": {
        "base_k": 1.2,         # 从 1.4 降低到 1.2，减小价差
        "soft_liq": 0.40,      # 从 0.45 降低到 0.40，更早开始软清仓
        "hard_liq": 0.85,      # 从 0.90 降低到 0.85，更早开始硬清仓
        "skew_ratio": 0.6,     # 从 0.5 增加到 0.6，增加库存倾斜影响
        "vol_window": 20,      # 添加成交量窗口
        "risk_window": 10      # 添加风险窗口
    },
    # SQUID
    "squid": {
        "z_entry": 2.8,
        "z_exit": 1.2,         # 从 1.6 降低到 1.2，更快速地退出
        "lot": 8
    },
    # Basket
    "basket": {
        "spread_thr": 0.007,
        "max_leg": 40
    },
    # Voucher
    "voucher": {
        "iv_thr_sigma": 1.2,
        "max_lot": 150,
        "trend_window": 20,    # 添加趋势窗口
        "vol_window": 50       # 添加波动率窗口
    },
    # Macarons
    "macarons": {
        "ma_s": 12,
        "ma_l": 60,
        "psi_thr": 0.65,
        "press_thr": 0.72,
        "imb_thr": 0.30,
        "lot": 12,
        "trend_window": 30,    # 添加趋势窗口
        "vol_window": 20       # 添加成交量窗口
    }
}

# 组件映射
COMP_MAP1 = {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1}
COMP_MAP2 = {"CROISSANTS": 4, "JAMS": 2}

STRIKE = {
    "VOLCANIC_ROCK_VOUCHER_9500": 9500,
    "VOLCANIC_ROCK_VOUCHER_9750": 9750,
    "VOLCANIC_ROCK_VOUCHER_10000": 10000,
    "VOLCANIC_ROCK_VOUCHER_10250": 10250,
    "VOLCANIC_ROCK_VOUCHER_10500": 10500
}

POS_LIMIT = {
    # R1
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    # R2
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    # R3
    "VOLCANIC_ROCK": 400, **{k: 200 for k in STRIKE},
    # R4
    "MAGNIFICENT_MACARONS": 75
}

CONV_LIMIT = 10    # macarons conversion / round（占位，如有）

# ─────────────────────────── 工具函数 ─────────────────────────
def mid_price(depth: OrderDepth) -> float | None:
    if depth.buy_orders and depth.sell_orders:
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2
    return None

def best_bid_ask(depth: OrderDepth) -> Tuple[int | None, int | None]:
    bid = max(depth.buy_orders) if depth.buy_orders else None
    ask = min(depth.sell_orders) if depth.sell_orders else None
    return bid, ask

def log(*args):
    if DEBUG:
        print(" ".join(map(str, args)))

# ─────────────────────────── Trader ───────────────────────────
class Trader:
    def __init__(self):
        # 历史缓存
        self.hist: Dict[str, List[float]] = defaultdict(list)
        self.mm_hit: Dict[str, deque] = {p: deque(maxlen=20) for p in ("RAINFOREST_RESIN", "KELP")}
        # voucher 用于波动率
        self.iv_hist: List[float] = []

    # ████████ 1) 改进做市 ────────────────────────────────
    def _mm(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        k = PARAM["mm"]["base_k"]
        limit = POS_LIMIT[p]
        bid, ask = best_bid_ask(depth)
        if bid is None or ask is None:
            return []
        mid = (bid + ask) / 2
        spread = (ask - bid) if ask > bid else 2
        
        # 计算成交量趋势
        vol = sum(abs(q) for q in depth.buy_orders.values()) + sum(abs(q) for q in depth.sell_orders.values())
        self.hist[f"{p}_vol"].append(vol)
        if len(self.hist[f"{p}_vol"]) >= PARAM["mm"]["vol_window"]:
            vol_ma = statistics.mean(self.hist[f"{p}_vol"][-PARAM["mm"]["vol_window"]:])
            vol_ratio = vol / vol_ma if vol_ma > 0 else 1
            k = k * (1 + vol_ratio / 2)  # 根据成交量趋势调整 k
        
        # 计算风险指标
        if len(self.hist[p]) >= PARAM["mm"]["risk_window"]:
            price_std = statistics.stdev(self.hist[p][-PARAM["mm"]["risk_window"]:])
            risk_ratio = price_std / mid if mid > 0 else 0
            k = k * (1 + risk_ratio)  # 根据价格波动调整 k
        
        adj = k * spread
        # 库存倾斜
        skew = (pos / limit) * PARAM["mm"]["skew_ratio"] * spread
        price_buy = int(mid - adj - skew)
        price_sell = int(mid + adj - skew)
        
        # 动态调整仓位大小
        base_size = max(1, limit // 5)
        size = int(base_size * (1 - abs(pos) / limit))  # 根据持仓调整仓位
        
        # 软 / 硬清仓
        hit = abs(pos) >= limit * 0.9
        self.mm_hit[p].append(hit)
        soft = sum(self.mm_hit[p]) / self.mm_hit[p].maxlen >= PARAM["mm"]["soft_liq"]
        hard = all(self.mm_hit[p]) if len(self.mm_hit[p]) == self.mm_hit[p].maxlen else False
        
        orders = []
        if hard:
            # 直接以市价清仓
            if pos > 0:
                orders.append(Order(p, bid, -pos))
            elif pos < 0:
                orders.append(Order(p, ask, -pos))
            return orders
        
        if soft:
            size = size // 2
        
        # 普通做市
        buy_qty = min(size, limit - pos)
        sell_qty = min(size, limit + pos)
        if buy_qty > 0:
            orders.append(Order(p, price_buy, buy_qty))
        if sell_qty > 0:
            orders.append(Order(p, price_sell, -sell_qty))
        return orders

    # ████████ 2) SQUID 双重均值回复 ───────────────────────
    def _squid(self, depth: OrderDepth, pos: int) -> List[Order]:
        p = "SQUID_INK"
        limit = POS_LIMIT[p]
        m = mid_price(depth)
        if m is None:
            return []
        h = self.hist[p]
        h.append(m)
        if len(h) < 40:        # 从 30 增加到 40，使用更长的历史数据
            return []
        ma = statistics.mean(h[-30:])
        sd = statistics.stdev(h[-30:])
        if sd == 0:
            return []
        z = (m - ma) / sd
        orders = []
        entry, exit_ = PARAM["squid"]["z_entry"], PARAM["squid"]["z_exit"]
        lot = PARAM["squid"]["lot"]
        bid, ask = best_bid_ask(depth)
        
        # 添加趋势过滤
        trend_up = statistics.mean(h[-10:]) > statistics.mean(h[-30:])
        
        if z < -entry and pos < limit and trend_up:    # 只在上升趋势时做多
            qty = min(lot, limit - pos)
            orders.append(Order(p, ask, qty))
        elif z > entry and pos > -limit and not trend_up:    # 只在下降趋势时做空
            qty = min(lot, limit + pos)
            orders.append(Order(p, bid, -qty))
        elif abs(z) < exit_:
            if pos > 0:
                orders.append(Order(p, bid, -pos))
            elif pos < 0:
                orders.append(Order(p, ask, -pos))
        return orders

    # ████████ 3) PICNIC 篮子套利 ──────────────────────────
    def _basket(self, state: TradingState, result: Dict[str, List[Order]]):
        d = state.order_depths
        if not all(p in d for p in ("PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES")):
            return
        # 计算组件 mid
        mid_c = mid_price(d["CROISSANTS"])
        mid_j = mid_price(d["JAMS"])
        mid_dj = mid_price(d["DJEMBES"])
        if None in (mid_c, mid_j, mid_dj):
            return
        fv1 = 6*mid_c + 3*mid_j + 1*mid_dj
        fv2 = 4*mid_c + 2*mid_j
        for bask, fv in (("PICNIC_BASKET1", fv1), ("PICNIC_BASKET2", fv2)):
            bid, ask = best_bid_ask(d[bask])
            if bid is None or ask is None: continue
            spread = (ask - bid) / fv
            if spread < PARAM["basket"]["spread_thr"]:
                continue
            # 若篮子贵→卖篮子买组件
            pos_b = state.position.get(bask, 0)
            limit_b = POS_LIMIT[bask]
            if ask > fv * 1.002:
                max_qty = min(PARAM["basket"]["max_leg"], limit_b + pos_b)
                if max_qty > 0:
                    result.setdefault(bask, []).append(Order(bask, ask, -max_qty))
                    # 反向组件腿
                    leg_size = max_qty
                    result.setdefault("CROISSANTS", []).append(Order("CROISSANTS", d["CROISSANTS"].buy_orders and max(d["CROISSANTS"].buy_orders) or int(mid_c), 6*leg_size))
                    result.setdefault("JAMS", []).append(Order("JAMS", d["JAMS"].buy_orders and max(d["JAMS"].buy_orders) or int(mid_j), 3*leg_size if bask=="PICNIC_BASKET1" else 2*leg_size))
                    if bask == "PICNIC_BASKET1":
                        result.setdefault("DJEMBES", []).append(Order("DJEMBES", d["DJEMBES"].buy_orders and max(d["DJEMBES"].buy_orders) or int(mid_dj), 1*leg_size))
            # 反向：篮子便宜→买篮子卖组件
            elif bid < fv * 0.998:
                max_qty = min(PARAM["basket"]["max_leg"], limit_b - pos_b)
                if max_qty > 0:
                    result.setdefault(bask, []).append(Order(bask, bid, max_qty))
                    leg_size = max_qty
                    result.setdefault("CROISSANTS", []).append(Order("CROISSANTS", d["CROISSANTS"].sell_orders and min(d["CROISSANTS"].sell_orders) or int(mid_c), -6*leg_size))
                    result.setdefault("JAMS", []).append(Order("JAMS", d["JAMS"].sell_orders and min(d["JAMS"].sell_orders) or int(mid_j), -3*leg_size if bask=="PICNIC_BASKET1" else -2*leg_size))
                    if bask == "PICNIC_BASKET1":
                        result.setdefault("DJEMBES", []).append(Order("DJEMBES", d["DJEMBES"].sell_orders and min(d["DJEMBES"].sell_orders) or int(mid_dj), -1*leg_size))

    # ████████ 4) Voucher 波面套利 ──────────────────────────
    def _black_scholes(self, S, K, T, σ, call=True):
        if T <= 0 or σ <= 0:
            return max(0, S-K) if call else max(0, K-S)
        d1 = (math.log(S/K) + 0.5*σ*σ*T)/(σ*math.sqrt(T))
        d2 = d1 - σ*math.sqrt(T)
        N = 0.5*(1+math.erf(d1/math.sqrt(2)))
        N2 = 0.5*(1+math.erf(d2/math.sqrt(2)))
        if call:
            return S*N - K*math.exp(-0*T)*N2
        else:
            return K*math.exp(-0*T)*(1-N2) - S*(1-N)
    def _voucher(self, state: TradingState, result: Dict[str, List[Order]]):
        d = state.order_depths
        if "VOLCANIC_ROCK" not in d: return
        S = mid_price(d["VOLCANIC_ROCK"])
        if S is None: return
        self.hist["VOLCANIC_ROCK"].append(S)
        if len(self.hist["VOLCANIC_ROCK"]) < PARAM["voucher"]["vol_window"]: return
        
        # 计算趋势
        short_ma = statistics.mean(self.hist["VOLCANIC_ROCK"][-PARAM["voucher"]["trend_window"]:])
        long_ma = statistics.mean(self.hist["VOLCANIC_ROCK"][-PARAM["voucher"]["vol_window"]:])
        trend_up = short_ma > long_ma
        
        # 计算动态波动率
        ret = np.diff(np.log(self.hist["VOLCANIC_ROCK"][-PARAM["voucher"]["vol_window"]:]))
        σ = np.std(ret) * math.sqrt(252)
        self.iv_hist.append(σ)
        
        # 使用最近10个波动率的平均值
        if len(self.iv_hist) >= 10:
            σ = statistics.mean(self.iv_hist[-10:])
        
        T = 2/252  # 剩 2 天
        for v in STRIKE:
            depth = d.get(v)
            if depth is None: continue
            mid = mid_price(depth)
            if mid is None: continue
            theo = self._black_scholes(S, STRIKE[v], T, σ, call=True)
            dev = (mid - theo)/theo if theo>0 else 0
            thr = PARAM["voucher"]["iv_thr_sigma"]
            pos = state.position.get(v, 0)
            lim = POS_LIMIT[v]
            max_lot = PARAM["voucher"]["max_lot"]
            bid, ask = best_bid_ask(depth)
            
            # 根据趋势调整阈值
            adj_thr = thr * (1.2 if trend_up else 0.8)
            
            if dev > adj_thr and pos > -lim and trend_up:   # 高估且趋势向上 → 卖
                qty = min(max_lot, lim + pos)
                if qty>0: result.setdefault(v, []).append(Order(v, bid, -qty))
                # 对冲基岩
                if state.position.get("VOLCANIC_ROCK",0)+qty <= POS_LIMIT["VOLCANIC_ROCK"]:
                    result.setdefault("VOLCANIC_ROCK", []).append(Order("VOLCANIC_ROCK", bid, qty))
            elif dev < -adj_thr and pos < lim and not trend_up: # 低估且趋势向下 → 买
                qty = min(max_lot, lim - pos)
                if qty>0: result.setdefault(v, []).append(Order(v, ask, qty))
                if state.position.get("VOLCANIC_ROCK",0)-qty >= -POS_LIMIT["VOLCANIC_ROCK"]:
                    result.setdefault("VOLCANIC_ROCK", []).append(Order("VOLCANIC_ROCK", ask, -qty))

    # ████████ 5) Macarons 多因子微观结构 ──────────────────
    def _macarons(self, state: TradingState, depth: OrderDepth, pos: int) -> List[Order]:
        p = "MAGNIFICENT_MACARONS"
        bid, ask = best_bid_ask(depth)
        if bid is None or ask is None: return []
        mid = (bid + ask)/2
        self.hist[p].append(mid)
        # 因子
        sugar = getattr(state.observations, 'sugarPrice', 0)
        sun = getattr(state.observations, 'sunlightIndex', 0)
        tariff = getattr(state.observations, 'importTariff', 0)
        # 价格得分（归一）
        for hkey,val in (("sugarPrice", sugar), ("sunlightIndex", sun), ("importTariff", tariff)):
            self.hist[hkey].append(val)
        
        def norm(val,key):
            h = self.hist[key]
            if len(h)<2 or max(h)==min(h): return 0.5
            return (val-min(h))/(max(h)-min(h))
        
        # 计算动态权重
        sugar_vol = len(self.hist["sugarPrice"])
        sun_vol = len(self.hist["sunlightIndex"])
        tariff_vol = len(self.hist["importTariff"])
        total_vol = sugar_vol + sun_vol + tariff_vol
        
        sugar_weight = 0.5 * (1 + sugar_vol / total_vol)
        sun_weight = 0.3 * (1 + sun_vol / total_vol)
        tariff_weight = 0.2 * (1 + tariff_vol / total_vol)
        
        # 计算趋势
        if len(self.hist[p]) >= PARAM["macarons"]["trend_window"]:
            short_ma = statistics.mean(self.hist[p][-PARAM["macarons"]["trend_window"]//2:])
            long_ma = statistics.mean(self.hist[p][-PARAM["macarons"]["trend_window"]:])
            trend_up = short_ma > long_ma
        else:
            trend_up = True
        
        # 计算成交量
        vol = sum(abs(q) for q in depth.buy_orders.values()) + sum(abs(q) for q in depth.sell_orders.values())
        self.hist[f"{p}_vol"].append(vol)
        if len(self.hist[f"{p}_vol"]) >= PARAM["macarons"]["vol_window"]:
            vol_ma = statistics.mean(self.hist[f"{p}_vol"][-PARAM["macarons"]["vol_window"]:])
            vol_ratio = vol / vol_ma if vol_ma > 0 else 1
        else:
            vol_ratio = 1
        
        price_score = ( norm(sugar,"sugarPrice")*sugar_weight
                       - norm(sun,"sunlightIndex")*sun_weight
                       - norm(tariff,"importTariff")*tariff_weight )
        
        # 根据趋势和成交量调整得分
        if trend_up:
            price_score *= (1 + vol_ratio * 0.2)
        else:
            price_score *= (1 - vol_ratio * 0.2)
        
        # 均线
        ma_s = statistics.mean(self.hist[p][-PARAM["macarons"]["ma_s"]:] )
        ma_l = statistics.mean(self.hist[p][-PARAM["macarons"]["ma_l"]:] ) if len(self.hist[p])>=PARAM["macarons"]["ma_l"] else ma_s
        trend_up = ma_s > ma_l
        # 盘口压强
        buy_press, sell_press = 0, 0
        max_depth = 3  # 考虑前3档
        for i, (pr, q) in enumerate(sorted(depth.buy_orders.items(), reverse=True)):
            if i >= max_depth: break
            weight = (max_depth - i) / max_depth  # 越近的档位权重越大
            buy_press += pr * q * weight
        for i, (pr, q) in enumerate(sorted(depth.sell_orders.items())):
            if i >= max_depth: break
            weight = (max_depth - i) / max_depth
            sell_press += pr * q * weight
        total = buy_press + sell_press if buy_press + sell_press != 0 else 1
        buy_press /= total
        sell_press /= total
        imb = (buy_press - sell_press)
        rel_spread = (ask - bid) / bid
        # 信号
        buy_sig = (price_score > PARAM["macarons"]["psi_thr"] 
                  and trend_up 
                  and sell_press > PARAM["macarons"]["press_thr"] 
                  and imb < -PARAM["macarons"]["imb_thr"]
                  and rel_spread < 0.003)  # 添加价差过滤
        sell_sig = (price_score < -PARAM["macarons"]["psi_thr"] 
                   and (not trend_up) 
                   and buy_press > PARAM["macarons"]["press_thr"] 
                   and imb > PARAM["macarons"]["imb_thr"]
                   and rel_spread < 0.003)  # 添加价差过滤
        orders=[]
        lot = PARAM["macarons"]["lot"]
        limit = POS_LIMIT[p]
        if buy_sig and pos<limit:
            qty = min(lot, limit-pos)
            orders.append(Order(p, ask, qty))
        elif sell_sig and pos>-limit:
            qty = min(lot, limit+pos)
            orders.append(Order(p, bid, -qty))
        # 超宽 spread 平仓
        elif rel_spread > 0.004 and abs(pos)>0:
            if pos>0:
                orders.append(Order(p, bid, -pos))
            else:
                orders.append(Order(p, ask, -pos))
        return orders

    # ████████ 主入口 ──────────────────────────────────────
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, Any]:
        result: Dict[str,List[Order]] = {}
        d = state.order_depths
        # 1) 做市双品
        for p in ("RAINFOREST_RESIN","KELP"):
            if p in d:
                result[p] = self._mm(p, d[p], state.position.get(p,0))
        # 2) SQUID
        if "SQUID_INK" in d:
            result["SQUID_INK"] = self._squid(d["SQUID_INK"], state.position.get("SQUID_INK",0))
        # 3) 篮子套利
        self._basket(state, result)
        # 4) Voucher
        self._voucher(state, result)
        # 5) Macarons
        if "MAGNIFICENT_MACARONS" in d:
            mac_orders = self._macarons(state, d["MAGNIFICENT_MACARONS"], state.position.get("MAGNIFICENT_MACARONS",0))
            if mac_orders: result["MAGNIFICENT_MACARONS"] = mac_orders
        # 保存 traderData（仅价格历史，压缩）
        trader_data = jsonpickle.encode({"len":len(self.hist.get("VOLCANIC_ROCK",[]))})  # 占位
        return result, 0, trader_data
