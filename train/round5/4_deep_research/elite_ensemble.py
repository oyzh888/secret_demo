from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from datamodel import OrderDepth, Order, TradingState, Trade

# Shared utils
POSITION_LIMITS: Dict[str, int] = {
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
    "SQUID_INK": 50,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,
    "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200,
    "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75,
}

def clamp(qty: int, limit: int) -> int:
    """Ensure resulting position stays within ±limit."""
    return max(-limit, min(limit, qty))

def best_bid_ask(depth: OrderDepth) -> Tuple[int | None, int | None, int | None, int | None]:
    bid_p = bid_q = ask_p = ask_q = None
    if depth.buy_orders:
        bid_p, bid_q = max(depth.buy_orders.items(), key=lambda x: x[0])
    if depth.sell_orders:
        ask_p, ask_q = min(depth.sell_orders.items(), key=lambda x: x[0])
    return bid_p, bid_q, ask_p, ask_q

class EliteEnsembleTrader:
    """Elite Ensemble combining best aspects of all strategies."""
    
    TAG = "EE"
    CSI = 750  # Optimal sunlight index for macarons
    VOUCHERS = [
        "VOLCANIC_ROCK_VOUCHER_9500",
        "VOLCANIC_ROCK_VOUCHER_9750",
        "VOLCANIC_ROCK_VOUCHER_10000",
        "VOLCANIC_ROCK_VOUCHER_10250",
        "VOLCANIC_ROCK_VOUCHER_10500",
    ]
    COMP_MAP = {
        "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
        "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2},
    }
    # 风险分类 - 用于动态调整风险评估
    LOW_RISK_PRODUCTS = {"MAGNIFICENT_MACARONS", "PICNIC_BASKET1", "PICNIC_BASKET2"}
    HIGH_RISK_PRODUCTS = {"VOLCANIC_ROCK"} | set(VOUCHERS)
    
    def __init__(self):
        self.last_mid: float | None = None
        self.streak: Dict[str, Tuple[str | None, int]] = {}
        self.last_prices: Dict[str, float] = {}
        self.market_trends: Dict[str, float] = {}  # 追踪市场趋势
        self.day_volatility: float = 0.0  # 估计当天市场波动
        self.iteration: int = 0  # 跟踪交易日进度
        self.position_history: Dict[str, List[int]] = {}  # 跟踪历史头寸
        self.price_history: Dict[str, List[float]] = {}  # 跟踪价格历史
        self.high_vol_detected: bool = False  # 高波动标记
        
    def _estimate_market_state(self, state: TradingState):
        """估计整体市场状态和波动性"""
        self.iteration += 1
        
        # 分析市场波动
        volatility_samples = []
        high_vol_products = 0
        total_products = 0
        
        for product, depth in state.order_depths.items():
            bid, _, ask, _ = best_bid_ask(depth)
            if bid is None or ask is None:
                continue
                
            mid = (bid + ask) / 2
            last_price = self.last_prices.get(product)
            
            # 更新价格历史
            if product not in self.price_history:
                self.price_history[product] = []
            if len(self.price_history[product]) > 30:  # 保留最近30个价格
                self.price_history[product].pop(0)
            self.price_history[product].append(mid)
            
            # 计算即时波动性
            if last_price:
                price_change = abs(mid - last_price) / last_price
                volatility_samples.append(price_change)
                
                # 检测高波动产品
                total_products += 1
                if price_change > 0.02:  # 2%以上波动视为高波动
                    high_vol_products += 1
                
            self.last_prices[product] = mid
            
            # 更新头寸历史
            pos = state.position.get(product, 0)
            if product not in self.position_history:
                self.position_history[product] = []
            if len(self.position_history[product]) > 10:  # 保留最近10个头寸
                self.position_history[product].pop(0)
            self.position_history[product].append(pos)
            
        # 更新整体波动估计
        if volatility_samples:
            current_vol = np.mean(volatility_samples)
            # 使用指数移动平均更新波动性估计
            self.day_volatility = 0.9 * self.day_volatility + 0.1 * current_vol if self.day_volatility > 0 else current_vol
        
        # 检测市场高波动状态
        if total_products > 0 and (high_vol_products / total_products) > 0.3:
            self.high_vol_detected = True
        else:
            # 波动性持续较低时才重置
            if self.day_volatility < 0.01 and self.iteration > 100:
                self.high_vol_detected = False
        
    def _position_exposure(self, product: str) -> float:
        """计算产品当前头寸利用率，0表示无头寸，1表示满仓"""
        if product not in self.position_history or not self.position_history[product]:
            return 0
            
        pos = self.position_history[product][-1]
        limit = POSITION_LIMITS[product]
        return abs(pos) / limit if limit > 0 else 0
        
    def _price_trend(self, product: str, window: int = 5) -> float:
        """计算产品价格趋势，返回-1到1之间的值，0表示无趋势"""
        if product not in self.price_history or len(self.price_history[product]) < window:
            return 0
            
        prices = self.price_history[product][-window:]
        if prices[0] == 0:
            return 0
            
        # 使用简单线性回归计算趋势
        x = np.arange(len(prices))
        y = np.array(prices)
        A = np.vstack([x, np.ones(len(x))]).T
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # 归一化斜率到-1到1之间
        normalized_slope = m * window / prices[0]
        return max(-1, min(1, normalized_slope * 5))  # 乘5增加敏感度
        
    def _handle_macarons(self, state: TradingState) -> List[Order]:
        """进一步增强马卡龙策略，灵感来自于SunlightSniper"""
        obs = state.observations.conversionObservations.get("MAGNIFICENT_MACARONS")
        if not obs:
            return []
            
        sun = obs.sunlightIndex
        sugar = obs.sugarPrice
        depth = state.order_depths["MAGNIFICENT_MACARONS"]
        bid, bid_q, ask, ask_q = best_bid_ask(depth)
        if bid is None or ask is None:
            return []
            
        # 更精确的公允价值计算
        sun_effect = (self.CSI - sun) * 10
        # 基础价值: 基准价 + 糖价影响 + 阳光效应
        base_fair = 10000 + 2 * sugar + sun_effect
        
        product = "MAGNIFICENT_MACARONS"
        
        # 使用更长期的趋势分析
        short_trend = self._price_trend(product, 5)  # 短期趋势(5个数据点)
        long_trend = self._price_trend(product, 15)  # 长期趋势(15个数据点)
        
        # 趋势组合：短期占70%，长期占30%
        combined_trend = short_trend * 0.7 + long_trend * 0.3
        trend_adj = combined_trend * 120  # 增加趋势权重
        
        # 记录趋势
        self.market_trends[product] = 0.8 * self.market_trends.get(product, 0) + 0.2 * combined_trend
        
        # 综合考虑所有因素的公允价值
        fair = base_fair + trend_adj
        
        # 更新价格
        mid = (bid + ask) / 2
        self.last_prices[product] = mid
        
        # 检查头寸
        pos = state.position.get(product, 0)
        limit = POSITION_LIMITS[product]
        exposure = self._position_exposure(product)
        
        orders = []
        
        # 根据阳光指数和最优值的距离计算信号强度
        sun_distance = abs(self.CSI - sun)
        sun_signal = min(1.0, sun_distance / 120)  # 标准化到0-1
        
        # 动态阈值 - 根据信号强度和趋势强度调整
        base_thresh = 80  # 基础阈值
        trend_intensity = abs(combined_trend)
        
        # 阈值调整：信号越强，阈值越小；趋势越明确，阈值越小
        thresh_modifier = 1.0 - (sun_signal * 0.3 + trend_intensity * 0.3)
        thresh_modifier = max(0.7, thresh_modifier)  # 最多降低30%
        
        buy_threshold = max(60, base_thresh * thresh_modifier - sun_effect * 0.2)
        sell_threshold = max(60, base_thresh * thresh_modifier + sun_effect * 0.2)
        
        # 风险调整
        if self.high_vol_detected:
            buy_threshold *= 1.1  # 高波动时增加阈值10%
            sell_threshold *= 1.1
        
        # 基础头寸规模计算 - 强信号更激进
        base_size_factor = 0.6 + 0.3 * sun_signal  # 60-90% 基础仓位
        
        # 买入条件: 阳光指数低于最优值(阳光不足)且价格低于公允价值
        if sun < self.CSI and ask < fair - buy_threshold:
            # 确保足够流动性
            if ask_q and abs(ask_q) >= 3:
                # 趋势确认时加仓
                trend_multiplier = 1.0
                if combined_trend > 0.2:  # 强上升趋势
                    trend_multiplier = 1.3
                elif combined_trend < -0.2:  # 强下降趋势
                    trend_multiplier = 0.7
                
                # 考虑阈值因素 - 价格远低于阈值时加仓
                discount = (fair - buy_threshold - ask) / fair
                discount_boost = 1.0 + min(0.3, discount * 10)  # 最多增加30%
                
                # 考虑现有头寸 - 减少对大头寸的加仓
                position_factor = max(0.5, 1.0 - exposure * 0.6)  # 满仓最多降低50%
                
                # 综合所有因素
                buy_size = base_size_factor * trend_multiplier * position_factor * discount_boost
                
                # 确保不超过限制
                q = clamp(int(buy_size * limit), limit - pos)
                if q > 0:
                    orders.append(Order(product, ask, q))
                    
        # 卖出条件: 阳光指数高于最优值(阳光过剩)且价格高于公允价值
        if sun > self.CSI and bid > fair + sell_threshold:
            # 确保足够流动性
            if bid_q and abs(bid_q) >= 3:
                # 趋势确认时加仓
                trend_multiplier = 1.0
                if combined_trend < -0.2:  # 强下降趋势
                    trend_multiplier = 1.3
                elif combined_trend > 0.2:  # 强上升趋势
                    trend_multiplier = 0.7
                
                # 考虑阈值因素 - 价格远高于阈值时加仓
                premium = (bid - fair - sell_threshold) / fair
                premium_boost = 1.0 + min(0.3, premium * 10)  # 最多增加30%
                
                # 考虑现有头寸 - 减少对大头寸的加仓
                position_factor = max(0.5, 1.0 - exposure * 0.6)  # 满仓最多降低50%
                
                # 综合所有因素
                sell_size = base_size_factor * trend_multiplier * position_factor * premium_boost
                
                # 确保不超过限制
                q = clamp(int(sell_size * limit), -limit - pos)
                if q < 0:
                    orders.append(Order(product, bid, q))
        
        return orders
        
    def _handle_vouchers(self, state: TradingState) -> Dict[str, List[Order]]:
        """改进的火山岩凭证策略，极度保守的风险控制"""
        res: Dict[str, List[Order]] = {}
        rock_depth = state.order_depths["VOLCANIC_ROCK"]
        bid_r, _, ask_r, _ = best_bid_ask(rock_depth)
        if bid_r is None or ask_r is None:
            return res
            
        mid = (bid_r + ask_r) / 2
        vol = 0.0
        
        # 计算价格变化和波动性
        if self.last_mid is not None:
            vol = abs(mid - self.last_mid) / self.last_mid
        
        self.last_mid = mid
        
        # 火山岩价格趋势
        rock_trend = self._price_trend("VOLCANIC_ROCK", 8)
        
        # 极端风险控制
        if self.high_vol_detected:
            # 高波动环境下:
            # 1. 降低整体参与度
            # 2. 减少已有头寸，避免进一步风险
            
            # 先处理风险减轻 - 现有多头在暴跌时减仓，现有空头在暴涨时减仓
            for v in self.VOUCHERS:
                pos = state.position.get(v, 0)
                if abs(pos) < 10:  # 头寸很小时不处理
                    continue
                    
                d = state.order_depths[v]
                bid, _, ask, _ = best_bid_ask(d)
                if bid is None or ask is None:
                    continue
                
                # 风险控制逻辑 - 大波动时对冲减仓
                risk_orders = []
                
                # 多头减仓条件: 价格下跌且持有多头
                if rock_trend < -0.2 and pos > 20:
                    # 卖出一部分减少风险
                    reduce_qty = int(-pos * 0.3)  # 减少30%的头寸
                    if reduce_qty < 0 and bid:
                        risk_orders.append(Order(v, bid, reduce_qty))
                
                # 空头减仓条件: 价格上涨且持有空头
                elif rock_trend > 0.2 and pos < -20:
                    # 买入一部分减少风险
                    reduce_qty = int(-pos * 0.3)  # 减少30%的头寸
                    if reduce_qty > 0 and ask:
                        risk_orders.append(Order(v, ask, reduce_qty))
                
                if risk_orders:
                    res[v] = risk_orders
                    
            # 高波动环境下，避免开新仓
            if vol > 0.02:  # 超高波动
                return res  # 不开新仓
        
        # 如果市场波动超高，直接返回风险控制订单
        if vol > 0.03:
            return res
            
        # 动态调整期权溢价
        base_premium = 50
        
        # 在高波动时增加溢价
        vol_multiplier = 1 + min(5, vol * 10)  # 限制最大乘数为6
        vol_premium = base_premium * vol_multiplier
        
        # 根据市场整体波动调整风险承受度
        if self.day_volatility > 0.015 or self.high_vol_detected:  # 高波动日
            size_scaling = 0.5  # 大幅降低头寸规模
            thresh_scaling = 1.3  # 大幅增加阈值
        elif self.day_volatility > 0.01:  # 中等波动
            size_scaling = 0.7  # 中等降低头寸规模
            thresh_scaling = 1.2  # 中等增加阈值
        else:
            size_scaling = 1.0
            thresh_scaling = 1.0
        
        for v in self.VOUCHERS:
            # 跳过已有风险控制订单的凭证
            if v in res:
                continue
                
            d = state.order_depths[v]
            bid, bid_q, ask, ask_q = best_bid_ask(d)
            if bid is None or ask is None:
                continue
                
            strike = int(v.split("_")[-1])
            intrinsic = max(0, mid - strike)
            
            # 通过远离平值期权调整溢价
            moneyness = abs(mid - strike) / mid
            theo = intrinsic + vol_premium * (1 - moneyness * 0.5)
            
            # 检查头寸暴露
            pos = state.position.get(v, 0)
            limit = POSITION_LIMITS[v]
            exposure = self._position_exposure(v)
            
            # 若暴露度太高，就跳过开新仓
            if exposure > 0.7:
                continue
                
            lst: List[Order] = []
            
            # 动态阈值基于波动性
            base_thresh = 25  # 更保守的基础阈值
            buy_thresh = base_thresh * (1 + vol * 3) * thresh_scaling
            sell_thresh = base_thresh * (1 + vol * 3) * thresh_scaling
            
            # 买入逻辑
            if ask < theo - buy_thresh:
                # 检查流动性
                if ask_q and abs(ask_q) >= 3:
                    # 根据价差大小调整头寸规模
                    mispricing = (theo - buy_thresh - ask) / theo
                    size = 0.25 + min(0.25, mispricing * 5)  # 25-50%的限额，更保守
                    size *= size_scaling  # 波动调整
                    
                    # 仓位控制
                    if pos > 0:
                        # 已有多头时降低买入规模
                        position_factor = 1.0 - (pos / limit) * 0.7
                        size *= position_factor
                    
                    # 趋势确认则加仓，反之减仓
                    if rock_trend > 0.3:  # 强上涨趋势
                        size *= 1.2
                    elif rock_trend < -0.3:  # 强下跌趋势
                        size *= 0.7
                        
                    q = clamp(int(size * limit), limit - pos)
                    if q > 0:
                        lst.append(Order(v, ask, q))
            
            # 卖出逻辑
            if bid > theo + sell_thresh:
                # 检查流动性
                if bid_q and abs(bid_q) >= 3:
                    mispricing = (bid - theo - sell_thresh) / theo
                    size = 0.25 + min(0.25, mispricing * 5)  # 25-50%的限额，更保守
                    size *= size_scaling  # 波动调整
                    
                    # 仓位控制
                    if pos < 0:
                        # 已有空头时降低卖出规模
                        position_factor = 1.0 - (abs(pos) / limit) * 0.7
                        size *= position_factor
                    
                    # 趋势确认则加仓，反之减仓
                    if rock_trend < -0.3:  # 强下跌趋势
                        size *= 1.2
                    elif rock_trend > 0.3:  # 强上涨趋势
                        size *= 0.7
                        
                    q = clamp(int(size * limit), -limit - pos)
                    if q < 0:
                        lst.append(Order(v, bid, q))
                    
            if lst:
                res[v] = lst
        return res
        
    def _handle_baskets(self, state: TradingState) -> Dict[str, List[Order]]:
        """增强的篮子套利策略，更频繁的执行和严格的风险控制"""
        res: Dict[str, List[Order]] = {}
        depth = state.order_depths
        
        # 先计算所有组件的公允价值
        component_values = {}
        for basket, components in self.COMP_MAP.items():
            for p in components:
                if p not in component_values:
                    b, _, a, _ = best_bid_ask(depth.get(p, OrderDepth({}, {})))
                    if b is not None and a is not None:
                        mid = (b + a) / 2
                        # 添加趋势调整
                        trend = self._price_trend(p, 8)
                        mid_adjusted = mid * (1 + trend * 0.1)  # 10%趋势权重
                        component_values[p] = mid_adjusted
        
        for basket, components in self.COMP_MAP.items():
            # 计算组件公允价值
            total = 0
            all_available = True
            
            for p, qty in components.items():
                if p not in component_values:
                    all_available = False
                    break
                total += qty * component_values[p]
                
            if not all_available:
                continue
                
            bid, bid_q, ask, ask_q = best_bid_ask(depth[basket])
            if bid is None or ask is None:
                continue
                
            pos = state.position.get(basket, 0)
            limit = POSITION_LIMITS[basket]
            exposure = self._position_exposure(basket)
            
            # 高暴露跳过
            if exposure > 0.7:
                continue
                
            lst: List[Order] = []
            
            # 根据市场环境动态调整阈值
            if self.high_vol_detected:
                base_thresh = 0.01  # 高波动环境使用更高阈值
            else:
                base_thresh = 0.006  # 正常环境使用更低阈值
                
            # 计算篮子与组件间的价差
            basket_mid = (bid + ask) / 2
            relative_spread = (total - basket_mid) / total  # 正值表示篮子低估
            
            # 根据价差调整头寸规模
            spread_factor = min(1.0, abs(relative_spread) / 0.01)  # 标准化到0-1
            
            # 确保流动性充足
            min_quantity = 5
            
            # 买入条件：篮子价格低于组件总和
            if relative_spread > base_thresh:  # 篮子低估
                if ask_q and abs(ask_q) >= min_quantity:  # 确保足够流动性
                    # 计算头寸规模
                    position_factor = 1.0 - exposure * 0.7  # 随着暴露度增加而降低规模
                    size_factor = 0.3 + 0.3 * spread_factor * position_factor  # 30-60%
                    
                    # 根据趋势调整
                    basket_trend = self._price_trend(basket, 5)
                    if basket_trend > 0.2:  # 篮子已经在上涨
                        size_factor *= 1.2  # 加仓
                    elif basket_trend < -0.2:  # 篮子在下跌
                        size_factor *= 0.8  # 减仓
                    
                    q = clamp(int(size_factor * limit), limit - pos)
                    if q > 0:
                        lst.append(Order(basket, ask, q))
            
            # 卖出条件：篮子价格高于组件总和
            elif relative_spread < -base_thresh:  # 篮子高估
                if bid_q and abs(bid_q) >= min_quantity:  # 确保足够流动性
                    # 计算头寸规模
                    position_factor = 1.0 - exposure * 0.7  # 随着暴露度增加而降低规模
                    size_factor = 0.3 + 0.3 * spread_factor * position_factor  # 30-60%
                    
                    # 根据趋势调整
                    basket_trend = self._price_trend(basket, 5)
                    if basket_trend < -0.2:  # 篮子已经在下跌
                        size_factor *= 1.2  # 加仓
                    elif basket_trend > 0.2:  # 篮子在上涨
                        size_factor *= 0.8  # 减仓
                    
                    q = clamp(int(size_factor * limit), -limit - pos)
                    if q < 0:
                        lst.append(Order(basket, bid, q))
                    
            # 同时添加对篮子组件的对冲订单
            if lst and len(lst) == 1:  # 只有一个篮子订单
                order = lst[0]
                hedge_orders = []
                hedge_complete = True
                
                # 对篮子中每个组件进行对冲
                for comp, qty_per_basket in components.items():
                    comp_pos = state.position.get(comp, 0)
                    comp_limit = POSITION_LIMITS[comp]
                    
                    # 计算篮子交易需要的组件数量
                    hedge_qty = -order.quantity * qty_per_basket
                    
                    if hedge_qty > 0:  # 需要买入组件
                        # 验证组件流动性
                        _, _, comp_ask, comp_ask_q = best_bid_ask(depth[comp])
                        if comp_ask and comp_ask_q and abs(comp_ask_q) >= abs(hedge_qty):
                            hedge_qty = clamp(hedge_qty, comp_limit - comp_pos)
                            if hedge_qty > 0:
                                hedge_orders.append(Order(comp, comp_ask, hedge_qty))
                            else:
                                hedge_complete = False
                        else:
                            hedge_complete = False
                                
                    elif hedge_qty < 0:  # 需要卖出组件
                        # 验证组件流动性
                        comp_bid, comp_bid_q, _, _ = best_bid_ask(depth[comp])
                        if comp_bid and comp_bid_q and abs(comp_bid_q) >= abs(hedge_qty):
                            hedge_qty = clamp(hedge_qty, -comp_limit - comp_pos)
                            if hedge_qty < 0:
                                hedge_orders.append(Order(comp, comp_bid, hedge_qty))
                            else:
                                hedge_complete = False
                        else:
                            hedge_complete = False
                
                # 只有当所有组件都能完全对冲时才执行篮子交易
                if hedge_complete and len(hedge_orders) == len(components):
                    res[basket] = lst
                    for h_order in hedge_orders:
                        res.setdefault(h_order.symbol, []).append(h_order)
                
            # 如果没有添加对冲订单但有篮子交易，也执行
            elif lst and basket not in res:
                res[basket] = lst
                
        return res
        
    def _handle_momentum(self, state: TradingState) -> Dict[str, List[Order]]:
        """极度保守的动量策略，仅在确定趋势时交易"""
        orders: Dict[str, List[Order]] = {}
        
        # 高波动环境下大幅降低动量交易
        if self.high_vol_detected:
            momentum_scaling = 0.3  # 高风险环境下只用30%的规模
        elif self.day_volatility > 0.015:
            momentum_scaling = 0.5  # 中高波动降低规模
        elif self.day_volatility > 0.01:
            momentum_scaling = 0.7  # 中等波动轻度降低规模
        else:
            momentum_scaling = 1.0  # 低波动正常交易
        
        # 如果波动性极高则禁用动量策略
        if self.day_volatility > 0.025:  # 2.5%以上视为极高波动
            return {}
        
        for product, depth in state.order_depths.items():
            # 跳过其他策略处理的产品
            if (product == "MAGNIFICENT_MACARONS" or 
                product in self.VOUCHERS or 
                product in self.COMP_MAP or
                product in self.COMP_MAP["PICNIC_BASKET1"] or 
                product in self.COMP_MAP["PICNIC_BASKET2"]):
                continue
                
            bid_p, bid_q, ask_p, ask_q = best_bid_ask(depth)
            if bid_p is None or ask_p is None:
                continue
                
            # 检查暴露度
            exposure = self._position_exposure(product)
            if exposure > 0.7:  # 已有大量头寸则跳过
                continue
                
            # 获取交易连续性
            recent = state.market_trades.get(product, [])
            side, length = self.streak.get(product, (None, 0))
            
            # 更新交易连续性
            for t in recent:
                if t.buyer == "SUBMISSION" or t.seller == "SUBMISSION":
                    continue
                s = "BUY" if t.buyer in ("Caesar", "Camilla") else "SELL" if t.seller in ("Caesar", "Camilla") else None
                if not s:
                    continue
                if s == side:
                    length += 1
                else:
                    side, length = s, 1
                    
            self.streak[product] = (side, length)
            
            # 获取价格趋势确认
            trend = self._price_trend(product, 5)
            
            # 根据环境动态调整所需的连续性
            min_streak = 5 if self.high_vol_detected else 4 if self.day_volatility > 0.01 else 3
            
            # 只有同时满足连续交易和价格趋势时才交易
            trend_confirmed = (side == "BUY" and trend > 0.2) or (side == "SELL" and trend < -0.2)
            
            if length >= min_streak and trend_confirmed:
                pos = state.position.get(product, 0)
                limit = POSITION_LIMITS[product]
                
                # 计算规模因子
                base_size = min(0.25 + 0.05 * (length - min_streak), 0.4)  # 25-40%的限额
                size_factor = base_size * momentum_scaling * (1.0 - exposure * 0.7)
                
                # 交易执行
                if side == "BUY":
                    # 确保有足够流动性
                    max_qty = abs(ask_q) if ask_q else 0
                    qty = clamp(int(size_factor * limit), min(limit - pos, max_qty))
                    if qty > 0:
                        orders[product] = [Order(product, ask_p, qty)]
                else:  # SELL
                    max_qty = abs(bid_q) if bid_q else 0
                    qty = clamp(int(size_factor * limit), max(-limit - pos, -max_qty))
                    if qty < 0:
                        orders[product] = [Order(product, bid_p, qty)]
                        
        return orders
        
    def _risk_management(self, state: TradingState) -> Dict[str, List[Order]]:
        """风险管理 - 在极端情况下减少头寸"""
        if not self.high_vol_detected:  # 只在高波动时执行
            return {}
            
        orders: Dict[str, List[Order]] = {}
        
        # 检查所有产品的头寸
        for product in POSITION_LIMITS:
            if product in self.HIGH_RISK_PRODUCTS:
                # 高风险产品主动减仓
                pos = state.position.get(product, 0)
                if abs(pos) < 20:  # 忽略小头寸
                    continue
                    
                # 对大头寸进行风险控制
                reduction_pct = 0.4  # 减少40%的头寸
                reduction_qty = int(-pos * reduction_pct)
                
                if reduction_qty != 0:
                    depth = state.order_depths.get(product, OrderDepth({}, {}))
                    bid, _, ask, _ = best_bid_ask(depth)
                    
                    if reduction_qty > 0 and ask:  # 买入减少空头
                        orders[product] = [Order(product, ask, reduction_qty)]
                    elif reduction_qty < 0 and bid:  # 卖出减少多头
                        orders[product] = [Order(product, bid, reduction_qty)]
        
        return orders

    def run(self, state: TradingState):
        """主交易逻辑，综合所有策略"""
        # 估计市场状态
        self._estimate_market_state(state)
        
        all_orders: Dict[str, List[Order]] = {}
        
        # 1. 高波动时先执行风险管理
        if self.high_vol_detected:
            risk_orders = self._risk_management(state)
            all_orders.update(risk_orders)
        
        # 2. 马卡龙交易 (最高优先级，经过SunlightSniper优化)
        # 这是低风险品种，即使在高波动环境下也可以交易
        macaron_orders = self._handle_macarons(state)
        if macaron_orders:
            all_orders["MAGNIFICENT_MACARONS"] = macaron_orders
            
        # 3. 处理篮子套利 (改进后的BasketBandit)
        # 这是相对低风险的策略
        basket_orders = self._handle_baskets(state)
        all_orders.update(basket_orders)
        
        # 4. 处理火山岩凭证 (改进后的GammaGorilla，风险控制更好)
        # 但在高波动时会更保守
        if not self.high_vol_detected or self.day_volatility < 0.02:
            voucher_orders = self._handle_vouchers(state)
            all_orders.update(voucher_orders)
        
        # 5. 处理其他产品的动量交易 (优化后的MomentumRipper)
        # 在高波动环境下大幅减少或禁用
        if not self.high_vol_detected or self.day_volatility < 0.015:
            momentum_orders = self._handle_momentum(state)
            all_orders.update(momentum_orders)
        
        return all_orders, 0, self.TAG

class Trader:
    def __init__(self):
        self.trader = EliteEnsembleTrader()

    def run(self, state: TradingState):
        return self.trader.run(state) 