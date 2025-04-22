from typing import Dict, List, Tuple, Optional, Set
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np

# 火山岩及其期权
VOLCANIC_PRODUCTS = {
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200,  # 行权价9500的看涨期权
    "VOLCANIC_ROCK_VOUCHER_9750": 200,  # 行权价9750的看涨期权
    "VOLCANIC_ROCK_VOUCHER_10000": 200, # 行权价10000的看涨期权
    "VOLCANIC_ROCK_VOUCHER_10250": 200, # 行权价10250的看涨期权
    "VOLCANIC_ROCK_VOUCHER_10500": 200  # 行权价10500的看涨期权
}

# 其他产品（不会主动交易）
OTHER_PRODUCTS = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "MAGNIFICENT_MACARONS": 75
}

# 合并所有产品限制
LIMIT = {**VOLCANIC_PRODUCTS, **OTHER_PRODUCTS}

# 参数设置 - 波动率交易
PARAM = {
    "vol_window_short": 10,     # 短期波动率窗口
    "vol_window_long": 30,      # 长期波动率窗口
    "vol_ratio_threshold": 1.3, # 波动率比率阈值
    "max_position_pct": 0.9,    # 最大仓位百分比
    "vol_trade_size_pct": 0.4,  # 波动率交易规模百分比
    "aggressive_entry": True,   # 激进进场
    "aggressive_exit": True,    # 激进出场
    "vol_scale": 2.0,           # 波动率缩放因子
    "max_vol_attempts": 8,      # 每个时间步最大波动率交易尝试次数
    "min_profit_target": 8,     # 最小利润目标
    "straddle_size_pct": 0.3,   # 跨式策略规模百分比
    "strangle_size_pct": 0.25,  # 宽跨式策略规模百分比
    "butterfly_size_pct": 0.2,  # 蝶式策略规模百分比
    "vol_rank_threshold": 0.8,  # 波动率排名阈值
    "vol_percentile_high": 80,  # 高波动率百分位
    "vol_percentile_low": 20    # 低波动率百分位
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.volatilities = defaultdict(list)  # 存储历史波动率
        self.fair_values = {}
        self.implied_vols = defaultdict(list)  # 存储隐含波动率
        self.vol_rank = defaultdict(float)     # 波动率排名
        self.vol_percentile = defaultdict(float)  # 波动率百分位
        self.atm_option = None  # 当前平值期权
        self.vol_regime = "normal"  # 波动率环境：high, normal, low
        
    def _vol(self, p: str, window: int = None) -> float:
        """计算产品波动率"""
        if window is None:
            window = PARAM["vol_window_short"]
            
        h = self.prices[p]
        if len(h) < window:
            return 3  # 默认较高波动率
        return statistics.stdev(h[-window:]) * PARAM["vol_scale"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def calculate_volatility_metrics(self, state: TradingState):
        """计算波动率相关指标"""
        # 首先计算VOLCANIC_ROCK的波动率
        if "VOLCANIC_ROCK" in state.order_depths:
            depth = state.order_depths["VOLCANIC_ROCK"]
            mid = self._mid(depth)
            if mid is None:
                return
                
            self.fair_values["VOLCANIC_ROCK"] = mid
            
            # 计算短期和长期波动率
            short_vol = self._vol("VOLCANIC_ROCK", PARAM["vol_window_short"])
            long_vol = self._vol("VOLCANIC_ROCK", PARAM["vol_window_long"])
            
            # 记录波动率
            self.volatilities["VOLCANIC_ROCK"].append(short_vol)
            
            # 保持波动率历史记录在合理范围内
            if len(self.volatilities["VOLCANIC_ROCK"]) > 100:
                self.volatilities["VOLCANIC_ROCK"].pop(0)
            
            # 计算波动率比率
            vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0
            
            # 确定波动率环境
            if vol_ratio > PARAM["vol_ratio_threshold"]:
                self.vol_regime = "high"
            elif vol_ratio < 1.0 / PARAM["vol_ratio_threshold"]:
                self.vol_regime = "low"
            else:
                self.vol_regime = "normal"
            
            # 计算波动率排名和百分位
            if len(self.volatilities["VOLCANIC_ROCK"]) > 10:
                sorted_vols = sorted(self.volatilities["VOLCANIC_ROCK"])
                current_rank = sorted_vols.index(short_vol) / len(sorted_vols)
                self.vol_rank["VOLCANIC_ROCK"] = current_rank
                
                # 计算百分位
                percentile = current_rank * 100
                self.vol_percentile["VOLCANIC_ROCK"] = percentile
            
            # 找到当前平值期权
            closest_strike = float('inf')
            self.atm_option = None
            
            # 计算各期权的隐含波动率
            for voucher in [p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"]:
                # 从期权名称中提取行权价
                try:
                    strike = int(voucher.split("_")[-1])
                    
                    # 找到最接近平值的期权
                    if abs(strike - mid) < abs(closest_strike - mid):
                        closest_strike = strike
                        self.atm_option = voucher
                    
                    # 如果期权在订单深度中，计算隐含波动率
                    if voucher in state.order_depths:
                        option_depth = state.order_depths[voucher]
                        option_mid = self._mid(option_depth)
                        
                        if option_mid is not None:
                            # 简化的隐含波动率计算
                            # 实际上应该使用Black-Scholes模型
                            intrinsic = max(0, mid - strike)
                            time_value = option_mid - intrinsic
                            
                            # 时间价值越高，隐含波动率越高
                            implied_vol = time_value / 100  # 简化计算
                            
                            # 记录隐含波动率
                            self.implied_vols[voucher].append(implied_vol)
                            
                            # 保持隐含波动率历史记录在合理范围内
                            if len(self.implied_vols[voucher]) > 50:
                                self.implied_vols[voucher].pop(0)
                except ValueError:
                    continue
    
    def find_volatility_trading_opportunities(self, state: TradingState) -> List[Dict]:
        """寻找波动率交易机会"""
        opportunities = []
        
        # 如果没有波动率数据，则不进行交易
        if "VOLCANIC_ROCK" not in self.volatilities or len(self.volatilities["VOLCANIC_ROCK"]) < 10:
            return opportunities
            
        # 根据波动率环境选择策略
        if self.vol_regime == "high":
            # 高波动率环境：卖出波动率
            opportunities.extend(self.find_short_vol_opportunities(state))
        elif self.vol_regime == "low":
            # 低波动率环境：买入波动率
            opportunities.extend(self.find_long_vol_opportunities(state))
        else:
            # 正常波动率环境：寻找相对价值机会
            opportunities.extend(self.find_relative_vol_opportunities(state))
        
        return opportunities
    
    def find_short_vol_opportunities(self, state: TradingState) -> List[Dict]:
        """寻找卖出波动率的机会"""
        opportunities = []
        
        # 如果没有平值期权，则不进行交易
        if not self.atm_option or self.atm_option not in state.order_depths:
            return opportunities
            
        # 获取平值期权的订单深度
        atm_depth = state.order_depths[self.atm_option]
        atm_b, atm_a = best_bid_ask(atm_depth)
        
        if atm_b is None or atm_a is None:
            return opportunities
            
        # 1. 卖出跨式策略（Straddle）
        # 需要找到一个虚值看跌期权
        atm_strike = int(self.atm_option.split("_")[-1])
        
        # 计算交易规模
        straddle_size = int(VOLCANIC_PRODUCTS[self.atm_option] * PARAM["straddle_size_pct"])
        
        # 添加卖出平值看涨期权的机会
        opportunities.append({
            "type": "sell_call",
            "option": self.atm_option,
            "price": atm_b,
            "max_size": min(
                atm_depth.buy_orders[atm_b],
                straddle_size
            )
        })
        
        # 2. 卖出蝶式策略（Butterfly）
        # 需要找到一个低行权价和一个高行权价的期权
        options = sorted([p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"], 
                         key=lambda x: int(x.split("_")[-1]))
        
        if len(options) >= 3:
            # 找到平值期权的索引
            atm_index = options.index(self.atm_option)
            
            # 确保有低行权价和高行权价的期权
            if atm_index > 0 and atm_index < len(options) - 1:
                low_strike_option = options[atm_index - 1]
                high_strike_option = options[atm_index + 1]
                
                # 确保这些期权在订单深度中
                if (low_strike_option in state.order_depths and 
                    high_strike_option in state.order_depths):
                    
                    low_depth = state.order_depths[low_strike_option]
                    high_depth = state.order_depths[high_strike_option]
                    
                    low_b, low_a = best_bid_ask(low_depth)
                    high_b, high_a = best_bid_ask(high_depth)
                    
                    if low_b and high_b and atm_a:
                        # 计算蝶式策略的交易规模
                        butterfly_size = int(min(
                            VOLCANIC_PRODUCTS[low_strike_option],
                            VOLCANIC_PRODUCTS[self.atm_option],
                            VOLCANIC_PRODUCTS[high_strike_option]
                        ) * PARAM["butterfly_size_pct"])
                        
                        # 添加蝶式策略的机会
                        opportunities.append({
                            "type": "butterfly",
                            "low_option": low_strike_option,
                            "atm_option": self.atm_option,
                            "high_option": high_strike_option,
                            "low_price": low_b,
                            "atm_price": atm_a,
                            "high_price": high_b,
                            "max_size": butterfly_size
                        })
        
        return opportunities
    
    def find_long_vol_opportunities(self, state: TradingState) -> List[Dict]:
        """寻找买入波动率的机会"""
        opportunities = []
        
        # 如果没有平值期权，则不进行交易
        if not self.atm_option or self.atm_option not in state.order_depths:
            return opportunities
            
        # 获取平值期权的订单深度
        atm_depth = state.order_depths[self.atm_option]
        atm_b, atm_a = best_bid_ask(atm_depth)
        
        if atm_b is None or atm_a is None:
            return opportunities
            
        # 1. 买入跨式策略（Straddle）
        # 计算交易规模
        straddle_size = int(VOLCANIC_PRODUCTS[self.atm_option] * PARAM["straddle_size_pct"])
        
        # 添加买入平值看涨期权的机会
        opportunities.append({
            "type": "buy_call",
            "option": self.atm_option,
            "price": atm_a,
            "max_size": min(
                abs(atm_depth.sell_orders[atm_a]),
                straddle_size
            )
        })
        
        # 2. 买入宽跨式策略（Strangle）
        # 需要找到一个低行权价和一个高行权价的期权
        options = sorted([p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"], 
                         key=lambda x: int(x.split("_")[-1]))
        
        if len(options) >= 3:
            # 找到平值期权的索引
            atm_index = options.index(self.atm_option)
            
            # 确保有低行权价和高行权价的期权
            if atm_index > 0 and atm_index < len(options) - 1:
                low_strike_option = options[atm_index - 1]
                high_strike_option = options[atm_index + 1]
                
                # 确保这些期权在订单深度中
                if (low_strike_option in state.order_depths and 
                    high_strike_option in state.order_depths):
                    
                    low_depth = state.order_depths[low_strike_option]
                    high_depth = state.order_depths[high_strike_option]
                    
                    low_a = min(low_depth.sell_orders.keys()) if low_depth.sell_orders else None
                    high_a = min(high_depth.sell_orders.keys()) if high_depth.sell_orders else None
                    
                    if low_a and high_a:
                        # 计算宽跨式策略的交易规模
                        strangle_size = int(min(
                            VOLCANIC_PRODUCTS[low_strike_option],
                            VOLCANIC_PRODUCTS[high_strike_option]
                        ) * PARAM["strangle_size_pct"])
                        
                        # 添加宽跨式策略的机会
                        opportunities.append({
                            "type": "strangle",
                            "low_option": low_strike_option,
                            "high_option": high_strike_option,
                            "low_price": low_a,
                            "high_price": high_a,
                            "max_size": strangle_size
                        })
        
        return opportunities
    
    def find_relative_vol_opportunities(self, state: TradingState) -> List[Dict]:
        """寻找相对价值的波动率交易机会"""
        opportunities = []
        
        # 计算所有期权的隐含波动率排名
        option_iv_ranks = {}
        
        for option in [p for p in VOLCANIC_PRODUCTS if p != "VOLCANIC_ROCK"]:
            if option in self.implied_vols and len(self.implied_vols[option]) > 5:
                current_iv = self.implied_vols[option][-1]
                sorted_ivs = sorted(self.implied_vols[option])
                rank = sorted_ivs.index(current_iv) / len(sorted_ivs)
                option_iv_ranks[option] = rank
        
        # 找到隐含波动率最高和最低的期权
        if option_iv_ranks:
            high_iv_option = max(option_iv_ranks.items(), key=lambda x: x[1])[0]
            low_iv_option = min(option_iv_ranks.items(), key=lambda x: x[1])[0]
            
            # 如果隐含波动率差异足够大，进行相对价值交易
            if (option_iv_ranks[high_iv_option] - option_iv_ranks[low_iv_option] > 
                PARAM["vol_rank_threshold"]):
                
                # 确保这些期权在订单深度中
                if (high_iv_option in state.order_depths and 
                    low_iv_option in state.order_depths):
                    
                    high_depth = state.order_depths[high_iv_option]
                    low_depth = state.order_depths[low_iv_option]
                    
                    high_b = max(high_depth.buy_orders.keys()) if high_depth.buy_orders else None
                    low_a = min(low_depth.sell_orders.keys()) if low_depth.sell_orders else None
                    
                    if high_b and low_a:
                        # 计算相对价值交易的规模
                        rel_vol_size = int(min(
                            VOLCANIC_PRODUCTS[high_iv_option],
                            VOLCANIC_PRODUCTS[low_iv_option]
                        ) * PARAM["vol_trade_size_pct"])
                        
                        # 添加相对价值交易的机会
                        opportunities.append({
                            "type": "relative_value",
                            "high_iv_option": high_iv_option,
                            "low_iv_option": low_iv_option,
                            "high_price": high_b,
                            "low_price": low_a,
                            "max_size": rel_vol_size
                        })
        
        return opportunities
    
    def execute_volatility_trading(self, state: TradingState, opportunities: List[Dict]) -> Dict[str, List[Order]]:
        """执行波动率交易策略"""
        result = {}
        
        # 限制每个时间步的交易尝试次数
        attempts = 0
        
        for opp in opportunities:
            if attempts >= PARAM["max_vol_attempts"]:
                break
                
            if opp["type"] == "buy_call":
                # 买入看涨期权
                option = opp["option"]
                price = opp["price"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS[option] - state.position.get(option, 0))
                
                if size > 0:
                    if option not in result:
                        result[option] = []
                    result[option].append(Order(option, price, size))
                    attempts += 1
            
            elif opp["type"] == "sell_call":
                # 卖出看涨期权
                option = opp["option"]
                price = opp["price"]
                size = min(opp["max_size"], VOLCANIC_PRODUCTS[option] + state.position.get(option, 0))
                
                if size > 0:
                    if option not in result:
                        result[option] = []
                    result[option].append(Order(option, price, -size))
                    attempts += 1
            
            elif opp["type"] == "butterfly":
                # 蝶式策略：买入低行权价期权，卖出两倍平值期权，买入高行权价期权
                low_option = opp["low_option"]
                atm_option = opp["atm_option"]
                high_option = opp["high_option"]
                low_price = opp["low_price"]
                atm_price = opp["atm_price"]
                high_price = opp["high_price"]
                size = opp["max_size"]
                
                # 检查仓位限制
                low_avail = VOLCANIC_PRODUCTS[low_option] - state.position.get(low_option, 0)
                atm_avail = VOLCANIC_PRODUCTS[atm_option] + state.position.get(atm_option, 0)
                high_avail = VOLCANIC_PRODUCTS[high_option] - state.position.get(high_option, 0)
                
                actual_size = min(size, low_avail, atm_avail // 2, high_avail)
                
                if actual_size > 0:
                    # 买入低行权价期权
                    if low_option not in result:
                        result[low_option] = []
                    result[low_option].append(Order(low_option, low_price, actual_size))
                    
                    # 卖出两倍平值期权
                    if atm_option not in result:
                        result[atm_option] = []
                    result[atm_option].append(Order(atm_option, atm_price, -actual_size * 2))
                    
                    # 买入高行权价期权
                    if high_option not in result:
                        result[high_option] = []
                    result[high_option].append(Order(high_option, high_price, actual_size))
                    
                    attempts += 1
            
            elif opp["type"] == "strangle":
                # 宽跨式策略：买入低行权价期权和高行权价期权
                low_option = opp["low_option"]
                high_option = opp["high_option"]
                low_price = opp["low_price"]
                high_price = opp["high_price"]
                size = opp["max_size"]
                
                # 检查仓位限制
                low_avail = VOLCANIC_PRODUCTS[low_option] - state.position.get(low_option, 0)
                high_avail = VOLCANIC_PRODUCTS[high_option] - state.position.get(high_option, 0)
                
                actual_size = min(size, low_avail, high_avail)
                
                if actual_size > 0:
                    # 买入低行权价期权
                    if low_option not in result:
                        result[low_option] = []
                    result[low_option].append(Order(low_option, low_price, actual_size))
                    
                    # 买入高行权价期权
                    if high_option not in result:
                        result[high_option] = []
                    result[high_option].append(Order(high_option, high_price, actual_size))
                    
                    attempts += 1
            
            elif opp["type"] == "relative_value":
                # 相对价值交易：卖出高隐含波动率期权，买入低隐含波动率期权
                high_iv_option = opp["high_iv_option"]
                low_iv_option = opp["low_iv_option"]
                high_price = opp["high_price"]
                low_price = opp["low_price"]
                size = opp["max_size"]
                
                # 检查仓位限制
                high_avail = VOLCANIC_PRODUCTS[high_iv_option] + state.position.get(high_iv_option, 0)
                low_avail = VOLCANIC_PRODUCTS[low_iv_option] - state.position.get(low_iv_option, 0)
                
                actual_size = min(size, high_avail, low_avail)
                
                if actual_size > 0:
                    # 卖出高隐含波动率期权
                    if high_iv_option not in result:
                        result[high_iv_option] = []
                    result[high_iv_option].append(Order(high_iv_option, high_price, -actual_size))
                    
                    # 买入低隐含波动率期权
                    if low_iv_option not in result:
                        result[low_iv_option] = []
                    result[low_iv_option].append(Order(low_iv_option, low_price, actual_size))
                    
                    attempts += 1
        
        return result
    
    def aggressive_vol_trading(self, p: str, depth: OrderDepth, pos: int) -> List[Order]:
        """激进的波动率交易策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: 
            return orders
        
        # 记录价格历史
        self.prices[p].append(mid)
        
        # 计算波动率
        vol = self._vol(p)
        
        # 计算价差 - 使用更窄的价差
        spread = max(1, int(vol * 0.4))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模 - 使用更大的规模
        max_position = int(VOLCANIC_PRODUCTS[p] * PARAM["max_position_pct"])
        size = max(1, max_position // 3)  # 每次用1/3的允许仓位
        
        # 根据波动率环境调整策略
        if self.vol_regime == "high":
            # 高波动率环境：更积极地卖出
            sell_px = int(mid + spread * 0.8)  # 降低卖出价格
            size_sell = int(size * 1.2)  # 增加卖出规模
            size_buy = int(size * 0.8)  # 减少买入规模
        elif self.vol_regime == "low":
            # 低波动率环境：更积极地买入
            buy_px = int(mid - spread * 0.8)  # 提高买入价格
            size_buy = int(size * 1.2)  # 增加买入规模
            size_sell = int(size * 0.8)  # 减少卖出规模
        else:
            # 正常波动率环境：平衡买卖
            size_buy = size
            size_sell = size
        
        # 主动吃单
        if PARAM["aggressive_entry"]:
            b, a = best_bid_ask(depth)
            if a is not None and b is not None:
                # 如果卖单价格低于我们的买入价，主动买入
                if a < buy_px and pos < max_position:
                    qty = min(size_buy, max_position - pos, abs(depth.sell_orders[a]))
                    if qty > 0: orders.append(Order(p, a, qty))
                
                # 如果买单价格高于我们的卖出价，主动卖出
                if b > sell_px and pos > -max_position:
                    qty = min(size_sell, max_position + pos, depth.buy_orders[b])
                    if qty > 0: orders.append(Order(p, b, -qty))
        
        # 常规做市订单
        if pos < max_position:
            orders.append(Order(p, buy_px, min(size_buy, max_position - pos)))
        if pos > -max_position:
            orders.append(Order(p, sell_px, -min(size_sell, max_position + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 计算波动率相关指标
        self.calculate_volatility_metrics(state)
        
        # 寻找波动率交易机会
        opportunities = self.find_volatility_trading_opportunities(state)
        
        # 执行波动率交易策略
        if opportunities:
            vol_orders = self.execute_volatility_trading(state, opportunities)
            result.update(vol_orders)
        
        # 对未在波动率交易中交易的产品应用激进波动率交易策略
        for p in VOLCANIC_PRODUCTS:
            # 如果已经在波动率交易中交易了这个产品，跳过
            if p in result:
                continue
                
            if p in state.order_depths:
                depth = state.order_depths[p]
                pos = state.position.get(p, 0)
                result[p] = self.aggressive_vol_trading(p, depth, pos)
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
