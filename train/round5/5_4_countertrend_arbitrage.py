from typing import Dict, List, Tuple, Optional, Set
from datamodel import OrderDepth, TradingState, Order
import statistics
from collections import defaultdict
import numpy as np

# 产品限制
LIMIT = {
    "RAINFOREST_RESIN": 50, "KELP": 50, "SQUID_INK": 50,
    "PICNIC_BASKET1": 60, "PICNIC_BASKET2": 100,
    "CROISSANTS": 250, "JAMS": 350, "DJEMBES": 60,
    "VOLCANIC_ROCK": 400,
    "VOLCANIC_ROCK_VOUCHER_9500": 200, "VOLCANIC_ROCK_VOUCHER_9750": 200,
    "VOLCANIC_ROCK_VOUCHER_10000": 200, "VOLCANIC_ROCK_VOUCHER_10250": 200,
    "VOLCANIC_ROCK_VOUCHER_10500": 200,
    "MAGNIFICENT_MACARONS": 75
}

# 产品组合 - 相关产品组合在一起
PRODUCT_GROUPS = {
    "volcanic": [
        "VOLCANIC_ROCK", 
        "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
        "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500"
    ],
    "picnic": ["PICNIC_BASKET1", "PICNIC_BASKET2"],
    "food": ["CROISSANTS", "JAMS", "MAGNIFICENT_MACARONS"],
    "materials": ["RAINFOREST_RESIN", "KELP", "SQUID_INK", "DJEMBES"]
}

# 反向查找产品所属组
PRODUCT_TO_GROUP = {}
for group, products in PRODUCT_GROUPS.items():
    for product in products:
        PRODUCT_TO_GROUP[product] = group

# 参数
PARAM = {
    # 趋势检测参数
    "trend_window": 30,       # 趋势检测窗口
    "trend_threshold": 0.7,   # 趋势检测阈值
    "reversal_window": 5,     # 反转检测窗口
    "reversal_threshold": 0.8, # 反转检测阈值
    
    # 交易参数
    "position_limit_pct": 0.6, # 仓位限制百分比
    "mm_size_frac": 0.15,     # 做市规模
    "counter_size_frac": 0.25, # 反趋势交易规模
    "arb_size_frac": 0.3,     # 套利交易规模
    "min_spread": 2,          # 最小价差
    "vol_scale": 1.2,         # 波动率缩放因子
    
    # 技术指标参数
    "rsi_period": 14,         # RSI周期
    "rsi_overbought": 70,     # RSI超买阈值
    "rsi_oversold": 30,       # RSI超卖阈值
    
    # 套利参数
    "correlation_window": 50, # 相关性计算窗口
    "min_correlation": 0.7,   # 最小相关性阈值
    "divergence_threshold": 2.0, # 价格偏离阈值
    "mean_reversion_threshold": 1.5, # 均值回归阈值
    "arb_cooldown": 10,       # 套利冷却期
    
    # 风险控制参数
    "stop_loss_pct": 0.03,    # 止损百分比
    "max_drawdown": 200,      # 最大回撤
    "max_trades_per_day": 5   # 每天最大交易次数
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.returns = defaultdict(list)  # 价格回报率
        self.position_history = defaultdict(list)  # 仓位历史
        self.rsi_values = defaultdict(float)  # RSI值
        self.last_counter_trade = defaultdict(int)  # 上次反向交易的时间戳
        self.last_arb_trade = defaultdict(int)  # 上次套利交易的时间戳
        self.trade_count = defaultdict(int)  # 交易计数
        self.entry_prices = defaultdict(float)  # 入场价格
        self.max_position_values = defaultdict(float)  # 最大仓位价值
        self.correlations = defaultdict(dict)  # 产品间相关性
        self.price_ratios = defaultdict(dict)  # 产品间价格比率
        self.z_scores = defaultdict(dict)  # 价格比率的Z分数
        self.arb_pairs = []  # 套利对
        
    def _vol(self, p: str) -> float:
        """计算产品波动率"""
        h = self.prices[p]
        if len(h) < 15: return 1
        return statistics.stdev(h[-15:]) * PARAM["vol_scale"] or 1
        
    def _mid(self, depth: OrderDepth) -> Optional[float]:
        """计算中间价"""
        b, a = best_bid_ask(depth)
        return (b+a)/2 if b is not None and a is not None else None
    
    def calculate_rsi(self, p: str):
        """计算相对强弱指数(RSI)"""
        if len(self.prices[p]) < PARAM["rsi_period"] + 1:
            self.rsi_values[p] = 50  # 默认中性值
            return
            
        # 计算价格变动
        price_changes = [self.prices[p][i] - self.prices[p][i-1] for i in range(1, len(self.prices[p]))]
        
        # 只使用最近的价格变动
        price_changes = price_changes[-PARAM["rsi_period"]:]
        
        # 计算上涨和下跌的平均值
        gains = [max(0, change) for change in price_changes]
        losses = [max(0, -change) for change in price_changes]
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        
        # 计算相对强度
        if avg_loss == 0:
            rs = 100
        else:
            rs = avg_gain / avg_loss
            
        # 计算RSI
        self.rsi_values[p] = 100 - (100 / (1 + rs))
    
    def detect_trend(self, p: str) -> int:
        """检测市场趋势"""
        prices = self.prices[p]
        if len(prices) < PARAM["trend_window"]:
            return 0  # 数据不足
            
        recent_prices = prices[-PARAM["trend_window"]:]  
        up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
        
        up_ratio = up_moves / (len(recent_prices) - 1)
        down_ratio = down_moves / (len(recent_prices) - 1)
        
        if up_ratio > PARAM["trend_threshold"]:
            return 1  # 上升趋势
        elif down_ratio > PARAM["trend_threshold"]:
            return -1  # 下降趋势
        return 0  # 中性
    
    def detect_reversal(self, p: str) -> int:
        """检测市场反转"""
        prices = self.prices[p]
        if len(prices) < PARAM["reversal_window"] + 5:
            return 0  # 数据不足
            
        # 检查之前的趋势
        prev_trend = self.detect_trend(p)
        if prev_trend == 0:
            return 0  # 没有明确趋势，无法判断反转
            
        # 检查最近的价格变动
        recent_prices = prices[-PARAM["reversal_window"]:]
        prev_prices = prices[-PARAM["reversal_window"]-5:-PARAM["reversal_window"]]
        
        if prev_trend == 1:  # 之前是上升趋势
            # 检查是否开始下跌
            down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
            down_ratio = down_moves / (len(recent_prices) - 1)
            
            if down_ratio > PARAM["reversal_threshold"]:
                return -1  # 上升趋势反转为下降
        
        elif prev_trend == -1:  # 之前是下降趋势
            # 检查是否开始上涨
            up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
            up_ratio = up_moves / (len(recent_prices) - 1)
            
            if up_ratio > PARAM["reversal_threshold"]:
                return 1  # 下降趋势反转为上升
                
        return 0  # 没有检测到反转
    
    def update_correlations(self):
        """更新产品间相关性"""
        # 对每个产品组内的产品计算相关性
        for group, products in PRODUCT_GROUPS.items():
            for i, p1 in enumerate(products):
                if p1 not in self.prices or len(self.prices[p1]) < PARAM["correlation_window"]:
                    continue
                    
                for p2 in products[i+1:]:
                    if p2 not in self.prices or len(self.prices[p2]) < PARAM["correlation_window"]:
                        continue
                        
                    # 计算相关性
                    try:
                        # 使用最近的价格
                        prices1 = self.prices[p1][-PARAM["correlation_window"]:]
                        prices2 = self.prices[p2][-PARAM["correlation_window"]:]
                        
                        # 计算价格变动
                        returns1 = [prices1[i] / prices1[i-1] - 1 for i in range(1, len(prices1))]
                        returns2 = [prices2[i] / prices2[i-1] - 1 for i in range(1, len(prices2))]
                        
                        # 计算相关性
                        correlation = np.corrcoef(returns1, returns2)[0, 1]
                        
                        # 存储相关性
                        self.correlations[p1][p2] = correlation
                        self.correlations[p2][p1] = correlation
                        
                        # 如果相关性足够高，计算价格比率
                        if abs(correlation) > PARAM["min_correlation"]:
                            # 计算价格比率
                            ratio = prices1[-1] / prices2[-1]
                            
                            # 存储价格比率
                            self.price_ratios[p1][p2] = ratio
                            self.price_ratios[p2][p1] = 1 / ratio
                            
                            # 如果有足够的历史数据，计算Z分数
                            if p1 in self.price_ratios and p2 in self.price_ratios[p1]:
                                ratios = [self.prices[p1][i] / self.prices[p2][i] for i in range(max(0, len(self.prices[p1]) - PARAM["correlation_window"]), len(self.prices[p1]))]
                                
                                # 计算均值和标准差
                                mean_ratio = sum(ratios) / len(ratios)
                                std_ratio = statistics.stdev(ratios) if len(ratios) > 1 else 1
                                
                                # 计算Z分数
                                z_score = (ratio - mean_ratio) / std_ratio if std_ratio > 0 else 0
                                
                                # 存储Z分数
                                self.z_scores[p1][p2] = z_score
                                self.z_scores[p2][p1] = -z_score
                    except:
                        # 忽略计算错误
                        pass
    
    def find_arbitrage_opportunities(self) -> List[Dict]:
        """寻找套利机会"""
        opportunities = []
        
        # 清空套利对
        self.arb_pairs = []
        
        # 对每个产品组内的产品寻找套利机会
        for group, products in PRODUCT_GROUPS.items():
            for i, p1 in enumerate(products):
                if p1 not in self.z_scores:
                    continue
                    
                for p2 in products[i+1:]:
                    if p2 not in self.z_scores[p1]:
                        continue
                        
                    # 获取Z分数
                    z_score = self.z_scores[p1][p2]
                    
                    # 如果Z分数超过阈值，可能存在套利机会
                    if abs(z_score) > PARAM["divergence_threshold"]:
                        # 确定交易方向
                        if z_score > 0:
                            # p1相对p2价格过高，卖出p1买入p2
                            self.arb_pairs.append((p1, p2))
                            opportunities.append({
                                "type": "pair_divergence",
                                "sell_product": p1,
                                "buy_product": p2,
                                "z_score": z_score
                            })
                        else:
                            # p2相对p1价格过高，卖出p2买入p1
                            self.arb_pairs.append((p2, p1))
                            opportunities.append({
                                "type": "pair_divergence",
                                "sell_product": p2,
                                "buy_product": p1,
                                "z_score": -z_score
                            })
        
        # 对火山岩及其期权寻找特殊套利机会
        if "VOLCANIC_ROCK" in self.prices and len(self.prices["VOLCANIC_ROCK"]) > 10:
            rock_price = self.prices["VOLCANIC_ROCK"][-1]
            
            for voucher in [p for p in PRODUCT_GROUPS["volcanic"] if p.startswith("VOLCANIC_ROCK_VOUCHER")]:
                if voucher not in self.prices or len(self.prices[voucher]) < 10:
                    continue
                    
                # 提取行权价
                try:
                    strike = int(voucher.split("_")[-1])
                    voucher_price = self.prices[voucher][-1]
                    
                    # 计算理论价值
                    intrinsic_value = max(0, rock_price - strike)
                    time_value = self._vol("VOLCANIC_ROCK") * 10  # 简化的时间价值计算
                    theoretical_value = intrinsic_value + time_value
                    
                    # 计算价格偏离
                    deviation = voucher_price / theoretical_value if theoretical_value > 0 else 1
                    
                    # 如果偏离足够大，可能存在套利机会
                    if deviation > PARAM["mean_reversion_threshold"]:
                        # 期权价格过高，卖出期权买入基础资产
                        opportunities.append({
                            "type": "option_overpriced",
                            "sell_product": voucher,
                            "buy_product": "VOLCANIC_ROCK",
                            "deviation": deviation
                        })
                    elif deviation < 1 / PARAM["mean_reversion_threshold"]:
                        # 期权价格过低，买入期权卖出基础资产
                        opportunities.append({
                            "type": "option_underpriced",
                            "buy_product": voucher,
                            "sell_product": "VOLCANIC_ROCK",
                            "deviation": 1 / deviation
                        })
                except:
                    # 忽略计算错误
                    pass
        
        return opportunities
    
    def check_risk_controls(self, p: str, pos: int, current_price: float) -> bool:
        """检查风险控制，返回是否允许交易"""
        # 检查交易次数限制
        if self.trade_count[p] >= PARAM["max_trades_per_day"]:
            return False  # 达到每日交易次数限制
            
        # 如果没有入场价格，设置当前价格为入场价格
        if self.entry_prices[p] == 0 and pos != 0:
            self.entry_prices[p] = current_price
            self.max_position_values[p] = abs(pos * current_price)
            return True
        
        # 如果有持仓，检查止损
        if pos != 0 and self.entry_prices[p] != 0:
            # 计算当前仓位价值
            current_value = abs(pos * current_price)
            
            # 更新最大仓位价值
            if current_value > self.max_position_values[p]:
                self.max_position_values[p] = current_value
            
            # 计算价格变动百分比
            price_change_pct = (current_price - self.entry_prices[p]) / self.entry_prices[p]
            
            # 检查止损
            if (pos > 0 and price_change_pct < -PARAM["stop_loss_pct"]) or \
               (pos < 0 and price_change_pct > PARAM["stop_loss_pct"]):
                return False  # 触发止损，不允许继续同方向交易
            
            # 检查最大回撤
            drawdown = self.max_position_values[p] - current_value
            if drawdown > PARAM["max_drawdown"]:
                return False  # 触发最大回撤，不允许继续同方向交易
        
        return True  # 通过风险控制，允许交易
    
    def execute_arbitrage(self, state: TradingState, opportunities: List[Dict], timestamp: int) -> Dict[str, List[Order]]:
        """执行套利策略"""
        result = {}
        
        for opp in opportunities:
            # 检查冷却期
            if timestamp - self.last_arb_trade.get(opp["sell_product"], 0) < PARAM["arb_cooldown"] or \
               timestamp - self.last_arb_trade.get(opp["buy_product"], 0) < PARAM["arb_cooldown"]:
                continue
                
            # 获取订单深度
            sell_depth = state.order_depths.get(opp["sell_product"])
            buy_depth = state.order_depths.get(opp["buy_product"])
            
            if not sell_depth or not buy_depth:
                continue
                
            # 获取最优价格
            sell_b, sell_a = best_bid_ask(sell_depth)
            buy_b, buy_a = best_bid_ask(buy_depth)
            
            if sell_b is None or buy_a is None:
                continue
                
            # 获取当前仓位
            sell_pos = state.position.get(opp["sell_product"], 0)
            buy_pos = state.position.get(opp["buy_product"], 0)
            
            # 检查风险控制
            if not self.check_risk_controls(opp["sell_product"], sell_pos, sell_b) or \
               not self.check_risk_controls(opp["buy_product"], buy_pos, buy_a):
                continue
                
            # 计算交易规模
            sell_limit = LIMIT[opp["sell_product"]]
            buy_limit = LIMIT[opp["buy_product"]]
            
            sell_size = min(
                int(sell_limit * PARAM["arb_size_frac"]),
                sell_limit + sell_pos,  # 考虑当前仓位
                sell_depth.buy_orders[sell_b]  # 考虑市场深度
            )
            
            buy_size = min(
                int(buy_limit * PARAM["arb_size_frac"]),
                buy_limit - buy_pos,  # 考虑当前仓位
                abs(buy_depth.sell_orders[buy_a])  # 考虑市场深度
            )
            
            # 确保买卖规模匹配
            size = min(sell_size, buy_size)
            
            if size <= 0:
                continue
                
            # 执行套利交易
            if opp["sell_product"] not in result:
                result[opp["sell_product"]] = []
            if opp["buy_product"] not in result:
                result[opp["buy_product"]] = []
                
            result[opp["sell_product"]].append(Order(opp["sell_product"], sell_b, -size))
            result[opp["buy_product"]].append(Order(opp["buy_product"], buy_a, size))
            
            # 更新最后交易时间
            self.last_arb_trade[opp["sell_product"]] = timestamp
            self.last_arb_trade[opp["buy_product"]] = timestamp
            
            # 更新交易计数
            self.trade_count[opp["sell_product"]] += 1
            self.trade_count[opp["buy_product"]] += 1
            
            # 更新入场价格
            self.entry_prices[opp["sell_product"]] = sell_b
            self.entry_prices[opp["buy_product"]] = buy_a
        
        return result
    
    def countertrend_strategy(self, p: str, depth: OrderDepth, pos: int, timestamp: int) -> List[Order]:
        """反趋势交易策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 记录仓位
        self.position_history[p].append(pos)
        
        # 计算RSI
        self.calculate_rsi(p)
        
        # 检测趋势和反转
        trend = self.detect_trend(p)
        reversal = self.detect_reversal(p)
        
        # 检查风险控制
        risk_ok = self.check_risk_controls(p, pos, mid)
        
        # 计算价差
        vol = self._vol(p)
        spread = max(PARAM["min_spread"], int(vol))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 计算交易规模
        max_position = int(LIMIT[p] * PARAM["position_limit_pct"])
        mm_size = max(1, int(LIMIT[p] * PARAM["mm_size_frac"]))
        counter_size = max(1, int(LIMIT[p] * PARAM["counter_size_frac"]))
        
        # 检查是否是套利对的一部分
        is_arb_pair = any(p in pair for pair in self.arb_pairs)
        
        # 如果是套利对的一部分，不进行反趋势交易
        if is_arb_pair:
            # 只进行常规做市
            if pos < max_position:
                orders.append(Order(p, buy_px, min(mm_size, max_position - pos)))
            if pos > -max_position:
                orders.append(Order(p, sell_px, -min(mm_size, max_position + pos)))
            return orders
        
        # 反趋势交易逻辑
        # 检查是否在冷却期
        cooldown_active = timestamp - self.last_counter_trade.get(p, 0) < PARAM["arb_cooldown"]
        
        if not cooldown_active and risk_ok:
            # 检查是否有强烈趋势
            if trend == 1 and self.rsi_values[p] > PARAM["rsi_overbought"]:
                # 强烈上升趋势，反向做空
                # 更积极地卖出
                sell_px = int(mid - 1)  # 降低卖出价格以确保成交
                orders.append(Order(p, sell_px, -counter_size))
                self.last_counter_trade[p] = timestamp
                self.entry_prices[p] = sell_px
                self.trade_count[p] += 1
                return orders  # 只做反向交易，不做常规做市
                
            elif trend == -1 and self.rsi_values[p] < PARAM["rsi_oversold"]:
                # 强烈下降趋势，反向做多
                # 更积极地买入
                buy_px = int(mid + 1)  # 提高买入价格以确保成交
                orders.append(Order(p, buy_px, counter_size))
                self.last_counter_trade[p] = timestamp
                self.entry_prices[p] = buy_px
                self.trade_count[p] += 1
                return orders  # 只做反向交易，不做常规做市
                
            # 检查是否有明确的反转信号
            elif reversal != 0:
                if reversal == 1:  # 下降趋势反转为上升
                    # 积极买入
                    buy_px = int(mid + 1)
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    self.entry_prices[p] = buy_px
                    self.trade_count[p] += 1
                    return orders
                    
                elif reversal == -1:  # 上升趋势反转为下降
                    # 积极卖出
                    sell_px = int(mid - 1)
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    self.entry_prices[p] = sell_px
                    self.trade_count[p] += 1
                    return orders
        
        # 如果没有反趋势交易信号，执行常规做市
        # 根据趋势调整价格
        if trend == 1:  # 上升趋势
            buy_px += 1
            sell_px += 1
        elif trend == -1:  # 下降趋势
            buy_px -= 1
            sell_px -= 1
        
        # 常规做市订单
        if pos < max_position:
            orders.append(Order(p, buy_px, min(mm_size, max_position - pos)))
        if pos > -max_position:
            orders.append(Order(p, sell_px, -min(mm_size, max_position + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 更新相关性
        self.update_correlations()
        
        # 寻找套利机会
        opportunities = self.find_arbitrage_opportunities()
        
        # 执行套利策略
        if opportunities:
            arb_orders = self.execute_arbitrage(state, opportunities, state.timestamp)
            result.update(arb_orders)
        
        # 对每个产品应用反趋势策略
        for p, depth in state.order_depths.items():
            # 如果已经在套利中交易了这个产品，跳过
            if p in result:
                continue
                
            if p in LIMIT:
                result[p] = self.countertrend_strategy(
                    p, 
                    depth, 
                    state.position.get(p, 0),
                    state.timestamp
                )
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
