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

# 高波动性产品
HIGH_VOL_PRODUCTS = {
    "VOLCANIC_ROCK", "VOLCANIC_ROCK_VOUCHER_9500", "VOLCANIC_ROCK_VOUCHER_9750",
    "VOLCANIC_ROCK_VOUCHER_10000", "VOLCANIC_ROCK_VOUCHER_10250", "VOLCANIC_ROCK_VOUCHER_10500",
    "PICNIC_BASKET1", "PICNIC_BASKET2"
}

# 参数 - 更激进的设置
PARAM = {
    "trend_window": 20,           # 趋势检测窗口 (减少以更快检测趋势)
    "trend_threshold": 0.6,       # 趋势检测阈值 (降低以更容易触发趋势)
    "reversal_window": 3,         # 反转检测窗口 (减少以更快检测反转)
    "reversal_threshold": 0.7,    # 反转检测阈值 (降低以更容易触发反转)
    "position_limit_pct": 0.8,    # 仓位限制百分比 (增加以使用更多仓位)
    "mm_size_frac": 0.2,          # 做市规模 (增加以提高做市量)
    "counter_size_frac": 0.4,     # 反趋势交易规模 (增加以提高反趋势交易量)
    "min_spread": 1,              # 最小价差 (减少以更激进地定价)
    "vol_scale": 1.5,             # 波动率缩放因子 (增加以更好地利用波动)
    "rsi_period": 10,             # RSI周期 (减少以更快响应价格变化)
    "rsi_overbought": 65,         # RSI超买阈值 (降低以更早触发卖出信号)
    "rsi_oversold": 35,           # RSI超卖阈值 (提高以更早触发买入信号)
    "aggressive_take": True,      # 是否积极吃单
    "cp_memory": 30,              # 交易对手记忆长度
    "cp_threshold": 0.6,          # 交易对手分析阈值
    "cp_alpha": 0.7,              # 交易对手分析衰减因子
    "z_score_window": 20,         # Z分数计算窗口
    "z_score_threshold": 1.5      # Z分数阈值
}

# 火山岩产品特殊参数 - 更激进
VOLCANIC_PARAM = {
    "position_limit_pct": 0.9,    # 更高的仓位限制
    "counter_size_frac": 0.5,     # 更大的反趋势交易规模
    "min_spread": 1,              # 更小的价差
    "rsi_overbought": 60,         # 更低的RSI超买阈值
    "rsi_oversold": 40,           # 更高的RSI超卖阈值
    "aggressive_take": True       # 积极吃单
}

def best_bid_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """获取最优买卖价"""
    return (max(depth.buy_orders) if depth.buy_orders else None,
            min(depth.sell_orders) if depth.sell_orders else None)

class CounterpartyAnalyzer:
    """分析交易对手行为以识别有利的交易机会"""
    
    def __init__(self):
        # 跟踪与每个交易对手的交易
        self.cp_trades = defaultdict(list)  # {counterparty_id: [(product, price, quantity, timestamp), ...]}
        # 跟踪每个交易对手的盈利能力评分
        self.cp_score = defaultdict(float)  # {counterparty_id: score}
        # 跟踪与每个交易对手交易后的价格走势
        self.cp_price_impact = defaultdict(list)  # {counterparty_id: [(price_before, price_after), ...]}
        # 有利交易的交易对手集合
        self.profitable_cps = set()
        # 不利交易的交易对手集合
        self.unprofitable_cps = set()
        # 最后的中间价格
        self.last_mid_price = {}
        
    def record_trade(self, product: str, trade_price: int, trade_qty: int,
                    counterparty: str, timestamp: int, mid_price: float):
        """记录与交易对手的交易"""
        if not counterparty:
            return
            
        # 存储交易信息
        self.cp_trades[counterparty].append((product, trade_price, trade_qty, timestamp))
        
        # 只保留最近的交易
        if len(self.cp_trades[counterparty]) > PARAM["cp_memory"]:
            self.cp_trades[counterparty].pop(0)
            
        # 更新价格影响数据
        if product in self.last_mid_price:
            price_before = self.last_mid_price[product]
            self.cp_price_impact[counterparty].append((price_before, mid_price))
            
            # 只保留最近的价格影响
            if len(self.cp_price_impact[counterparty]) > PARAM["cp_memory"]:
                self.cp_price_impact[counterparty].pop(0)
                
    def update_counterparty_scores(self):
        """更新所有交易对手的盈利能力评分"""
        for cp, price_impacts in self.cp_price_impact.items():
            if not price_impacts:
                continue
                
            # 计算与该交易对手交易后价格有利变动的频率
            favorable_moves = 0
            total_moves = len(price_impacts)
            
            for before, after in price_impacts:
                # 检查最近的交易
                recent_trades = self.cp_trades[cp][-5:]
                
                for _, _, qty, _ in recent_trades:
                    # 如果我们买入(正qty)，价格上涨是有利的
                    if qty > 0 and after > before:
                        favorable_moves += 1
                    # 如果我们卖出(负qty)，价格下跌是有利的
                    elif qty < 0 and after < before:
                        favorable_moves += 1
                        
            # 计算有利变动的比例
            if total_moves > 0:
                favorable_ratio = favorable_moves / total_moves
                
                # 更新评分 (使用指数移动平均)
                alpha = PARAM["cp_alpha"]
                self.cp_score[cp] = alpha * favorable_ratio + (1 - alpha) * self.cp_score.get(cp, 0.5)
                
                # 更新交易对手分类
                if self.cp_score[cp] > PARAM["cp_threshold"]:
                    self.profitable_cps.add(cp)
                    if cp in self.unprofitable_cps:
                        self.unprofitable_cps.remove(cp)
                elif self.cp_score[cp] < (1 - PARAM["cp_threshold"]):
                    self.unprofitable_cps.add(cp)
                    if cp in self.profitable_cps:
                        self.profitable_cps.remove(cp)

class Trader:
    def __init__(self):
        self.prices = defaultdict(list)
        self.returns = defaultdict(list)  # 价格回报率
        self.position_history = defaultdict(list)  # 仓位历史
        self.rsi_values = defaultdict(float)  # RSI值
        self.last_counter_trade = defaultdict(int)  # 上次反向交易的时间戳
        self.cp_analyzer = CounterpartyAnalyzer()  # 交易对手分析器
        self.z_scores = defaultdict(float)  # Z分数
        
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
        # 使用产品特定参数
        rsi_period = VOLCANIC_PARAM["rsi_period"] if p in HIGH_VOL_PRODUCTS else PARAM["rsi_period"]
        
        if len(self.prices[p]) < rsi_period + 1:
            self.rsi_values[p] = 50  # 默认中性值
            return
            
        # 计算价格变动
        price_changes = [self.prices[p][i] - self.prices[p][i-1] for i in range(1, len(self.prices[p]))]
        
        # 只使用最近的价格变动
        price_changes = price_changes[-rsi_period:]
        
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
    
    def calculate_z_score(self, p: str):
        """计算Z分数"""
        prices = self.prices[p]
        window = PARAM["z_score_window"]
        
        if len(prices) < window:
            self.z_scores[p] = 0
            return
            
        # 计算移动平均和标准差
        recent_prices = prices[-window:]
        mean = sum(recent_prices) / len(recent_prices)
        std = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 1
        
        # 计算当前价格的Z分数
        if std > 0:
            self.z_scores[p] = (prices[-1] - mean) / std
        else:
            self.z_scores[p] = 0
    
    def detect_trend(self, p: str) -> int:
        """检测市场趋势"""
        # 使用产品特定参数
        trend_window = VOLCANIC_PARAM["trend_window"] if p in HIGH_VOL_PRODUCTS else PARAM["trend_window"]
        trend_threshold = VOLCANIC_PARAM["trend_threshold"] if p in HIGH_VOL_PRODUCTS else PARAM["trend_threshold"]
        
        prices = self.prices[p]
        if len(prices) < trend_window:
            return 0  # 数据不足
            
        recent_prices = prices[-trend_window:]  
        up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
        down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
        
        up_ratio = up_moves / (len(recent_prices) - 1)
        down_ratio = down_moves / (len(recent_prices) - 1)
        
        if up_ratio > trend_threshold:
            return 1  # 上升趋势
        elif down_ratio > trend_threshold:
            return -1  # 下降趋势
        return 0  # 中性
    
    def detect_reversal(self, p: str) -> int:
        """检测市场反转"""
        # 使用产品特定参数
        reversal_window = VOLCANIC_PARAM["reversal_window"] if p in HIGH_VOL_PRODUCTS else PARAM["reversal_window"]
        reversal_threshold = VOLCANIC_PARAM["reversal_threshold"] if p in HIGH_VOL_PRODUCTS else PARAM["reversal_threshold"]
        
        prices = self.prices[p]
        if len(prices) < reversal_window + 5:
            return 0  # 数据不足
            
        # 检查之前的趋势
        prev_trend = self.detect_trend(p)
        if prev_trend == 0:
            return 0  # 没有明确趋势，无法判断反转
            
        # 检查最近的价格变动
        recent_prices = prices[-reversal_window:]
        prev_prices = prices[-reversal_window-5:-reversal_window]
        
        if prev_trend == 1:  # 之前是上升趋势
            # 检查是否开始下跌
            down_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] < recent_prices[i-1])
            down_ratio = down_moves / (len(recent_prices) - 1)
            
            if down_ratio > reversal_threshold:
                return -1  # 上升趋势反转为下降
        
        elif prev_trend == -1:  # 之前是下降趋势
            # 检查是否开始上涨
            up_moves = sum(1 for i in range(1, len(recent_prices)) if recent_prices[i] > recent_prices[i-1])
            up_ratio = up_moves / (len(recent_prices) - 1)
            
            if up_ratio > reversal_threshold:
                return 1  # 下降趋势反转为上升
                
        return 0  # 没有检测到反转
    
    def analyze_recent_trades(self, product: str, trades, mid_price: float):
        """分析最近的交易以识别交易对手模式"""
        if not trades:
            return
            
        for trade in trades:
            # 检查交易是否有counter_party属性
            counterparty = getattr(trade, 'counter_party', None)
            if not counterparty:
                # 尝试在回测环境中获取buyer/seller
                counterparty = getattr(trade, 'buyer', None) or getattr(trade, 'seller', None)
                
            self.cp_analyzer.record_trade(
                product,
                trade.price,
                trade.quantity,
                counterparty,
                getattr(trade, 'timestamp', 0),
                mid_price
            )
            
        # 更新交易对手评分
        self.cp_analyzer.update_counterparty_scores()
    
    def adjust_price_for_counterparty(self, base_price: int, is_buy: bool, counterparties: Set[str]) -> int:
        """根据交易对手分析调整价格"""
        # 如果没有交易对手信息，使用基础价格
        if not counterparties:
            return base_price
            
        # 计算有利和不利交易对手的数量
        profitable_count = sum(1 for cp in counterparties if cp in self.cp_analyzer.profitable_cps)
        unprofitable_count = sum(1 for cp in counterparties if cp in self.cp_analyzer.unprofitable_cps)
        
        # 如果有利交易对手多于不利交易对手
        if profitable_count > unprofitable_count:
            # 更激进的定价
            return base_price + (2 if is_buy else -2)
        # 如果不利交易对手多于有利交易对手
        elif unprofitable_count > profitable_count:
            # 更保守的定价
            return base_price + (-1 if is_buy else 1)
            
        return base_price
    
    def get_active_counterparties(self, own_trades, market_trades):
        """获取活跃的交易对手"""
        active_cps = set()
        
        # 从自己的交易中获取交易对手
        if own_trades:
            for trade in own_trades:
                cp = getattr(trade, 'counter_party', None)
                if cp:
                    active_cps.add(cp)
                    
        # 从市场交易中获取交易对手
        if market_trades:
            for trade in market_trades:
                buyer = getattr(trade, 'buyer', None)
                seller = getattr(trade, 'seller', None)
                if buyer:
                    active_cps.add(buyer)
                if seller:
                    active_cps.add(seller)
                    
        return active_cps
    
    def countertrend_strategy(self, p: str, depth: OrderDepth, pos: int, timestamp: int, 
                             own_trades=None, market_trades=None) -> List[Order]:
        """反趋势交易策略"""
        orders = []
        mid = self._mid(depth)
        if mid is None: return orders
        
        # 记录价格
        self.prices[p].append(mid)
        
        # 记录仓位
        self.position_history[p].append(pos)
        
        # 计算RSI和Z分数
        self.calculate_rsi(p)
        self.calculate_z_score(p)
        
        # 获取活跃的交易对手
        active_counterparties = self.get_active_counterparties(own_trades, market_trades)
        
        # 分析最近的交易
        self.analyze_recent_trades(p, own_trades, mid)
        
        # 检测趋势和反转
        trend = self.detect_trend(p)
        reversal = self.detect_reversal(p)
        
        # 使用产品特定参数
        if p in HIGH_VOL_PRODUCTS:
            min_spread = VOLCANIC_PARAM["min_spread"]
            position_limit_pct = VOLCANIC_PARAM["position_limit_pct"]
            mm_size_frac = VOLCANIC_PARAM["mm_size_frac"]
            counter_size_frac = VOLCANIC_PARAM["counter_size_frac"]
            rsi_overbought = VOLCANIC_PARAM["rsi_overbought"]
            rsi_oversold = VOLCANIC_PARAM["rsi_oversold"]
            aggressive_take = VOLCANIC_PARAM["aggressive_take"]
        else:
            min_spread = PARAM["min_spread"]
            position_limit_pct = PARAM["position_limit_pct"]
            mm_size_frac = PARAM["mm_size_frac"]
            counter_size_frac = PARAM["counter_size_frac"]
            rsi_overbought = PARAM["rsi_overbought"]
            rsi_oversold = PARAM["rsi_oversold"]
            aggressive_take = PARAM["aggressive_take"]
        
        # 计算价差
        vol = self._vol(p)
        spread = max(min_spread, int(vol))
        
        # 基础价格
        buy_px = int(mid - spread)
        sell_px = int(mid + spread)
        
        # 根据交易对手调整价格
        buy_px = self.adjust_price_for_counterparty(buy_px, True, active_counterparties)
        sell_px = self.adjust_price_for_counterparty(sell_px, False, active_counterparties)
        
        # 计算交易规模
        max_position = int(LIMIT[p] * position_limit_pct)
        mm_size = max(1, int(LIMIT[p] * mm_size_frac))
        counter_size = max(1, int(LIMIT[p] * counter_size_frac))
        
        # 反趋势交易逻辑
        if p in HIGH_VOL_PRODUCTS:  # 只对高波动性产品应用反趋势策略
            # 检查是否有强烈趋势
            if trend == 1 and self.rsi_values[p] > rsi_overbought:
                # 强烈上升趋势，反向做空
                # 更积极地卖出
                sell_px = int(mid - 1)  # 降低卖出价格以确保成交
                orders.append(Order(p, sell_px, -counter_size))
                self.last_counter_trade[p] = timestamp
                return orders  # 只做反向交易，不做常规做市
                
            elif trend == -1 and self.rsi_values[p] < rsi_oversold:
                # 强烈下降趋势，反向做多
                # 更积极地买入
                buy_px = int(mid + 1)  # 提高买入价格以确保成交
                orders.append(Order(p, buy_px, counter_size))
                self.last_counter_trade[p] = timestamp
                return orders  # 只做反向交易，不做常规做市
                
            # 检查是否有明确的反转信号
            elif reversal != 0:
                if reversal == 1:  # 下降趋势反转为上升
                    # 积极买入
                    buy_px = int(mid + 1)
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders
                    
                elif reversal == -1:  # 上升趋势反转为下降
                    # 积极卖出
                    sell_px = int(mid - 1)
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders
                    
            # 检查Z分数是否超过阈值
            if abs(self.z_scores[p]) > PARAM["z_score_threshold"]:
                if self.z_scores[p] > PARAM["z_score_threshold"]:  # 价格过高
                    # 卖出
                    sell_px = int(mid)  # 更激进的价格
                    orders.append(Order(p, sell_px, -counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders
                elif self.z_scores[p] < -PARAM["z_score_threshold"]:  # 价格过低
                    # 买入
                    buy_px = int(mid)  # 更激进的价格
                    orders.append(Order(p, buy_px, counter_size))
                    self.last_counter_trade[p] = timestamp
                    return orders
        
        # 如果没有反趋势交易信号，执行常规做市
        # 根据趋势调整价格
        if trend == 1:  # 上升趋势
            buy_px += 1
            sell_px += 1
        elif trend == -1:  # 下降趋势
            buy_px -= 1
            sell_px -= 1
        
        # 积极吃单逻辑
        if aggressive_take:
            b, a = best_bid_ask(depth)
            if b is not None and a is not None:
                # 如果市场买价高于我们的卖价，吃掉它
                if b > sell_px and pos > -max_position:
                    sell_qty = min(counter_size, max_position + pos, depth.buy_orders[b])
                    if sell_qty > 0:
                        orders.append(Order(p, b, -sell_qty))
                        
                # 如果市场卖价低于我们的买价，吃掉它
                if a < buy_px and pos < max_position:
                    buy_qty = min(counter_size, max_position - pos, abs(depth.sell_orders[a]))
                    if buy_qty > 0:
                        orders.append(Order(p, a, buy_qty))
        
        # 常规做市订单
        if pos < max_position:
            orders.append(Order(p, buy_px, min(mm_size, max_position - pos)))
        if pos > -max_position:
            orders.append(Order(p, sell_px, -min(mm_size, max_position + pos)))
        
        return orders
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        """主交易逻辑"""
        result: Dict[str, List[Order]] = {}
        
        # 对每个产品应用反趋势策略
        for p, depth in state.order_depths.items():
            if p in LIMIT:
                own_trades = state.own_trades.get(p, [])
                market_trades = state.market_trades.get(p, [])
                
                result[p] = self.countertrend_strategy(
                    p, 
                    depth, 
                    state.position.get(p, 0),
                    state.timestamp,
                    own_trades,
                    market_trades
                )
        
        # 不进行转换
        conversions = 0
        
        return result, conversions, state.traderData
