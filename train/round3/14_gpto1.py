from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List
import math
import numpy as np

# We need norm.cdf for Black-Scholes:
from math import exp, log, sqrt
from statistics import NormalDist

def norm_cdf(x: float) -> float:
    """Cumulative distribution function for standard normal."""
    return NormalDist(mu=0, sigma=1).cdf(x)

class Trader:
    def __init__(self):
        # Time-to-expiry: Round 3 → 4 days left
        self.TTE = 4.0 / 365.0  
        
        # Position limits per the round's rules:
        self.position_limits = {
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }
        
        # How aggressively to scale trades relative to leftover capacity
        self.size_factor = 0.5  # 50% of leftover capacity each iteration

    def bs_call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Black-Scholes call option pricing formula (no interest rate / dividends).
        """
        if T <= 0 or S <= 0 or K <= 0 or sigma <= 1e-6:
            return max(S - K, 0)
        d1 = (math.log(S/K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm_cdf(d1) - K * norm_cdf(d2)

    def implied_volatility(self, S: float, K: float, T: float, market_price: float) -> float:
        """
        Very simple bracket search to invert BS formula and find implied volatility.
        If we can't find a solution, return None or a fallback.
        """
        if market_price < max(S - K, 0):
            # The price can't be below intrinsic value. If it is, there's no real vol solution.
            return None
        
        lower, upper = 1e-5, 3.0
        for _ in range(25):
            mid = 0.5 * (lower + upper)
            price = self.bs_call_price(S, K, T, mid)
            if price > market_price:
                upper = mid
            else:
                lower = mid
        return 0.5 * (lower + upper)

    def run(self, state: TradingState):
        """
        For each iteration (tick), we:
          1) Estimate the underlying price of VOLCANIC_ROCK from the best bid/ask midprice
          2) For each voucher, estimate implied vol
          3) Fit v vs. m = log(K/S)/sqrt(TTE) with a parabola
          4) Compare each voucher's vol to fitted vol -> if above, we 'sell'; if below, we 'buy'
          5) Output orders, sized up to self.size_factor * leftover capacity
        """
        orders_to_place = {}
        
        # ---------------
        # 1) Get the underlying price S from the best bid/ask of VOLCANIC_ROCK
        # ---------------
        rock_symbol = "VOLCANIC_ROCK"
        if rock_symbol not in state.order_depths:
            # No data => return empty
            return {}, 0, state.traderData
        
        rock_depth: OrderDepth = state.order_depths[rock_symbol]
        best_bid = None
        if len(rock_depth.buy_orders) > 0:
            best_bid = max(rock_depth.buy_orders.keys())
        best_ask = None
        if len(rock_depth.sell_orders) > 0:
            best_ask = min(rock_depth.sell_orders.keys())
        
        if best_bid is None or best_ask is None:
            # Can't compute a midprice. No trades
            return {}, 0, state.traderData
        
        S_mid = 0.5 * (best_bid + best_ask)

        # ---------------
        # 2) For each voucher, glean a single "market price" (premium) from the best bid-ask mid
        #    Then compute implied vol.
        # ---------------
        vouchers = [
            "VOLCANIC_ROCK_VOUCHER_9500",
            "VOLCANIC_ROCK_VOUCHER_9750",
            "VOLCANIC_ROCK_VOUCHER_10000",
            "VOLCANIC_ROCK_VOUCHER_10250",
            "VOLCANIC_ROCK_VOUCHER_10500"
        ]
        data_points = []
        
        for v in vouchers:
            if v not in state.order_depths:
                continue
            depth_v: OrderDepth = state.order_depths[v]
            if len(depth_v.buy_orders) == 0 or len(depth_v.sell_orders) == 0:
                continue  # no midprice if no quotes
            best_bid_v = max(depth_v.buy_orders.keys())
            best_ask_v = min(depth_v.sell_orders.keys())
            premium_mid = 0.5 * (best_bid_v + best_ask_v)
            
            # Extract strike from the symbol name:
            # e.g. VOLCANIC_ROCK_VOUCHER_9500 -> strike = 9500
            try:
                # last underscore part is the numeric
                strike = int(v.split('_')[-1])
            except:
                continue

            iv_est = self.implied_volatility(S_mid, strike, self.TTE, premium_mid)
            if iv_est is not None and not math.isnan(iv_est):
                # moneyness metric from the hint: m_t = log(K/S)/sqrt(TTE)
                m_t = math.log(strike / S_mid) / math.sqrt(self.TTE)
                data_points.append((v, strike, premium_mid, m_t, iv_est))

        # If no data, no trades
        if len(data_points) < 2:
            return {}, 0, state.traderData

        # ---------------
        # 3) Fit a parabola:  iv(m) ~ a*m^2 + b*m + c
        # ---------------
        # Separate x, y
        xvals = np.array([dp[3] for dp in data_points])  # the m_t
        yvals = np.array([dp[4] for dp in data_points])  # the implied vol
        fit_coef = np.polyfit(xvals, yvals, 2)  # [a, b, c]

        # 4) Compare each voucher's actual IV to fitted -> decide buy/sell
        #    iv_diff = actual - fitted
        #    If negative => underpriced => buy
        #    If positive => overpriced => sell
        diffs = []
        for (v, strike, prem, m_t, iv_est) in data_points:
            iv_fitted = np.polyval(fit_coef, m_t)
            iv_diff = iv_est - iv_fitted
            diffs.append((v, strike, prem, iv_est, iv_fitted, iv_diff))
        
        # Sort by iv_diff ascending => first is biggest negative -> best "buy"
        #                           => last is biggest positive -> best "sell"
        diffs.sort(key=lambda x: x[5])  # sort by iv_diff
        # pick 1 or 2 biggest negative for buy
        buy_candidates = diffs[:2]
        # pick 1 or 2 biggest positive for sell
        sell_candidates = diffs[-2:]

        # Prepare to create orders
        # We'll buy up to capacity * size_factor if negative diff, sell if positive
        result_orders: Dict[str, List[Order]] = {}

        for bc in buy_candidates:
            symbol_name = bc[0]
            iv_diff_val = bc[5]
            if iv_diff_val >= 0:
                break  # no longer "underpriced"
            # we want to BUY
            pos = state.position.get(symbol_name, 0)
            limit = self.position_limits[symbol_name]
            # leftover capacity
            can_buy = limit - pos
            if can_buy <= 0:
                continue
            trade_size = int(self.size_factor * can_buy)
            if trade_size <= 0:
                continue

            # We buy at best ask to ensure immediate fill
            voucher_depth = state.order_depths[symbol_name]
            if len(voucher_depth.sell_orders) == 0:
                continue
            best_ask_v = min(voucher_depth.sell_orders.keys())
            volume_at_ask = abs(voucher_depth.sell_orders[best_ask_v])  # remember negative
            quantity = min(trade_size, volume_at_ask)

            if quantity > 0:
                if symbol_name not in result_orders:
                    result_orders[symbol_name] = []
                # Positive quantity => buy
                result_orders[symbol_name].append(Order(symbol_name, best_ask_v, quantity))

        for sc in sell_candidates[::-1]:
            # sc is (symbol, K, prem, iv_est, iv_fit, diff)
            symbol_name = sc[0]
            iv_diff_val = sc[5]
            if iv_diff_val <= 0:
                break  # no longer "overpriced"
            # we want to SELL
            pos = state.position.get(symbol_name, 0)
            limit = self.position_limits[symbol_name]
            # leftover capacity if we want to SELL => how much we can go short
            can_sell = limit + pos  # e.g. if pos= -20, can_sell= 180 if limit=200
            if can_sell <= 0:
                continue
            trade_size = int(self.size_factor * can_sell)
            if trade_size <= 0:
                continue

            # Sell at best bid to ensure immediate fill
            voucher_depth = state.order_depths[symbol_name]
            if len(voucher_depth.buy_orders) == 0:
                continue
            best_bid_v = max(voucher_depth.buy_orders.keys())
            volume_at_bid = voucher_depth.buy_orders[best_bid_v]  # positive
            quantity = min(trade_size, volume_at_bid)

            if quantity > 0:
                if symbol_name not in result_orders:
                    result_orders[symbol_name] = []
                # Negative quantity => sell
                result_orders[symbol_name].append(Order(symbol_name, best_bid_v, -quantity))

        # That’s it for this iteration
        # No conversions in this example => 0
        # If you want to store dynamic data for next iteration, you can JSON-encode into traderData
        return result_orders, 0, state.traderData
