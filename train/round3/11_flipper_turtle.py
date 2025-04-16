from datamodel import OrderDepth, TradingState, Order, Symbol
from typing import Dict, List

class Product:
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"

class Logger:
    def print(self, *args, **kwargs):
        print(*args, **kwargs)

logger = Logger()

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.params = params
        self.voucher_strikes = [9500, 9750, 10000, 10250, 10500]
        
    def get_swmid(self, order_depth: OrderDepth) -> float:
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        return None

class FlipperTurtleStrategy:
    def __init__(self, trader):
        self.trader = trader
        self.params = {
            # 基础配置
            "initial_capital": 100000,
            "risk_per_trade": 0.3,
            "max_drawdown": 0.4,
            
            # 拍卖参数
            "bid_rounds": [
                {
                    "name": "bid_1",
                    "price": 165,  # 第一轮出价
                    "position_percentage": 0.5  # 使用50%资金
                },
                {
                    "name": "bid_2",
                    "price": 215,  # 第二轮出价
                    "position_percentage": 0.5  # 使用50%资金
                }
            ],
            
            # 风险控制
            "stop_loss_percentage": 0.1,
            "max_position_per_strike": 20,
            "max_total_position": 40
        }
        
    def calculate_position_size(self, bid_config: dict, available_capital: float) -> int:
        position_value = available_capital * bid_config["position_percentage"]
        position_size = int(position_value / bid_config["price"])
        
        risk_limit = int(available_capital * self.params["risk_per_trade"] / bid_config["price"])
        position_size = min(position_size, risk_limit)
        
        return position_size
        
    def execute_bid(self, bid_config: dict, available_capital: float) -> Dict[Symbol, List[Order]]:
        result = {}
        
        position_size = self.calculate_position_size(bid_config, available_capital)
        
        if position_size <= 0:
            return result
            
        # 执行出价
        if Product.VOLCANIC_ROCK not in result:
            result[Product.VOLCANIC_ROCK] = []
        result[Product.VOLCANIC_ROCK].append(Order(
            Product.VOLCANIC_ROCK,
            bid_config["price"],
            position_size
        ))
        
        logger.print(f"FLIPPER TURTLE {bid_config['name']}: Bid {bid_config['price']} with size {position_size}")
            
        return result
        
    def run(self, state: TradingState, trader_data: dict) -> Dict[Symbol, List[Order]]:
        result = {}
        
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return result
            
        # 获取当前价格
        current_price = self.trader.get_swmid(state.order_depths[Product.VOLCANIC_ROCK])
        if current_price is None:
            return result
            
        # 计算可用资金
        available_capital = self.params["initial_capital"]
        if "capital" in trader_data:
            available_capital = trader_data["capital"]
            
        # 执行出价
        for bid_config in self.params["bid_rounds"]:
            bid_orders = self.execute_bid(bid_config, available_capital)
            for symbol, orders in bid_orders.items():
                if symbol not in result:
                    result[symbol] = []
                result[symbol].extend(orders)
                
        return result 