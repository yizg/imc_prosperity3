from typing import Dict, List, Any
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


def get_mid_price(state: TradingState, symbol: str) -> float:
    order_depth = state.order_depths[symbol]
    buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(order_depth.sell_orders.items())

    popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
    popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

    return (popular_buy_price + popular_sell_price) / 2


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(
                        state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [
                order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        print(observations)
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 1
        trader_data = ""
        # logger.print(state.timestamp)
        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():
            # Retrieve the Order Depth containing all the market BUY and SELL orders
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            # if product not in state.market_trades:
            #     continue

            # market_trades = state.market_trades[product]

            # if len(market_trades) > 0:
            #     # Get the last trade price
            #     last_trade_price = market_trades[-1].price
            #     # acceptable_price = min(order_depth.sell_orders.keys()) + max(
            #     #     order_depth.buy_orders.keys()) / 2
            if product in state.position:
                cur_position = state.position[product]
            else:
                cur_position = 0
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]
            best_bid_volume = order_depth.buy_orders[best_bid]
            value = (best_ask + best_bid) / 2
            spread = (best_ask - best_bid) + value*0.005
            skew = -cur_position * value*0.002
            ask_price = int(value + skew + spread/2)
            bid_price = int(value + skew - spread/2)
            logger.print("skew", skew, " spread", spread)
            logger.print("ask_price", ask_price, " bid_price", bid_price)
            # # If statement checks if there are any SELL orders in the market
            # if len(order_depth.sell_orders) > 0:

            #     # Sort all the available sell orders by their price,
            #     # and select only the sell order with the lowest price
            #     best_ask = min(order_depth.sell_orders.keys())
            #     best_ask_volume = order_depth.sell_orders[best_ask]

            if product == "RAINFOREST_RESIN":
                if best_ask < 10000:
                    orders.append(Order(product, best_ask, -best_ask_volume))
                    # logger.print("SELL", str(-1) + "x", best_ask)
                if best_bid > 10000:
                    orders.append(Order(product, best_bid, -best_bid_volume))
                    # logger.print("BUY", str(-1) + "x", best_bid)

            else:
                price = get_mid_price(state, product)
                if best_ask < price:
                    orders.append(Order(product, best_ask, -best_ask_volume))
                    # logger.print("SELL", str(-1) + "x", best_ask)
                if best_bid > price:
                    orders.append(Order(product, best_bid, -best_bid_volume))
                    # logger.print("BUY", str(-1) + "x", best_bid)

            # Add all the above the orders to the result dict
            result[product] = orders

        # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
