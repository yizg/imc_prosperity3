import math
import statistics
from typing import Dict, List, Any
import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import pandas as pd


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

        # fit the log limit
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


def get_mid_price(state: TradingState, symbol: str) -> float:
    order_depth = state.order_depths[symbol]
    buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(order_depth.sell_orders.items())

    popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
    popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

    return round((popular_buy_price + popular_sell_price) / 2)


def get_position(state: TradingState, product: str):
    return state.position.get(product, 0)


class Product:
    def __init__(self):
        self.params = {}
        self.take_sell_volume = 0
        self.take_buy_volume = 0
        self.limits = 50
        self.prices = []

    def get_fair_value(self, state: TradingState, product: str):
        return get_mid_price(state, product)

    def take_orders(self, state, product):
        output = []
        order_depth = state.order_depths[product]
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        best_ask_volume = order_depth.sell_orders[best_ask]
        best_bid_volume = order_depth.buy_orders[best_bid]

        fair_value = self.get_fair_value(state, product)
        if best_ask < fair_value:
            output.append(Order(product, best_ask, -best_ask_volume))
            self.take_sell_volume = abs(best_ask_volume)
        if best_bid > fair_value:
            output.append(Order(product, best_bid, -best_bid_volume))
            self.take_buy_volume = abs(best_bid_volume)

        return output

    def market_orders(self, state: TradingState, product: str):
        pass


class Kelp(Product):
    def __init__(self):
        super().__init__()

    def market_orders(self, state: TradingState, product: str):
        output = []
        order_depth = state.order_depths[product]

        # buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        # sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        fair_value = self.get_fair_value(state, product)

        limit = self.limits
        pos = get_position(state, product)

        asks_above_fair = [
            price for price in sell_orders.keys() if price > fair_value]
        bids_below_fair = [
            price for price in buy_orders.keys() if price < fair_value]

        best_ask_above_fair = min(asks_above_fair) if len(
            asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else None

        default_edge = 0
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= 3:
                ask = best_ask_above_fair  # join
                if pos < limit * -0.2:
                    ask += 1
            else:
                ask = best_ask_above_fair - 1  # penny
            # ask = best_ask_above_fair

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= 3:
                bid = best_bid_below_fair
                if pos > limit * 0.75:
                    bid -= 1
            else:
                bid = best_bid_below_fair + 1
            # bid = best_bid_below_fair

        buy_quantity = (limit - pos - self.take_buy_volume)
        if buy_quantity > 0:
            output.append(Order(product, round(bid),
                                buy_quantity))  # Buy order

        sell_quantity = (limit + pos - self.take_sell_volume)
        if sell_quantity > 0:
            output.append(Order(product, round(ask), -
                                sell_quantity))  # Sell order

        return output


class Resin(Product):
    def __init__(self):
        super().__init__()
        self.limits = 50

        self.SOFT_LIMIT = 0.9  # TODO ENTRE 0 ET 0.9
        self.MARGIN = 6

    def get_fair_value(self, state: TradingState, product: str):
        return 10000

    def take_orders(self, state, product):
        output = []
        order_depth = state.order_depths[product]
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        best_ask_volume = order_depth.sell_orders[best_ask]
        best_bid_volume = order_depth.buy_orders[best_bid]

        fair_value = self.get_fair_value(state, product)
        pos = get_position(state, product)

        if pos < 0:
            fair_value += 1
        elif pos > 0:
            fair_value -= 1

        if best_ask < fair_value:
            output.append(Order(product, best_ask, -best_ask_volume))
            self.take_sell_volume = abs(best_ask_volume)
        if best_bid > fair_value:
            output.append(Order(product, best_bid, -best_bid_volume))
            self.take_buy_volume = abs(best_bid_volume)

        return output

    def market_orders(self, state: TradingState, product: str):
        output = []
        order_depth = state.order_depths[product]

        # buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        # sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        fair_value = self.get_fair_value(state, product)

        limit = self.limits
        pos = get_position(state, product)

        asks_above_fair = [
            price for price in sell_orders.keys() if price >= fair_value]
        bids_below_fair = [
            price for price in buy_orders.keys() if price <= fair_value]

        best_ask_above_fair = min(asks_above_fair) if len(
            asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else None


        default_edge = 3  # no effect
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= self.MARGIN:
                ask = best_ask_above_fair  # join
                if pos < limit * -self.SOFT_LIMIT:
                    ask += 1
            else:
                ask = best_ask_above_fair - 1  # penny
         

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= self.MARGIN:
                bid = best_bid_below_fair
                if pos > limit * self.SOFT_LIMIT:
                    bid -= 1
               
            else:
                bid = best_bid_below_fair + 1
           

        buy_quantity = (limit - pos - self.take_buy_volume)
        if buy_quantity > 0:
            output.append(Order(product, round(bid),
                                buy_quantity)) 

        sell_quantity = (limit + pos - self.take_sell_volume)
        if sell_quantity > 0:
            output.append(Order(product, round(ask), -
                                sell_quantity)) 

        return output


class Squid(Product):
    def __init__(self):
        super().__init__()

    def market_orders(self, state: TradingState, product: str):
        output = []
        order_depth = state.order_depths[product]

        # buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        # sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        fair_value = self.get_fair_value(state, product)

        limit = self.limits
        pos = get_position(state, product)

        asks_above_fair = [
            price for price in sell_orders.keys() if price > fair_value]
        bids_below_fair = [
            price for price in buy_orders.keys() if price < fair_value]

        best_ask_above_fair = min(asks_above_fair) if len(
            asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else None

        # max_buy_price = fair_value - 1 if pos > limit * 0.5 else fair_value
        # min_sell_price = fair_value + 1 if pos < limit * -0.5 else fair_value

        default_edge = 3
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= 6:
                ask = best_ask_above_fair
                if pos < limit * -0.1:
                    ask += 1
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= 6:
                bid = best_bid_below_fair
                if pos > limit * 0.1:
                    bid -= 1
            else:
                bid = best_bid_below_fair + 1

        buy_quantity = limit - pos - self.take_buy_volume
        if buy_quantity > 0:
            output.append(Order(product, bid, buy_quantity))

        sell_quantity = limit + pos - self.take_sell_volume
        if sell_quantity > 0:
            output.append(Order(product, ask, -sell_quantity))

        return output


class Trader:
    def __init__(self):
        self.products_strategy = {
            "KELP": Kelp(),
            "SQUID_INK": Squid()
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 1
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = json.loads(state.traderData)

        for product in state.order_depths.keys():
            orders = []
            prices = traderObject.get(product, []) + [get_mid_price(state, product)]
            if len(prices) > 100:
                prices = prices[-100:]
            traderObject[product] = prices

            if product in self.products_strategy:
                strat = self.products_strategy[product]
                strat.prices = prices
                orders += strat.take_orders(state, product)
                orders += strat.market_orders(state, product)

            elif product == "RAINFOREST_RESIN":
                limit = 50
                pos = get_position(state, "RAINFOREST_RESIN")
                if pos < limit:
                    orders.append(Order("RAINFOREST_RESIN", 9998, limit - pos))
                if pos > -limit:
                    orders.append(Order("RAINFOREST_RESIN", 10002, -(limit + pos)))

            elif product == "PICNIC_BASKET1":
                cro = get_mid_price(state, "CROISSANTS")
                jam = get_mid_price(state, "JAMS")
                dje = get_mid_price(state, "DJEMBES")
                if None not in [cro, jam, dje]:
                    fair = 6 * cro + 3 * jam + 1 * dje
                    d = state.order_depths["PICNIC_BASKET1"]
                    if d.buy_orders and d.sell_orders:
                        best_bid = max(d.buy_orders)
                        best_ask = min(d.sell_orders)
                        spread = best_ask - best_bid
                        pos = get_position(state, "PICNIC_BASKET1")
                        if fair - best_ask > 4 and spread >= 2 and pos < 60:
                            orders.append(Order("PICNIC_BASKET1", best_ask, min(10, 60 - pos)))
                        if best_bid - fair > 4 and spread >= 2 and pos > -60:
                            orders.append(Order("PICNIC_BASKET1", best_bid, -min(10, 60 + pos)))

            elif product == "PICNIC_BASKET2":
                cro = get_mid_price(state, "CROISSANTS")
                jam = get_mid_price(state, "JAMS")
                if None not in [cro, jam]:
                    fair = 4 * cro + 2 * jam
                    d = state.order_depths["PICNIC_BASKET2"]
                    if d.buy_orders and d.sell_orders:
                        best_bid = max(d.buy_orders)
                        best_ask = min(d.sell_orders)
                        spread = best_ask - best_bid
                        pos = get_position(state, "PICNIC_BASKET2")
                        if fair - best_ask > 4 and spread >= 2 and pos < 100:
                            orders.append(Order("PICNIC_BASKET2", best_ask, min(10, 100 - pos)))
                        if best_bid - fair > 4 and spread >= 2 and pos > -100:
                            orders.append(Order("PICNIC_BASKET2", best_bid, -min(10, 100 + pos)))

            result[product] = orders

        trader_data = json.dumps(traderObject)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
