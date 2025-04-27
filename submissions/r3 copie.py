import math
import statistics
from typing import Dict, List, Any
import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import numpy as np
import pandas as pd
from statistics import NormalDist


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


def get_mid_price(state: TradingState, symbol: str) -> float:
    order_depth = state.order_depths[symbol]
    buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(order_depth.sell_orders.items())

    popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
    popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

    return round((popular_buy_price + popular_sell_price) / 2)


def get_position(state: TradingState, product: str):
    return state.position.get(product, 0)


def go_long(state: TradingState, quantity, product) -> None:
    order_depth = state.order_depths[product]
    price = max(order_depth.sell_orders.keys())

    return [Order(product, price, quantity)]


def go_short(state: TradingState, quantity, product):
    order_depth = state.order_depths[product]
    price = min(order_depth.buy_orders.keys())

    return [Order(product, price, -quantity)]


def BS_CALL(S, K, T, r, sigma):
    N = NormalDist().cdf
    d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * N(d1) - K * math.exp(-r*T) * N(d2)


def DELTA(S, K, T, r, sigma):
    N = NormalDist().cdf
    d1 = (math.log(S/K) + (r + sigma**2/2)*T) / (sigma*math.sqrt(T))
    return N(d1)


def vol_smile(x):
    a, b, c = 5.06109762, 0.02447706, 0.00800132
    return a*x**2 + b*x + c


def brent(f, a, b, tol=1e-12, max_iter=1000):
    """
    Brent's method to find a root of f in the interval [a, b]

    Parameters:
        f        : function for which to find the root
        a, b     : interval endpoints (must bracket the root)
        tol      : desired tolerance
        max_iter : maximum number of iterations

    Returns:
        root     : approximate root of f
    """
    fa = f(a)
    fb = f(b)

    # if fa * fb >= 0:
    #     raise ValueError(
    #         "Function must have opposite signs at endpoints a and b.")

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = e = b - a

    for _ in range(max_iter):
        if fb == 0:
            return b

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) \
                + (b * fa * fc) / ((fb - fa) * (fb - fc)) \
                + (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        conditions = [
            not (3 * a + b) / 4 < s < b if b > a else not b < s < (3 * a + b) / 4,
            (e is not None and abs(s - b) >= abs(e) / 2),
            (d is not None and abs(e) < tol),
            (d is not None and abs(fa) <= abs(fb)),
        ]

        if any(conditions):
            s = (a + b) / 2
            e = d = b - a
        else:
            d = e
            e = b - s

        fs = f(s)
        c, fc = b, fb

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        if abs(b - a) < tol:
            return b

    raise RuntimeError(
        "Maximum number of iterations reached without convergence.")


def get_implied_volatility(S, K, T, r, market_price):
    # Define the objective function to minimize
    def objective_function(sigma):
        return BS_CALL(S, K, T, r, sigma) - market_price

    implied_vol = brent(objective_function, 1e-8, 1)
    return implied_vol


class Product:
    def __init__(self):
        self.params = {}
        self.take_sell_volume = 0
        self.take_buy_volume = 0
        self.limits = 50
        self.prices = []
        self.ts_length = 100

    def get_fair_value(self, state: TradingState, product: str):
        return get_mid_price(state, product)

    def add_price(self, trader_data, state, product):
        new_price = get_mid_price(state, product)
        if trader_data is None:
            trader_data = [new_price]
        else:
            trader_data = trader_data + [new_price]
        self.prices = trader_data
        if len(trader_data) > self.ts_length:
            trader_data.pop(0)
        return trader_data

    def take_orders(self, state, product):
        output = []
        order_depth = state.order_depths[product]
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        best_ask_volume = order_depth.sell_orders[best_ask]
        best_bid_volume = order_depth.buy_orders[best_bid]

        fair_value = self.get_fair_value(state, product)
        pos = get_position(state, product)
        buy_volume = min(self.limits - pos, -best_ask_volume)
        sell_volume = min(self.limits + pos, best_bid_volume)

        if best_ask < fair_value:
            output.append(Order(product, best_ask, buy_volume))
            self.take_buy_volume = abs(best_ask_volume)
        if best_bid > fair_value:
            output.append(Order(product, best_bid, -sell_volume))
            self.take_sell_volume = abs(best_bid_volume)

        return output

    def market_orders(self, state: TradingState, product: str):
        pass


class Kelp(Product):
    def __init__(self):
        super().__init__()
        self.limits = 50
        self.SOFT_LIMIT = 0.5  # TODO ENTRE 0 ET 0.9
        self.MARGIN = 1

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
            asks_above_fair) > 0 else min(sell_orders.keys())
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else max(buy_orders.keys())

        default_edge = 1
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= self.MARGIN:
                ask = best_ask_above_fair  # join
                # if pos < limit * -0.2:
                #     ask += 1
            else:
                ask = best_ask_above_fair - 1  # penny
            # ask = best_ask_above_fair

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= self.MARGIN:
                bid = best_bid_below_fair
                # if pos > limit * 0.75:
                #     bid -= 1
            else:
                bid = best_bid_below_fair + 1
            # bid = best_bid_below_fair

        if pos < limit * -self.SOFT_LIMIT:
            ask = best_ask_above_fair + 1
            # bid = best_bid_below_fair + 1
        if pos > limit * self.SOFT_LIMIT:
            bid = best_bid_below_fair - 1

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
        self.MARGIN = 2
        # TODO ENTRE 0 ET 3 (en vrai on sait pas ça depend du bid ask spread mais ça change pas grand chose)

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
            self.take_buy_volume = abs(best_ask_volume)
        if best_bid > fair_value:
            output.append(Order(product, best_bid, -best_bid_volume))
            self.take_sell_volume = abs(best_bid_volume)

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
            price for price in sell_orders.keys() if price > fair_value]
        bids_below_fair = [
            price for price in buy_orders.keys() if price < fair_value]

        best_ask_above_fair = min(asks_above_fair) if len(
            asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else None

        # max_buy_price = fair_value - 1 if pos > limit * 0.5 else fair_value
        # min_sell_price = fair_value + 1 if pos < limit * -0.5 else fair_value
        default_edge = 1
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= self.MARGIN:
                ask = best_ask_above_fair  # join
                # if pos < limit * -self.SOFT_LIMIT:
                #     ask += 1
                # if pos > limit * self.SOFT_LIMIT:
                #     ask -= 1
            else:
                ask = best_ask_above_fair - 1  # penny
        # ask = best_ask_above_fair - 1
        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= self.MARGIN:
                bid = best_bid_below_fair
                # if pos > limit * self.SOFT_LIMIT:
                #     bid -= 1
                # if pos < limit * -self.SOFT_LIMIT:
                #     bid += 1
            else:
                bid = best_bid_below_fair + 1
            # bid = best_bid_below_fair + 1

        # if ask == best_ask_above_fair and pos < limit * -self.SOFT_LIMIT:
        #     ask += 1
        # if bid == best_bid_below_fair and pos > limit * self.SOFT_LIMIT:
        #     bid -= 1

        buy_quantity = (limit - pos - self.take_buy_volume)
        if buy_quantity > 0:
            output.append(Order(product, round(bid),
                                buy_quantity))  # Buy order

        sell_quantity = (limit + pos - self.take_sell_volume)
        if sell_quantity > 0:
            output.append(Order(product, round(ask), -
                                sell_quantity))  # Sell order

        return output


class Squid(Product):
    def __init__(self):
        super().__init__()
        self.limits = 50
        self.SOFT_LIMIT = 0.  # TODO ENTRE 0 ET 0.9
        self.MARGIN = 1

    # def get_fair_value(self, state: TradingState, product: str):

    #     prices = pd.Series(self.prices)
    #     fair_value = prices.ewm(
    #         alpha=0.3, min_periods=30).mean().iloc[-1]
    #     if fair_value > 0:
    #         return round(fair_value)
    #     else:
    #         return get_mid_price(state, product)

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
            asks_above_fair) > 0 else min(sell_orders.keys())
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else max(buy_orders.keys())
        # max_buy_price = fair_value - 1 if pos > limit * 0.5 else fair_value
        # min_sell_price = fair_value + 1 if pos < limit * -0.5 else fair_value

        default_edge = 1
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= self.MARGIN:
                ask = best_ask_above_fair  # join
                # if pos < limit * -0.1:
                #     ask += 1
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= self.MARGIN:
                bid = best_bid_below_fair
            #     if pos > limit * 0.1:
            #         bid -= 1
            else:
                bid = best_bid_below_fair + 1
        if pos < limit * -self.SOFT_LIMIT:
            ask = best_ask_above_fair + 1
            bid = best_bid_below_fair + 1
        if pos > limit * self.SOFT_LIMIT:
            bid = best_bid_below_fair - 1
            ask = best_ask_above_fair - 1

        buy_quantity = (limit - pos - self.take_buy_volume)
        if buy_quantity > 0:
            output.append(Order(product, round(bid),
                                buy_quantity))  # Buy order

        sell_quantity = (limit + pos - self.take_sell_volume)
        if sell_quantity > 0:
            output.append(Order(product, round(ask), -
                                sell_quantity))  # Sell order

        return output


class Croissants(Product):
    def __init__(self):
        super().__init__()
        self.limits = 250
        self.SOFT_LIMIT = 0.1
        self.MARGIN = 1

    def take_orders(self, state, product):
        # output = []
        # order_depth = state.order_depths[product]
        # best_ask = min(order_depth.sell_orders.keys())
        # best_bid = max(order_depth.buy_orders.keys())
        # best_ask_volume = order_depth.sell_orders[best_ask]
        # best_bid_volume = order_depth.buy_orders[best_bid]

        # fair_value = self.get_fair_value(state, product)
        # if best_ask < fair_value:
        #     output.append(Order(product, best_ask, -best_ask_volume))
        #     self.take_buy_volume = abs(best_ask_volume)
        # if best_bid > fair_value:
        #     output.append(Order(product, best_bid, -best_bid_volume))
        #     self.take_sell_volume = abs(best_bid_volume)

        target = - (6*get_position(state, "PICNIC_BASKET1") +
                    4 * get_position(state, "PICNIC_BASKET2"))
        if target > 0:
            target = max(target, self.limits)
        elif target < 0:
            target = max(target, -self.limits)

        diff = target - get_position(state, product)
        if diff > 0:
            output = go_long(state, diff, product)
            self.take_buy_volume = abs(target)
        elif diff < 0:
            output = go_short(state, -diff, product)
            self.take_sell_volume = abs(target)
        else:
            output = []

        logger.print(target)
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
            price for price in sell_orders.keys() if price > fair_value]
        bids_below_fair = [
            price for price in buy_orders.keys() if price < fair_value]

        best_ask_above_fair = min(asks_above_fair) if len(
            asks_above_fair) > 0 else min(sell_orders.keys())
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else max(buy_orders.keys())
        # max_buy_price = fair_value - 1 if pos > limit * 0.5 else fair_value
        # min_sell_price = fair_value + 1 if pos < limit * -0.5 else fair_value

        default_edge = 1
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= self.MARGIN:
                ask = best_ask_above_fair  # join
                # if pos < limit * -0.1:
                #     ask += 1
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= self.MARGIN:
                bid = best_bid_below_fair
            #     if pos > limit * 0.1:
            #         bid -= 1
            else:
                bid = best_bid_below_fair + 1

        if pos < limit * -self.SOFT_LIMIT:
            ask = best_ask_above_fair + 1
            # bid = best_bid_below_fair + 1
        if pos > limit * self.SOFT_LIMIT:
            bid = best_bid_below_fair - 1
            # ask = best_ask_above_fair - 1
        # if pos > limit * 0.1:
        #     bid -= 1
        # elif pos < limit * -0.1:
        #     ask += 1

        buy_quantity = (limit - pos - self.take_buy_volume)
        if buy_quantity > 0:
            output.append(Order(product, round(bid),
                                buy_quantity))  # Buy order

        sell_quantity = (limit + pos - self.take_sell_volume)
        if sell_quantity > 0:
            output.append(Order(product, round(ask), -
                                sell_quantity))  # Sell order

        return output


class Jams(Product):
    def __init__(self):
        super().__init__()
        self.limits = 350
        self.SOFT_LIMIT = 0.5
        self.MARGIN = 1

    def take_orders(self, state, product):
        # output = []
        # order_depth = state.order_depths[product]
        # best_ask = min(order_depth.sell_orders.keys())
        # best_bid = max(order_depth.buy_orders.keys())
        # best_ask_volume = order_depth.sell_orders[best_ask]
        # best_bid_volume = order_depth.buy_orders[best_bid]

        # fair_value = self.get_fair_value(state, product)
        # if best_ask < fair_value:
        #     output.append(Order(product, best_ask, -best_ask_volume))
        #     self.take_buy_volume = abs(best_ask_volume)
        # if best_bid > fair_value:
        #     output.append(Order(product, best_bid, -best_bid_volume))
        #     self.take_sell_volume = abs(best_bid_volume)

        target = - (3*get_position(state, "PICNIC_BASKET1") +
                    2 * get_position(state, "PICNIC_BASKET2"))
        if target > 0:
            target = max(target, self.limits)
        elif target < 0:
            target = max(target, -self.limits)

        diff = target - get_position(state, product)
        if diff > 0:
            output = go_long(state, diff, product)
            self.take_buy_volume = abs(target)
        elif diff < 0:
            output = go_short(state, -diff, product)
            self.take_sell_volume = abs(target)
        else:
            output = []

        # logger.print(target)
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
            price for price in sell_orders.keys() if price > fair_value]
        bids_below_fair = [
            price for price in buy_orders.keys() if price < fair_value]

        best_ask_above_fair = min(asks_above_fair) if len(
            asks_above_fair) > 0 else min(sell_orders.keys())
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else max(buy_orders.keys())
        # max_buy_price = fair_value - 1 if pos > limit * 0.5 else fair_value
        # min_sell_price = fair_value + 1 if pos < limit * -0.5 else fair_value

        default_edge = 1
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= self.MARGIN:
                ask = best_ask_above_fair  # join
                # if pos < limit * -0.1:
                #     ask += 1
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= self.MARGIN:
                bid = best_bid_below_fair
            #     if pos > limit * 0.1:
            #         bid -= 1
            else:
                bid = best_bid_below_fair + 1

        if pos < limit * -self.SOFT_LIMIT:
            ask = best_ask_above_fair + 1
            # bid = best_bid_below_fair + 1
        if pos > limit * self.SOFT_LIMIT:
            bid = best_bid_below_fair - 1
            # ask = best_ask_above_fair - 1
        # if pos > limit * 0.1:
        #     bid -= 1
        # elif pos < limit * -0.1:
        #     ask += 1

        buy_quantity = (limit - pos - self.take_buy_volume)
        if buy_quantity > 0:
            output.append(Order(product, round(bid),
                                buy_quantity))  # Buy order

        sell_quantity = (limit + pos - self.take_sell_volume)
        if sell_quantity > 0:
            output.append(Order(product, round(ask), -
                                sell_quantity))  # Sell order

        return output


class Djembes(Product):
    def __init__(self):
        super().__init__()
        self.limits = 60

    def take_orders(self, state, product):
        # output = []
        # order_depth = state.order_depths[product]
        # best_ask = min(order_depth.sell_orders.keys())
        # best_bid = max(order_depth.buy_orders.keys())
        # best_ask_volume = order_depth.sell_orders[best_ask]
        # best_bid_volume = order_depth.buy_orders[best_bid]

        # fair_value = self.get_fair_value(state, product)
        # if best_ask < fair_value:
        #     output.append(Order(product, best_ask, -best_ask_volume))
        #     self.take_buy_volume = abs(best_ask_volume)
        # if best_bid > fair_value:
        #     output.append(Order(product, best_bid, -best_bid_volume))
        #     self.take_sell_volume = abs(best_bid_volume)

        target = -get_position(state, "PICNIC_BASKET1") - \
            get_position(state, product)

        if target > 0:
            output = go_long(state, target, product)
            self.take_buy_volume = abs(target)
        elif target < 0:
            output = go_short(state, -target, product)
            self.take_sell_volume = abs(target)
        else:
            output = []

        logger.print(output)
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
            if abs(best_ask_above_fair - fair_value) <= 1:
                ask = best_ask_above_fair  # join
                if pos < limit * -0.5:
                    ask += 1
            else:
                ask = best_ask_above_fair - 1  # penny
            # ask = best_ask_above_fair

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= 1:
                bid = best_bid_below_fair
                if pos > limit * 0.5:
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


class Picnic1(Product):
    def __init__(self):
        super().__init__()
        self.limits = 60
        self.SOFT_LIMIT = 1
        self.MARGIN = 1

    def get_fair_value(self, state: TradingState, product: str):
        croissant_mid_price = get_mid_price(state, "CROISSANTS")
        djembes_mid_price = get_mid_price(state, "DJEMBES")
        jams_mid_price = get_mid_price(state, "JAMS")

        return 6 * croissant_mid_price + djembes_mid_price + 3 * jams_mid_price

    def market_orders(self, state: TradingState, product: str):
        output = []
        order_depth = state.order_depths[product]

        # buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        # sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        # self.get_fair_value(state, product)
        fair_value = get_mid_price(state, product)

        limit = self.limits
        pos = get_position(state, product)

        asks_above_fair = [
            price for price in sell_orders.keys() if price > fair_value]
        bids_below_fair = [
            price for price in buy_orders.keys() if price < fair_value]

        best_ask_above_fair = min(asks_above_fair) if len(
            asks_above_fair) > 0 else min(sell_orders.keys())
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else max(buy_orders.keys())
        # max_buy_price = fair_value - 1 if pos > limit * 0.5 else fair_value
        # min_sell_price = fair_value + 1 if pos < limit * -0.5 else fair_value

        default_edge = 1
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= self.MARGIN:
                ask = best_ask_above_fair  # join
                # if pos < limit * -0.1:
                #     ask += 1
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= self.MARGIN:
                bid = best_bid_below_fair
            #     if pos > limit * 0.1:
            #         bid -= 1
            else:
                bid = best_bid_below_fair + 1

        # if pos < limit * -self.SOFT_LIMIT:
        #     ask = best_ask_above_fair + 1
        #     # bid = best_bid_below_fair + 1
        # if pos > limit * self.SOFT_LIMIT:
        #     bid = best_bid_below_fair - 1
            # ask = best_ask_above_fair - 1
        # if pos > limit * 0.1:
        #     bid -= 1
        # elif pos < limit * -0.1:
        #     ask += 1

        buy_quantity = (limit - pos - self.take_buy_volume)
        if buy_quantity > 0:
            output.append(Order(product, round(bid),
                                buy_quantity))  # Buy order

        sell_quantity = (limit + pos - self.take_sell_volume)
        if sell_quantity > 0:
            output.append(Order(product, round(ask), -
                                sell_quantity))  # Sell order

        return output


class Picnic2(Product):
    def __init__(self):
        super().__init__()
        self.limits = 100
        self.SOFT_LIMIT = 1
        self.MARGIN = 1

    def get_fair_value(self, state: TradingState, product: str):
        croissant_mid_price = get_mid_price(state, "CROISSANTS")
        djembes_mid_price = get_mid_price(state, "DJEMBES")
        jams_mid_price = get_mid_price(state, "JAMS")

        return 4 * croissant_mid_price + 2 * jams_mid_price

    # def take_orders(self, state, product):
    #     output = []
    #     order_depth = state.order_depths[product]
    #     best_ask = min(order_depth.sell_orders.keys())
    #     best_bid = max(order_depth.buy_orders.keys())
    #     best_ask_volume = order_depth.sell_orders[best_ask]
    #     best_bid_volume = order_depth.buy_orders[best_bid]

    #     fair_value = self.get_fair_value(state, product)
    #     pos = get_position(state, product)

    #     if pos < 0:
    #         fair_value += 1
    #     elif pos > 0:
    #         fair_value -= 1

    #     if best_ask < fair_value:
    #         output.append(Order(product, best_ask, -best_ask_volume))
    #         self.take_sell_volume = abs(best_ask_volume)
    #     if best_bid > fair_value:
    #         output.append(Order(product, best_bid, -best_bid_volume))
    #         self.take_buy_volume = abs(best_bid_volume)
    #     logger.print(output)
    #     return output

    def market_orders(self, state: TradingState, product: str):
        output = []
        order_depth = state.order_depths[product]

        # buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        # sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        # self.get_fair_value(state, product)
        fair_value = get_mid_price(state, product)

        limit = self.limits
        pos = get_position(state, product)

        asks_above_fair = [
            price for price in sell_orders.keys() if price > fair_value]
        bids_below_fair = [
            price for price in buy_orders.keys() if price < fair_value]

        best_ask_above_fair = min(asks_above_fair) if len(
            asks_above_fair) > 0 else min(sell_orders.keys())
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else max(buy_orders.keys())
        # max_buy_price = fair_value - 1 if pos > limit * 0.5 else fair_value
        # min_sell_price = fair_value + 1 if pos < limit * -0.5 else fair_value

        default_edge = 1
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= self.MARGIN:
                ask = best_ask_above_fair  # join
                # if pos < limit * -0.1:
                #     ask += 1
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= self.MARGIN:
                bid = best_bid_below_fair
            #     if pos > limit * 0.1:
            #         bid -= 1
            else:
                bid = best_bid_below_fair + 1

        # if pos < limit * -self.SOFT_LIMIT:
        #     ask = best_ask_above_fair + 1
        #     bid = best_bid_below_fair + 1
        # if pos > limit * self.SOFT_LIMIT:
        #     bid = best_bid_below_fair - 1
        #     ask = best_ask_above_fair - 1

        # if pos > limit * 0.1:
        #     bid -= 1
        # elif pos < limit * -0.1:
        #     ask += 1

        buy_quantity = (limit - pos - self.take_buy_volume)

        if buy_quantity > 0:
            output.append(Order(product, round(bid),
                                buy_quantity))  # Buy order

        sell_quantity = (limit + pos - self.take_sell_volume)
        if sell_quantity > 0:
            output.append(Order(product, round(ask), -
                                sell_quantity))  # Sell order
        logger.print(limit, pos, self.take_buy_volume, self.take_sell_volume)
        logger.print(output)
        return output


class Spread1(Product):
    def __init__(self, status=None):
        super().__init__()
        self.limits = 100
        self.quantity = 30
        self.status = status

    def get_fair_value(self, state: TradingState, product: str):
        croissant_mid_price = get_mid_price(state, "CROISSANTS")
        djembes_mid_price = get_mid_price(state, "DJEMBES")
        jams_mid_price = get_mid_price(state, "JAMS")
        picnic1 = get_mid_price(state, "PICNIC_BASKET1")
        picnic2 = get_mid_price(state, "PICNIC_BASKET2")

        return 2*picnic1 - 3*picnic2 - 2*djembes_mid_price

    def get_arb_signal(self, state: TradingState, product: str):

        target_long = {"DJEMBES": -self.quantity*2,
                       "PICNIC_BASKET1": self.quantity*2, "PICNIC_BASKET2": -self.quantity*3}
        target_short = {"DJEMBES": self.quantity*2, "PICNIC_BASKET1": -
                        self.quantity*2, "PICNIC_BASKET2": self.quantity*3}
        target_neutral = {"DJEMBES": 0,
                          "PICNIC_BASKET1": 0, "PICNIC_BASKET2": 0}

        if self.status == None:
            if self.get_fair_value(state, product) > 150:
                self.status = "short"
                return target_short
            elif self.get_fair_value(state, product) < -150:
                self.status = "long"
                return target_long
            else:
                return target_neutral

        elif self.status == "long":
            if self.get_fair_value(state, product) > 0:
                self.status = None
                return target_neutral
            else:
                return target_long
        elif self.status == "short":
            if self.get_fair_value(state, product) < 0:
                self.status = None
                return target_neutral
            else:
                return target_short

    def get_arb_orders(self, state, product: str):
        orders = {"DJEMBES": [], "PICNIC_BASKET1": [], "PICNIC_BASKET2": []}
        target = self.get_arb_signal(state, product)

        if target != None:
            for p in orders.keys():
                diff = target[p] - get_position(state, p)
                logger.print(diff)
                if diff > 0:
                    orders[p] = self.go_long(state, diff, p)
                elif diff < 0:
                    orders[p] = self.go_short(state, -diff, p)
        return orders

        # if side == "short":
        #     orders['DJEMBES'] = self.go_long(state, 2*self.quantity, 'DJEMBES')
        #     orders['PICNIC_BASKET2'] = self.go_long(state, 3*self.quantity,
        #                                             'PICNIC_BASKET2')
        #     orders['PICNIC_BASKET1'] = self.go_short(state, 2*self.quantity,
        #                                              'PICNIC_BASKET1')
        # elif side == "long":
        #     orders['DJEMBES'] = self.go_short(state, 2*self.quantity,
        #                                       'DJEMBES')
        #     orders['PICNIC_BASKET2'] = self.go_short(state, 3*self.quantity,
        #                                              'PICNIC_BASKET2')
        #     orders['PICNIC_BASKET1'] = self.go_long(state, 2*self.quantity,
        #                                             'PICNIC_BASKET1')
        # return orders[product]

    # def buy(self, price: int, quantity: int) -> None:
    #     self.orders.append()

    # def sell(self, price: int, quantity: int) -> None:
    #     self.orders.append(Order(self.symbol, price, -quantity))

    def go_long(self, state: TradingState, quantity, product) -> None:
        order_depth = state.order_depths[product]
        price = max(order_depth.sell_orders.keys())

        # position = state.position.get(self.symbol, 0)
        # to_buy = self.limit - position

        return [Order(product, price, quantity)]

    def go_short(self, state: TradingState, quantity, product):
        order_depth = state.order_depths[product]
        price = min(order_depth.buy_orders.keys())

        # position = state.position.get(self.symbol, 0)
        # to_sell = self.limit + position

        return [Order(product, price, -quantity)]


class Rock(Product):
    def __init__(self):
        super().__init__()
        self.SOFT_LIMIT = 1
        self.MARGIN = 1
        self.nb_days = 5
        self.limits = 400
        self.voucher_limit = 200
        self.price_length = 5
        self.iv_length = 100
        self.VOUCHER = {
            'VOLCANIC_ROCK_VOUCHER_9500': 9500,
            'VOLCANIC_ROCK_VOUCHER_9750': 9750,
            'VOLCANIC_ROCK_VOUCHER_10000': 10000,
            'VOLCANIC_ROCK_VOUCHER_10250': 10250,
            'VOLCANIC_ROCK_VOUCHER_10500': 10500,
        }

        self.implied_volatility = {
            'VOLCANIC_ROCK_VOUCHER_9500': 0.013414,
            'VOLCANIC_ROCK_VOUCHER_9750': 0.010562,
            'VOLCANIC_ROCK_VOUCHER_10000': 0.008991,
            'VOLCANIC_ROCK_VOUCHER_10250': 0.008192,
            'VOLCANIC_ROCK_VOUCHER_10500':  0.008003,
        }
        # self.ivs = {
        #     'IV_VOLCANIC_ROCK_VOUCHER_9500': [],
        #     'IV_VOLCANIC_ROCK_VOUCHER_9750': [],
        #     'IV_VOLCANIC_ROCK_VOUCHER_10000': [],
        #     'IV_VOLCANIC_ROCK_VOUCHER_10250': [],
        #     'IV_VOLCANIC_ROCK_VOUCHER_10500': [],
        # }
    # def get_sigma(self):
    #     if len(self.prices) < 100:
    #         return 0.0084
    #     price = pd.Series(self.prices)
    #     return 0.0084  # price.pct_change().std() * np.sqrt(10000)

    def add_price(self, trader_data, state, product):
        if trader_data is None:
            trader_data = {}
        for v in list(self.VOUCHER.keys())+['VOLCANIC_ROCK']:
            if len(state.order_depths[v].buy_orders) == 0 or len(state.order_depths[v].sell_orders) == 0:
                continue
            new_price = get_mid_price(state, v)
            if trader_data.get(v, None) is None:
                trader_data[v] = [new_price]
            else:
                trader_data[v] = trader_data[v] + [new_price]
            if len(trader_data[v]) > self.price_length:
                trader_data[v].pop(0)

        for v in list(self.VOUCHER.keys()):
            if len(state.order_depths[v].buy_orders) == 0 or len(state.order_depths[v].sell_orders) == 0:
                continue
            try:
                iv = self.get_iv(state, v)
            except:
                continue
            k = 'IV_' + v
            if trader_data.get(k, None) is None:
                trader_data[k] = [iv]
            else:
                trader_data[k] = trader_data[k] + [iv]
            if len(trader_data[k]) > self.iv_length:
                trader_data[k].pop(0)

        self.prices = trader_data
        return trader_data

    def get_iv(self, state, product):
        iv = get_implied_volatility(
            get_mid_price(state, "VOLCANIC_ROCK"), self.VOUCHER[product], self.nb_days - state.timestamp/1000000, 0, get_mid_price(state, product))
        return iv

    def get_voucher_fair_value(self, state: TradingState, product: str):
        market_price = get_mid_price(state, product)
        S = get_mid_price(state, "VOLCANIC_ROCK")
        K = self.VOUCHER[product]
        T = self.nb_days - state.timestamp/1000000
        r = 0
        iv = statistics.mean(self.prices['IV_' + product])
        # mt = np.log(K/rock_price)/np.sqrt(T)
        # new_vol = vol_smile(mt)
        return round(BS_CALL(S, K, T, r, iv))

    def market_orders(self, state: TradingState, product: str):
        output = []
        order_depth = state.order_depths[product]

        # buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        # sell_orders = sorted(order_depth.sell_orders.items())
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        fair_value = self.get_voucher_fair_value(state, product)

        limit = self.limits
        pos = get_position(state, product)

        asks_above_fair = [
            price for price in sell_orders.keys() if price > fair_value]
        bids_below_fair = [
            price for price in buy_orders.keys() if price < fair_value]

        best_ask_above_fair = min(asks_above_fair) if len(
            asks_above_fair) > 0 else min(sell_orders.keys())
        best_bid_below_fair = max(bids_below_fair) if len(
            bids_below_fair) > 0 else max(buy_orders.keys())
        # max_buy_price = fair_value - 1 if pos > limit * 0.5 else fair_value
        # min_sell_price = fair_value + 1 if pos < limit * -0.5 else fair_value

        default_edge = 1
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= self.MARGIN:
                ask = best_ask_above_fair  # join
                # if pos < limit * -0.1:
                #     ask += 1
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= self.MARGIN:
                bid = best_bid_below_fair
            #     if pos > limit * 0.1:
            #         bid -= 1
            else:
                bid = best_bid_below_fair + 1

        # if pos < limit * -self.SOFT_LIMIT:
        #     ask = best_ask_above_fair + 1
        #     # bid = best_bid_below_fair + 1
        # if pos > limit * self.SOFT_LIMIT:
        #     bid = best_bid_below_fair - 1
            # ask = best_ask_above_fair - 1
        # if pos > limit * 0.1:
        #     bid -= 1
        # elif pos < limit * -0.1:
        #     ask += 1

        buy_quantity = (limit - pos - self.take_buy_volume)
        if buy_quantity > 0:
            output.append(Order(product, round(bid),
                                buy_quantity))  # Buy order

        sell_quantity = (limit + pos - self.take_sell_volume)
        if sell_quantity > 0:
            output.append(Order(product, round(ask), -
                                sell_quantity))  # Sell order

        return output

    def take_orders(self, state, product):
        # output = []
        # order_depth = state.order_depths[product]
        # best_ask = min(order_depth.sell_orders.keys())
        # best_bid = max(order_depth.buy_orders.keys())
        # best_ask_volume = order_depth.sell_orders[best_ask]
        # best_bid_volume = order_depth.buy_orders[best_bid]

        # fair_value = self.get_fair_value(state, product)
        # if best_ask < fair_value:
        #     output.append(Order(product, best_ask, -best_ask_volume))
        #     self.take_buy_volume = abs(best_ask_volume)
        # if best_bid > fair_value:
        #     output.append(Order(product, best_bid, -best_bid_volume))
        #     self.take_sell_volume = abs(best_bid_volume)

        # fair_value = self.get_voucher_fair_value(state, product)

        try:
            fair_value = self.get_voucher_fair_value(state, product)
            logger.print(fair_value)
        except:
            return []
        cur_value = get_mid_price(state, product)
        if cur_value > fair_value:
            target = -self.voucher_limit
        elif cur_value < fair_value:
            target = self.voucher_limit
        else:
            return []

        diff = target - get_position(state, product)
        if diff > 0:
            output = go_long(state, diff, product)
            self.take_buy_volume = abs(target)
        elif diff < 0:
            output = go_short(state, -diff, product)
            self.take_sell_volume = abs(target)
        else:
            return []

        logger.print(fair_value)
        return output

    def get_rock_orders(self, state):
        out = []
        delta = 0
        S = get_mid_price(state, "VOLCANIC_ROCK")
        T = self.nb_days - state.timestamp/1000000
        r = 0

        for product in self.VOUCHER:
            iv = statistics.mean(self.prices['IV_' + product])
            K = self.VOUCHER[product]
            delta += DELTA(S, K, T, r, iv) * get_position(state, product)
        delta = round(delta)
        logger.print(delta)
        if delta > self.limits:
            delta = -max(delta, self.limits)
        elif delta < 0:
            delta = -max(delta, -self.limits)
        else:
            delta = -delta

        diff = delta - get_position(state, "VOLCANIC_ROCK")
        if diff > 0:
            out = go_long(state, diff, "VOLCANIC_ROCK")
        elif diff < 0:
            out = go_short(state, -diff, "VOLCANIC_ROCK")
        else:
            return []

        return out

    def get_orders(self, state):
        orders = {}
        for product in self.VOUCHER:
            if len(state.order_depths[product].buy_orders) == 0 or len(state.order_depths[product].sell_orders) == 0:
                continue

            if product in state.order_depths:
                orders[product] = self.take_orders(state, product)
        # if self.get_rock_orders(state) != None:
        #     orders['VOLCANIC_ROCK'] = self.get_rock_orders(state)

        return orders


class Trader:
    def __init__(self):
        self.products_strategy = {
            "KELP": Kelp(),
            "RAINFOREST_RESIN": Resin(),
            "SQUID_INK": Squid(),
            # "CROISSANTS": Croissants(),
            # "JAMS": Jams(),
            # "DJEMBES": Djembes(),
            "PICNIC_BASKET1": Picnic1(),
            "PICNIC_BASKET2": Picnic2(),
            # 'Spread1': Spread1(),
            "VOLCANIC_ROCK": Rock(),
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 1
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = json.loads(state.traderData)

        for product in self.products_strategy:
            # logger.print(get_mid_price(state, product))
            product_strategy = self.products_strategy[product]
            orders = []
            prices = product_strategy.add_price(traderObject.get(
                product, None), state=state, product=product)
            traderObject[product] = prices

            if product == "VOLCANIC_ROCK":
                orders = product_strategy.get_orders(state)
                for k, v in orders.items():
                    result[k] = v
            else:

                take_orders = product_strategy.take_orders(state, product)
                market_orders = product_strategy.market_orders(state, product)
                orders = take_orders + market_orders

                result[product] = orders

        # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        trader_data = json.dumps(traderObject)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
