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
            self.take_buy_volume = abs(best_ask_volume)
        if best_bid > fair_value:
            output.append(Order(product, best_bid, -best_bid_volume))
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


# class Squid(Product):

#     def __init__(self):
#         super().__init__()

#         self.std_deviation_threshold = 4
#         self.skew_adjustment = 1
#         self.position_limit_threshold = 0.1

#     def get_zscore(self, state: TradingState, product: str):
#         # --- 2. Historical Context for Mean Reversion ---
#         historical_mean = None
#         historical_std_dev = None
#         z_score = 0

#         relevant_prices = self.prices
#         fair_value = self.get_fair_value(state, product)
#         if len(relevant_prices) >= 5:  # Need a few data points for meaningful stats
#             try:
#                 historical_mean = statistics.mean(relevant_prices)
#                 if len(relevant_prices) >= 2:  # stdev needs at least 2 points
#                     historical_std_dev = statistics.stdev(relevant_prices)
#                 else:
#                     # Default if only 1 point after slicing (unlikely with >=5 check)
#                     historical_std_dev = 1.0

#                 # if historical_std_dev is not None and historical_std_dev > 1e-6:  # Avoid division by zero/tiny std dev
#                 z_score = (fair_value - historical_mean) / \
#                     historical_std_dev
#                 # else: z_score remains 0

#             except Exception as e:
#                 print(f"Error calculating statistics: {e}")
#                 z_score = 0

#         return z_score

#     def take_orders(self, state, product):
#         output = []
#         order_depth = state.order_depths[product]
#         best_ask = min(order_depth.sell_orders.keys())
#         best_bid = max(order_depth.buy_orders.keys())
#         best_ask_volume = order_depth.sell_orders[best_ask]
#         best_bid_volume = order_depth.buy_orders[best_bid]

#         fair_value = self.get_fair_value(state, product)
#         pos = get_position(state, product)
#         z_score = self.get_zscore(state, product)
#         logger.print(z_score)
#         max_buy_volume = min(self.limits - pos, abs(best_ask_volume))
#         max_sell_volume = min(self.limits + pos, abs(best_bid_volume))
#         if z_score > self.std_deviation_threshold:
#             output.append(Order(product, best_bid, -max_buy_volume))
#             self.take_buy_volume = abs(max_buy_volume)
#         elif z_score < -self.std_deviation_threshold:
#             output.append(Order(product, best_ask, max_sell_volume))
#             self.take_sell_volume = abs(max_sell_volume)
#         else:
#             if best_ask < fair_value:
#                 output.append(Order(product, best_ask, -best_ask_volume))
#                 self.take_sell_volume = abs(best_ask_volume)
#             if best_bid > fair_value:
#                 output.append(Order(product, best_bid, max_sell_volume))
#                 self.take_buy_volume = abs(max_sell_volume)

#         return output

#     def market_orders(self, state: TradingState, product: str):
#         output = []
#         order_depth = state.order_depths[product]
#         buy_orders = order_depth.buy_orders
#         sell_orders = order_depth.sell_orders

#         # --- 1. Calculate Fair Value ---
#         fair_value = self.get_fair_value(state, product)
#         z_score = self.get_zscore(state, product)

#         # --- 3. Calculate Base Bid/Ask (incorporating pennying/joining) ---
#         limit = self.limits
#         pos = get_position(state, product)

#         asks_above_fair = [
#             price for price in sell_orders.keys() if price > fair_value]
#         bids_below_fair = [
#             price for price in buy_orders.keys() if price < fair_value]

#         best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
#         best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

#         default_edge = 1
#         ask = round(fair_value + default_edge)
#         if best_ask_above_fair is not None:
#             if abs(best_ask_above_fair - fair_value) <= 5:
#                 ask = best_ask_above_fair  # Join
#                 # if pos < limit * -0.1:
#                 #     ask += 1
#             else:
#                 ask = best_ask_above_fair - 1  # Penny

#         bid = round(fair_value - default_edge)
#         if best_bid_below_fair is not None:
#             if abs(fair_value - best_bid_below_fair) <= 5:
#                 bid = best_bid_below_fair  # Join
#                 # if pos > limit * 0.1:
#                 #     bid -= 1
#             else:
#                 bid = best_bid_below_fair + 1  # Penny

#         # Ensure ask > bid after initial calculation (can happen with wide spreads and pennying)
#         # if bid >= ask:
#         #     # If overlap or touch, adjust slightly based on fair value
#         #     bid = math.floor(fair_value - default_edge)
#         #     ask = math.ceil(fair_value + default_edge)
#         #     # Re-evaluate pennying/joining if necessary, or just use this wider spread

#         # --- 4. Apply Skew based on Z-score ---

#         if z_score > self.std_deviation_threshold:  # Price is significantly HIGH -> be more aggressive selling
#             ask -= self.skew_adjustment  # Lower ask price to sell more easily
#             bid -= self.skew_adjustment  # Lower bid price to buy more easily
#         elif z_score < -self.std_deviation_threshold:  # Price is significantly LOW -> be more aggressive buying
#             ask += self.skew_adjustment  # Raise ask price to sell less easily
#             bid += self.skew_adjustment  # Raise bid price to buy less easily
#         else:
#             # --- 5. Apply Position-Based Skew ---
#             # This makes you less aggressive on the side where you already have a large position
#             if pos > limit * self.position_limit_threshold:
#                 bid -= 1  # Less willing to buy more
#             elif pos < -limit * self.position_limit_threshold:
#                 ask += 1  # Less willing to sell more

#         # # --- 6. Ensure Final Bid < Final Ask ---
#         # # Crucial check after all adjustments
#         # if bid >= ask:
#         #     # If they crossed or met, reset them with a minimum spread around the adjusted midpoint
#         #     mid = (bid + ask) / 2  # Midpoint *after* adjustments
#         #     bid = math.floor(mid - 0.5)  # Force a spread of at least 1
#         #     ask = math.ceil(mid + 0.5)
#         #     # Alternatively, could prioritize one side based on z_score or position

#         # --- 7. Calculate Order Quantities ---
#         # Adjust available limit based on potential fills from take orders (if applicable)
#         available_buy_limit = limit - self.take_buy_volume
#         available_sell_limit = limit - self.take_sell_volume  # Magnitude of sell volume

#         # Calculate how much more we *can* buy or sell to reach the limit
#         buy_quantity = available_buy_limit - pos
#         sell_quantity = available_sell_limit + pos  # pos is negative if short

#         # Ensure quantities are positive
#         buy_quantity = max(0, buy_quantity)
#         sell_quantity = max(0, sell_quantity)

#         # --- 8. Generate Orders ---
#         if buy_quantity > 0:
#             output.append(Order(product, round(bid),
#                           buy_quantity))  # Buy order

#         if sell_quantity > 0:
#             output.append(Order(product, round(ask), -
#                           sell_quantity))  # Sell order

#         return output


class Croissants(Product):
    def __init__(self):
        super().__init__()
        self.limits = 250
        self.SOFT_LIMIT = 0.1
        self.MARGIN = 2

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
        self.SOFT_LIMIT = 0.1
        self.MARGIN = 2

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
        self.SOFT_LIMIT = 0.2
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
        self.SOFT_LIMIT = 0.2
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


class Trader:
    def __init__(self):
        self.products_strategy = {
            "KELP": Kelp(),
            "RAINFOREST_RESIN": Resin(),
            "SQUID_INK": Squid(),
            "CROISSANTS": Croissants(),
            "JAMS": Jams(),
            "DJEMBES": Djembes(),
            "PICNIC_BASKET1": Picnic1(),
            "PICNIC_BASKET2": Picnic2()
            # 'Spread1': Spread1(),
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 1
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = json.loads(state.traderData)

        for product in state.order_depths.keys():
            # if product not in self.products_strategy:
            #     continue
            # logger.print(get_mid_price(state, product))
            orders = []
            prices = traderObject.get(product, []) + \
                [get_mid_price(state, product)]
            if len(prices) > 50:
                prices = prices[-50:]
            traderObject[product] = prices

            product_strategy = self.products_strategy[product]
            product_strategy.prices = prices

            take_orders = product_strategy.take_orders(state, product)
            market_orders = product_strategy.market_orders(state, product)
            orders = take_orders + market_orders

            # Add all the above the orders to the result dict
            result[product] = orders

        # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        trader_data = json.dumps(traderObject)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
