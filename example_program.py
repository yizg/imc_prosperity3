from typing import Dict, List, Any
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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


GRIDSEARCH = True
PARAMS = {
    "KELP": {
        "bid_fair_diff_threshold": 2,
        "ask_fair_diff_threshold": 2,
        "default_edge": 0
    },
    "RAINFOREST_RESIN": {
        "bid_fair_diff_threshold": 5,
        "ask_fair_diff_threshold": 5,
        "default_edge": 4
    },
    "SQUID_INK": {
        "bid_fair_diff_threshold": 6,
        "ask_fair_diff_threshold": 6,
        "default_edge": 4
    },


}



def get_mid_price(state: TradingState, symbol: str) -> float:
    order_depth = state.order_depths[symbol]
    buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
    sell_orders = sorted(order_depth.sell_orders.items())

    popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
    popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

    return (popular_buy_price + popular_sell_price) / 2


class Trader:
    def __init__(self):
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
        }
        self.take_buy_volume = {
            "RAINFOREST_RESIN": 0,
            "KELP": 0,
            "SQUID_INK": 0,
        }
        self.take_sell_volume = {
            "RAINFOREST_RESIN": 0,
            "KELP": 0,
            "SQUID_INK": 0,
        }
        
        self.params = {}
        if GRIDSEARCH:
            for product in self.limits:
                config_path = f"config/{product}.json"
                with open(config_path, "r") as f:
                    self.params[product] = json.load(f)
        else:
            self.params = PARAMS

    def get_fair_value(self, state: TradingState, product: str):
        if product == "RAINFOREST_RESIN":
            fair_value = 10000
        elif product == "KELP":
            fair_value = get_mid_price(state, product)
        elif product == "SQUID_INK":
            fair_value = get_mid_price(state, product)
        return fair_value

    def get_position(self, state: TradingState, product: str):
        return state.position.get(product, 0)

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
            self.take_sell_volume[product] = abs(best_ask_volume)
        if best_bid > fair_value:
            output.append(Order(product, best_bid, -best_bid_volume))
            self.take_buy_volume[product] = abs(best_bid_volume)

        return output

    def market_orders(self, state: TradingState, product: str):
        output = []
        order_depth = state.order_depths[product]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        fair_value = self.get_fair_value(state, product)
        limit = self.limits[product]
        pos = self.get_position(state, product)

        param = self.params[product]
        ask_threshold = param["ask_fair_diff_threshold"]
        bid_threshold = param["bid_fair_diff_threshold"]
        default_edge = param["default_edge"]
        ask_volume_threshold = param.get("ask_volume_threshold", 2)
        bid_volume_threshold = param.get("bid_volume_threshold", 2)

        asks_above_fair = [price for price in sell_orders.keys() if price > fair_value]
        bids_below_fair = [price for price in buy_orders.keys() if price < fair_value]


        #asks_volume_counter = sum([0 if sell_orders[price] < ask_volume_threshold else 1 for price in asks_above_fair])
        #bids_volume_counter = sum([0 if buy_orders[price] < bid_volume_threshold else 1 for price in bids_below_fair])

        if product in state.position:
            cur_position = state.position[product]
        else:
             cur_position = 0

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        value = (best_ask + best_bid) / 2
        skew = -cur_position * value*0.0000005

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= ask_threshold:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= bid_threshold:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1
        #if product == "SQUID_INK":
        #    bid += skew
        #    ask += skew

        buy_quantity = (limit - pos - self.take_buy_volume[product])
        if pos > limit * 0.5:
            buy_quantity = buy_quantity
        if buy_quantity > 0:
            output.append(Order(product, round(bid), buy_quantity))

        sell_quantity = (limit + pos - self.take_sell_volume[product])
        if pos < limit * -0.5:
            sell_quantity = sell_quantity
        if sell_quantity > 0:
            output.append(Order(product, round(ask), -sell_quantity))

        return output

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 1
        trader_data = ""

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():
            orders = []

            take_orders = self.take_orders(state, product)
            market_orders = self.market_orders(state, product)
            orders = take_orders + market_orders

            # Add all the above the orders to the result dict
            result[product] = orders

        # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
