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

        # max_buy_price = fair_value - 1 if pos > limit * 0.5 else fair_value
        # min_sell_price = fair_value + 1 if pos < limit * -0.5 else fair_value

        default_edge = 1
        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= 3:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= 3:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if pos < limit * -0.1:
            ask += 1

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

    def get_fair_value(self, state: TradingState, product: str):
        return 10000


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
            if abs(best_ask_above_fair - fair_value) <= 3:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= 3:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

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
        # Ensure self.prices exists, likely initialized in the parent or main strategy runner
        # self.prices = [] # Example initialization
        self.std_deviation_threshold = 0.75
        self.skew_adjustment = 1
        self.position_limit_threshold = 0.1

    def market_orders(self, state: TradingState, product: str):
        output = []
        order_depth = state.order_depths[product]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        # --- 1. Calculate Fair Value ---
        fair_value = self.get_fair_value(state, product)
        if fair_value is None: # Handle case where fair value couldn't be determined
            return []

        # --- 2. Historical Context for Mean Reversion ---
        historical_mean = None
        historical_std_dev = None
        z_score = 0 # Default to no skew

        # Use the most recent MAX_WINDOW prices
        relevant_prices = self.prices

        if len(relevant_prices) >= 5: # Need a few data points for meaningful stats
            try:
                historical_mean = statistics.mean(relevant_prices)
                if len(relevant_prices) >= 2: # stdev needs at least 2 points
                    historical_std_dev = statistics.stdev(relevant_prices)
                else:
                    historical_std_dev = 1.0 # Default if only 1 point after slicing (unlikely with >=5 check)


                if historical_std_dev is not None and historical_std_dev > 1e-6: # Avoid division by zero/tiny std dev
                    z_score = (fair_value - historical_mean) / historical_std_dev
                # else: z_score remains 0

            except Exception as e:
                # Handle potential errors in statistics calculation if needed
                print(f"Error calculating statistics: {e}")
                z_score = 0 # Revert to no skew on error


        # --- 3. Calculate Base Bid/Ask (incorporating pennying/joining) ---
        limit = self.limits
        pos = get_position(state, product)

        asks_above_fair = [price for price in sell_orders.keys() if price > fair_value]
        bids_below_fair = [price for price in buy_orders.keys() if price < fair_value]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        default_edge = 1
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= 3:
                ask = best_ask_above_fair # Join
            else:
                ask = best_ask_above_fair - 1 # Penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= 3:
                bid = best_bid_below_fair # Join
            else:
                bid = best_bid_below_fair + 1 # Penny

        # Ensure ask > bid after initial calculation (can happen with wide spreads and pennying)
        if bid >= ask:
            # If overlap or touch, adjust slightly based on fair value
            bid = math.floor(fair_value - default_edge)
            ask = math.ceil(fair_value + default_edge)
            # Re-evaluate pennying/joining if necessary, or just use this wider spread

        # --- 4. Apply Skew based on Z-score ---
        bid_adjustment = 0
        ask_adjustment = 0

        if z_score > self.std_deviation_threshold: # Price is significantly HIGH -> be more aggressive selling
            ask_adjustment = -self.skew_adjustment # Lower ask price to sell more easily
        elif z_score < -self.std_deviation_threshold: # Price is significantly LOW -> be more aggressive buying
            bid_adjustment = self.skew_adjustment # Raise bid price to buy more easily

        bid += bid_adjustment
        ask += ask_adjustment

        # --- 5. Apply Position-Based Skew ---
        # This makes you less aggressive on the side where you already have a large position
        if pos > limit * self.position_limit_threshold:
            bid -= 1 # Less willing to buy more
        elif pos < -limit * self.position_limit_threshold:
            ask += 1 # Less willing to sell more

        # --- 6. Ensure Final Bid < Final Ask ---
        # Crucial check after all adjustments
        if bid >= ask:
            # If they crossed or met, reset them with a minimum spread around the adjusted midpoint
            mid = (bid + ask) / 2 # Midpoint *after* adjustments
            bid = math.floor(mid - 0.5) # Force a spread of at least 1
            ask = math.ceil(mid + 0.5)
            # Alternatively, could prioritize one side based on z_score or position

        # --- 7. Calculate Order Quantities ---
        # Adjust available limit based on potential fills from take orders (if applicable)
        available_buy_limit = limit - self.take_buy_volume
        available_sell_limit = limit - self.take_sell_volume # Magnitude of sell volume

        # Calculate how much more we *can* buy or sell to reach the limit
        buy_quantity = available_buy_limit - pos
        sell_quantity = available_sell_limit + pos # pos is negative if short

        # Ensure quantities are positive
        buy_quantity = max(0, buy_quantity)
        sell_quantity = max(0, sell_quantity)


        # --- 8. Generate Orders ---
        if buy_quantity > 0:
            output.append(Order(product, round(bid), buy_quantity)) # Buy order

        if sell_quantity > 0:
            output.append(Order(product, round(ask), -sell_quantity)) # Sell order

        return output

class Trader:
    def __init__(self):
        self.products_strategy = {
            "KELP": Kelp(),
            "RAINFOREST_RESIN": Resin(),
            "SQUID_INK": Squid()
        }

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 1
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = json.loads(state.traderData)

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():
            logger.print(get_mid_price(state, product))
            orders = []
            prices = traderObject.get(product, []) + \
                [get_mid_price(state, product)]
            if len(prices) > 100:
                prices = prices[-100:]
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
