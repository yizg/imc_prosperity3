
from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict, Tuple
import numpy as np

POSITION_LIMITS = {
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "RAINFOREST_RESIN": 50,
    "KELP": 50,
}

class Trader:
    def __init__(self):
        self.price_history = {"KELP": []}

    def get_mid(self, d: OrderDepth) -> float:
        if d.buy_orders and d.sell_orders:
            return (max(d.buy_orders) + min(d.sell_orders)) / 2
        return None

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        orders = {}
        conversions = 0

        # trading of rainforest
        d = state.order_depths.get("RAINFOREST_RESIN")
        if d:
            pos = state.position.get("RAINFOREST_RESIN", 0)
            limit = POSITION_LIMITS["RAINFOREST_RESIN"]
            o = []
            if pos < limit:
                o.append(Order("RAINFOREST_RESIN", 9998, limit - pos))
            if pos > -limit:
                o.append(Order("RAINFOREST_RESIN", 10002, -(limit + pos)))
            orders["RAINFOREST_RESIN"] = o

        # Trading of KELP
        d = state.order_depths.get("KELP")
        if d and d.buy_orders and d.sell_orders:
            pos = state.position.get("KELP", 0)
            limit = POSITION_LIMITS["KELP"]
            best_bid = max(d.buy_orders)
            best_ask = min(d.sell_orders)
            mid = (best_bid + best_ask) / 2
            self.price_history["KELP"].append(mid)
            if len(self.price_history["KELP"]) > 20:
                self.price_history["KELP"] = self.price_history["KELP"][-20:]
            spread = best_ask - best_bid
            o = []
            if spread > 2:
                if pos < limit:
                    o.append(Order("KELP", best_bid + 1, min(6, limit - pos)))
                if pos > -limit:
                    o.append(Order("KELP", best_ask - 1, -min(6, limit + pos)))
            orders["KELP"] = o

        # === BASKET 2 ===
        b2 = state.order_depths.get("PICNIC_BASKET2")
        c = state.order_depths.get("CROISSANTS")
        j = state.order_depths.get("JAMS")
        if b2 and c and j and c.buy_orders and c.sell_orders and j.buy_orders and j.sell_orders:
            c_mid = self.get_mid(c)
            j_mid = self.get_mid(j)
            if c_mid and j_mid:
                fair = 4 * c_mid + 2 * j_mid
                pos = state.position.get("PICNIC_BASKET2", 0)
                limit = POSITION_LIMITS["PICNIC_BASKET2"]
                best_bid = max(b2.buy_orders) if b2.buy_orders else None
                best_ask = min(b2.sell_orders) if b2.sell_orders else None
                o = []
                if best_ask and fair - best_ask > 4 and pos < limit:
                    vol = min(b2.sell_orders[best_ask], limit - pos, 20)
                    o.append(Order("PICNIC_BASKET2", best_ask, vol))
                if best_bid and best_bid - fair > 4 and pos > -limit:
                    vol = min(b2.buy_orders[best_bid], limit + pos, 20)
                    o.append(Order("PICNIC_BASKET2", best_bid, -vol))
                orders["PICNIC_BASKET2"] = o

        # === BASKET 1 ===
        b1 = state.order_depths.get("PICNIC_BASKET1")
        djem = state.order_depths.get("DJEMBES")
        if b1 and djem and djem.buy_orders and djem.sell_orders:
            djem_mid = self.get_mid(djem)
            if djem_mid:
                c_mid = self.get_mid(c)
                j_mid = self.get_mid(j)
                if c_mid and j_mid:
                    fair = 6 * c_mid + 3 * j_mid + 1 * djem_mid
                    pos = state.position.get("PICNIC_BASKET1", 0)
                    limit = POSITION_LIMITS["PICNIC_BASKET1"]
                    best_bid = max(b1.buy_orders) if b1.buy_orders else None
                    best_ask = min(b1.sell_orders) if b1.sell_orders else None
                    o = []
                    if best_ask and fair - best_ask > 10 and pos < limit:
                        o.append(Order("PICNIC_BASKET1", best_ask, min(10, limit - pos)))
                    if best_bid and best_bid - fair > 10 and pos > -limit:
                        o.append(Order("PICNIC_BASKET1", best_bid, -min(10, limit + pos)))
                    orders["PICNIC_BASKET1"] = o

        return orders, conversions, ""
