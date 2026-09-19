"""Shared activation bounds checking for executors.

Determines whether the current price is close enough to the target
order price to justify placing the order — improving capital efficiency.
"""

from __future__ import annotations

from decimal import Decimal

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType


class ActivationBoundsMixin:
    """Mixin providing activation bounds checking.

    Checks whether the current market price is within configured
    activation bounds of the target order price.  Supports both
    limit-type orders (one-sided check) and market-type orders
    (two-sided range check).

    The executor must have:
      - self.config.activation_bounds  (Optional[List[Decimal]])
      - self.config.connector_name     (str)
      - self.config.trading_pair       (str)
      - self.get_price(connector, pair, price_type) method

    Usage:
        class MyExecutor(ActivationBoundsMixin, ExecutorBase):
            def control_open_order(self):
                if self._is_within_activation_bounds(
                    self.config.entry_price,
                    self.config.side,
                    self.config.triple_barrier_config.open_order_type,
                ):
                    self.place_open_order()
    """

    def _is_within_activation_bounds(self, order_price: Decimal, side: TradeType, order_type: OrderType) -> bool:
        """Check if current price is within activation bounds of order_price.

        :param order_price: The target order price.
        :param side: TradeType.BUY or TradeType.SELL.
        :param order_type: The order type (limit vs market determines check style).
        :return: True if within bounds (or no bounds configured).
        """
        activation_bounds = self.config.activation_bounds
        mid_price = self.get_price(self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)
        if activation_bounds:
            if order_type.is_limit_type():
                if side == TradeType.BUY:
                    return order_price >= mid_price * (1 - activation_bounds[0])
                else:
                    return order_price <= mid_price * (1 + activation_bounds[0])
            else:
                if side == TradeType.BUY:
                    min_price_to_buy = order_price * (1 - activation_bounds[0])
                    max_price_to_buy = order_price * (1 + activation_bounds[1])
                    return min_price_to_buy <= mid_price <= max_price_to_buy
                else:
                    min_price_to_sell = order_price * (1 - activation_bounds[1])
                    max_price_to_sell = order_price * (1 + activation_bounds[0])
                    return min_price_to_sell <= mid_price <= max_price_to_sell
        else:
            return True
