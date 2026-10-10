"""Shared order tracking logic for executors.

Provides update_tracked_orders_with_order_id() which fetches the
InFlightOrder from the connector and assigns it to the matching
TrackedOrder.  Each executor defines _get_trackable_orders() to
return the orders that should be searched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hummingbot.strategy_v2.models.executors import TrackedOrder


class OrderTrackingMixin:
    """Mixin providing order tracking via InFlightOrder lookup.

    Executors override _get_trackable_orders() to return the list of
    TrackedOrder instances that should be searched when an order event
    arrives.

    Usage:
        class MyExecutor(OrderTrackingMixin, ExecutorBase):
            def _get_trackable_orders(self) -> List[TrackedOrder]:
                return self._open_orders + self._close_orders

    The mixin also initialises _failed_orders in init_order_tracking().
    """

    def init_order_tracking(self) -> None:
        """Initialize order tracking state. Call from __init__ after super().__init__."""
        self._failed_orders: list[TrackedOrder] = []

    def _get_trackable_orders(self) -> list[TrackedOrder | None]:
        """Return all tracked orders to search through.

        Override in each executor to provide the relevant orders.
        Can include None entries — they are filtered out automatically.
        """
        raise NotImplementedError

    def update_tracked_orders_with_order_id(self, order_id: str) -> None:
        """Fetch InFlightOrder from connector and assign to matching TrackedOrder.

        Searches through _get_trackable_orders() for an order matching
        the given order_id, then fetches the InFlightOrder and assigns it.
        """
        all_orders = self._get_trackable_orders()
        active_order = next(
            (order for order in all_orders if order and order.order_id == order_id),
            None,
        )
        if active_order:
            in_flight_order = self.get_in_flight_order(self.config.connector_name, order_id)
            if in_flight_order:
                active_order.order = in_flight_order
