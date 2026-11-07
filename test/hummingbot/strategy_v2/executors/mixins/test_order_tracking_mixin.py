"""Tests for OrderTrackingMixin."""

from unittest.mock import MagicMock

from hummingbot.strategy_v2.executors.mixins.order_tracking import OrderTrackingMixin
from hummingbot.strategy_v2.models.executors import TrackedOrder


class MockExecutorWithOrderTracking(OrderTrackingMixin):
    """Mock executor with order tracking mixin."""

    def __init__(self):
        self.config = MagicMock()
        self.config.connector_name = "binance"
        self.get_in_flight_order = MagicMock()
        self._open_orders = []
        self._close_orders = []
        self.init_order_tracking()

    def _get_trackable_orders(self):
        return self._open_orders + self._close_orders


class TestOrderTrackingMixin:
    def test_init_order_tracking_creates_failed_orders_list(self):
        executor = MockExecutorWithOrderTracking()
        assert executor._failed_orders == []
        assert isinstance(executor._failed_orders, list)

    def test_get_trackable_orders_not_implemented(self):
        """Base _get_trackable_orders raises NotImplementedError."""
        mixin = OrderTrackingMixin()
        mixin.init_order_tracking()
        try:
            mixin._get_trackable_orders()
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass

    def test_update_tracked_orders_assigns_in_flight_order(self):
        executor = MockExecutorWithOrderTracking()
        tracked = TrackedOrder(order_id="order_123")
        executor._open_orders.append(tracked)

        mock_in_flight = MagicMock()
        executor.get_in_flight_order.return_value = mock_in_flight

        executor.update_tracked_orders_with_order_id("order_123")

        executor.get_in_flight_order.assert_called_once_with("binance", "order_123")
        assert tracked.order == mock_in_flight

    def test_update_tracked_orders_no_match_does_nothing(self):
        executor = MockExecutorWithOrderTracking()
        tracked = TrackedOrder(order_id="order_123")
        executor._open_orders.append(tracked)

        executor.update_tracked_orders_with_order_id("order_999")

        executor.get_in_flight_order.assert_not_called()
        assert tracked.order is None

    def test_update_tracked_orders_no_in_flight_order(self):
        executor = MockExecutorWithOrderTracking()
        tracked = TrackedOrder(order_id="order_123")
        executor._open_orders.append(tracked)

        executor.get_in_flight_order.return_value = None

        executor.update_tracked_orders_with_order_id("order_123")

        executor.get_in_flight_order.assert_called_once_with("binance", "order_123")
        assert tracked.order is None

    def test_update_tracked_orders_searches_close_orders_too(self):
        executor = MockExecutorWithOrderTracking()
        open_order = TrackedOrder(order_id="open_1")
        close_order = TrackedOrder(order_id="close_1")
        executor._open_orders.append(open_order)
        executor._close_orders.append(close_order)

        mock_in_flight = MagicMock()
        executor.get_in_flight_order.return_value = mock_in_flight

        executor.update_tracked_orders_with_order_id("close_1")

        assert close_order.order == mock_in_flight
        assert open_order.order is None

    def test_update_tracked_orders_handles_none_entries(self):
        """_get_trackable_orders can return None entries safely."""
        executor = MockExecutorWithOrderTracking()
        tracked = TrackedOrder(order_id="order_123")
        executor._open_orders = [None, tracked, None]

        mock_in_flight = MagicMock()
        executor.get_in_flight_order.return_value = mock_in_flight

        executor.update_tracked_orders_with_order_id("order_123")

        assert tracked.order == mock_in_flight

    def test_failed_orders_can_be_appended(self):
        executor = MockExecutorWithOrderTracking()
        tracked = TrackedOrder(order_id="failed_1")
        executor._failed_orders.append(tracked)
        assert len(executor._failed_orders) == 1
        assert executor._failed_orders[0].order_id == "failed_1"
